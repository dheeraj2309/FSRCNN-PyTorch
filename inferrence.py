# inference.py
import os
import cv2
import numpy as np
import torch
from natsort import natsorted
from skimage.metrics import structural_similarity

# --- Helper Imports (Ensure model.py and imgproc.py are in the same directory) ---
try:
    from model import FSRCNN  # From your provided model.py
    import imgproc            # Assumed to be present from the original project
except ImportError as e:
    print(f"Error importing helper modules: {e}")
    print("Please ensure 'model.py' (provided) and 'imgproc.py' (from the original project) "
          "are in the same directory as this inference.py script.")
    exit(1)

# --- Configuration ---
# TODO: Update these paths according to your environment
MODEL_PATH = "fsrcnn_x4-T91-97a30bfb.pth.tar"  # Path to your .pth.tar model file
UPSCALE_FACTOR = 4  # From model name "fsrcnn_x4..."

# Example Kaggle paths (modify if needed):
# LR_IMAGE_DIR = "/kaggle/input/your-dataset-name/lr_images"
# HR_IMAGE_DIR = "/kaggle/input/your-dataset-name/hr_images"
# SR_OUTPUT_DIR = "/kaggle/working/sr_output_images"

# Example local paths (MODIFY THESE):
LR_IMAGE_DIR = "./data/Set5/LRbicx4"
HR_IMAGE_DIR = "./data/Set5/HR"
SR_OUTPUT_DIR = "./results/Set5_FSRCNN_x4_inference"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_SR_IMAGES = True  # Set to False if you don't want to save Super-Resolved images
USE_HALF_PRECISION = False # Set to True for potential speedup on CUDA if your model/GPU supports it well

def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return
    if not os.path.isdir(LR_IMAGE_DIR):
        print(f"Error: LR image directory not found at '{LR_IMAGE_DIR}'")
        return
    if not os.path.isdir(HR_IMAGE_DIR):
        print(f"Error: HR image directory not found at '{HR_IMAGE_DIR}'")
        return

    if SAVE_SR_IMAGES:
        os.makedirs(SR_OUTPUT_DIR, exist_ok=True)
        print(f"SR images will be saved to: {os.path.abspath(SR_OUTPUT_DIR)}")

    # Initialize the FSRCNN model
    model = FSRCNN(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    print(f"Built FSRCNN model (x{UPSCALE_FACTOR}) successfully. Device: {DEVICE}")

    # Load the model weights
    print(f"Loading model weights from '{os.path.abspath(MODEL_PATH)}'...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Determine how to load state_dict based on checkpoint structure
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
        # Remove 'module.' prefix if model was saved with DataParallel
        if all(key.startswith("module.") for key in model_state_dict.keys()):
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        print("Model weights loaded successfully from checkpoint['state_dict'].")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint: # another common key
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully from checkpoint['model_state_dict'].")
    elif isinstance(checkpoint, dict) and "model" in checkpoint: # ESRGAN-style
         model.load_state_dict(checkpoint["model"])
         print("Model weights loaded successfully from checkpoint['model'].")
    else: # Assume the checkpoint *is* the state_dict itself
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully (assumed checkpoint is the state_dict itself).")

    model.eval()
    if USE_HALF_PRECISION and DEVICE.type == 'cuda':
        model.half()
        print("Using half precision for inference.")

    total_psnr_y = 0.0
    total_ssim_y = 0.0
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    lr_image_files = natsorted([
        os.path.join(LR_IMAGE_DIR, f) for f in os.listdir(LR_IMAGE_DIR) 
        if os.path.isfile(os.path.join(LR_IMAGE_DIR, f)) and f.lower().endswith(image_extensions)
    ])
    
    if not lr_image_files:
        print(f"No LR images found in '{LR_IMAGE_DIR}' with extensions {image_extensions}")
        return

    num_files_processed = 0

    for lr_image_path in lr_image_files:
        base_name = os.path.basename(lr_image_path)
        hr_image_path = os.path.join(HR_IMAGE_DIR, base_name)
        
        if SAVE_SR_IMAGES:
            sr_image_output_path = os.path.join(SR_OUTPUT_DIR, base_name)
        else:
            sr_image_output_path = None # Not saving

        if not os.path.exists(hr_image_path):
            print(f"Warning: Corresponding HR image not found for '{lr_image_path}' at '{hr_image_path}'. Skipping.")
            continue

        print(f"Processing: {base_name}")
        
        # Load images as BGR, convert to float32 [0, 1]
        try:
            lr_image_bgr = cv2.imread(lr_image_path).astype(np.float32) / 255.0
            hr_image_bgr = cv2.imread(hr_image_path).astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error reading image {base_name}: {e}. Skipping.")
            continue

        if lr_image_bgr is None:
            print(f"Warning: Could not read LR image '{lr_image_path}'. Skipping.")
            continue
        if hr_image_bgr is None:
            print(f"Warning: Could not read HR image '{hr_image_path}'. Skipping.")
            continue
        
        # Extract Y channel (luminance) for FSRCNN processing and metrics
        # imgproc.bgr2ycbcr returns float32 Y channel in range [0, 1]
        lr_y_image = imgproc.bgr2ycbcr(lr_image_bgr, use_y_channel=True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image_bgr, use_y_channel=True)

        # Convert LR Y image to tensor: [N, C, H, W]
        # range_norm=False means input is already [0,1] or [0,255] and imgproc.image2tensor handles it
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False).to(DEVICE).unsqueeze_(0)
        if USE_HALF_PRECISION and DEVICE.type == 'cuda':
            lr_y_tensor = lr_y_tensor.half()
        
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor) # Output is [N, C, H, W]
        
        # Process SR output: convert to float, clamp to [0, 1]
        sr_y_tensor = sr_y_tensor.float().clamp_(0.0, 1.0)

        # Prepare HR Y tensor for comparison
        # Ensure HR Y is on the correct device and also [N, C, H, W]
        hr_y_tensor_comp = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).to(DEVICE).unsqueeze_(0)
        
        # Crop HR to match SR spatial dimensions if necessary
        # FSRCNN output H, W should be upscale_factor * LR H, W.
        # HR images should ideally be this exact size.
        h_sr, w_sr = sr_y_tensor.shape[2], sr_y_tensor.shape[3]
        h_hr, w_hr = hr_y_tensor_comp.shape[2], hr_y_tensor_comp.shape[3]

        if h_hr != h_sr or w_hr != w_sr:
            # print(f"  Note: HR Y-channel dimensions ({h_hr}x{w_hr}) differ from SR Y-channel ({h_sr}x{w_sr}). Cropping HR for metrics.")
            hr_y_tensor_comp = hr_y_tensor_comp[:, :, :h_sr, :w_sr]
            
        # --- Calculate PSNR on Y channel ---
        mse_y = torch.mean((sr_y_tensor - hr_y_tensor_comp) ** 2)
        if mse_y.item() == 0: # Avoid log(0)
            current_psnr_y = float('inf')
        else:
            current_psnr_y = 10. * torch.log10(1.0 / mse_y).item()
        total_psnr_y += current_psnr_y
        
        # --- Calculate SSIM on Y channel ---
        # Convert Y-channel tensors to NumPy arrays: [H, W], range [0, 1]
        sr_y_numpy = sr_y_tensor.squeeze().cpu().numpy() # Squeeze N and C dimensions
        hr_y_numpy_comp = hr_y_tensor_comp.squeeze().cpu().numpy()
        
        # Ensure images are properly clipped (although sr_y_tensor was already clamped)
        sr_y_numpy = np.clip(sr_y_numpy, 0.0, 1.0)
        hr_y_numpy_comp = np.clip(hr_y_numpy_comp, 0.0, 1.0)
        
        # Determine a robust win_size for SSIM
        win_s = min(7, sr_y_numpy.shape[0], sr_y_numpy.shape[1])
        if win_s % 2 == 0: win_s -= 1 # Ensure win_size is odd
        
        current_ssim_y = 0.0
        if win_s >= 1:
            try:
                current_ssim_y = structural_similarity(
                    hr_y_numpy_comp, sr_y_numpy, 
                    win_size=win_s, 
                    data_range=1.0,
                    gaussian_weights=True,
                    sigma=1.5, # Standard for Gaussian window
                    use_sample_covariance=False # Recommended for skimage >= 0.16
                )
            except ValueError as e: # E.g., win_size too large for image dimensions
                 print(f"  Warning: Could not calculate SSIM for {base_name} (win_size={win_s}, shape={sr_y_numpy.shape}): {e}. Setting SSIM to 0.")
        else: # win_size is < 1, e.g. for very small images
            print(f"  Warning: SSIM win_size for {base_name} is < 1 ({win_s}). Setting SSIM to 0.")
        total_ssim_y += current_ssim_y
        
        num_files_processed += 1
        print(f"  Metrics (Y-channel): PSNR: {current_psnr_y:6.2f} dB, SSIM: {current_ssim_y:7.4f}")

        # --- Save Super-Resolved image (reconstruct with Cb,Cr channels) ---
        if SAVE_SR_IMAGES and sr_image_output_path:
            # Convert SR Y-channel tensor back to uint8 NumPy image [0, 255] then float [0,1]
            sr_y_image_for_save = imgproc.tensor2image(sr_y_tensor.cpu(), range_norm=False, half=False) 
            sr_y_image_float_for_save = sr_y_image_for_save.astype(np.float32) / 255.0

            # Get Cb, Cr channels from the original HR image (usually best quality)
            # These are already float32 [0,1] from earlier hr_image_bgr
            _, hr_cb_orig, hr_cr_orig = cv2.split(imgproc.bgr2ycbcr(hr_image_bgr, use_y_channel=False))
            
            # Resize Cb, Cr from HR to match SR Y channel's dimensions
            target_h, target_w = sr_y_image_float_for_save.shape[:2]
            if hr_cb_orig.shape[0] != target_h or hr_cb_orig.shape[1] != target_w:
                hr_cb_resized = cv2.resize(hr_cb_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                hr_cr_resized = cv2.resize(hr_cr_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            else:
                hr_cb_resized = hr_cb_orig
                hr_cr_resized = hr_cr_orig
                
            # Merge SR Y-channel with HR Cb,Cr
            sr_ycbcr_image = cv2.merge([sr_y_image_float_for_save, hr_cb_resized, hr_cr_resized])
            # Convert YCbCr [0,1] to BGR [0,1]
            sr_bgr_image_final = imgproc.ycbcr2bgr(sr_ycbcr_image) 
            # Convert to BGR uint8 [0,255] for saving
            sr_bgr_image_to_save = np.clip(sr_bgr_image_final * 255.0, 0, 255).astype(np.uint8)
            
            cv2.imwrite(sr_image_output_path, sr_bgr_image_to_save)
            # print(f"  Saved SR image to: {sr_image_output_path}")

    # --- Print Average Metrics ---
    if num_files_processed > 0:
        avg_psnr_y = total_psnr_y / num_files_processed
        avg_ssim_y = total_ssim_y / num_files_processed
        print("-" * 50)
        print(f"Inference Summary ({num_files_processed} images):")
        print(f"  Average PSNR (Y-channel): {avg_psnr_y:6.2f} dB")
        print(f"  Average SSIM (Y-channel): {avg_ssim_y:7.4f}")
        print("-" * 50)
    else:
        print("\nNo images were processed successfully. Check paths and image files.")

if __name__ == "__main__":
    # Basic check for helper modules before running main logic
    if 'FSRCNN' not in globals() or 'imgproc' not in globals():
        print("Exiting due to missing FSRCNN class or imgproc module. "
              "Ensure model.py and imgproc.py are correctly placed.")
    else:
        main()