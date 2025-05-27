# validate_single_model.py
import os
import cv2
import numpy as np
import torch
from natsort import natsorted
from skimage.metrics import structural_similarity

# --- Helper Imports (Ensure these files are in the same directory or Python path) ---
try:
    from model import FSRCNN # Assumes model.py is in the same directory
    import imgproc          # Assumes imgproc.py is in the same directory
except ImportError as e:
    print(f"Error importing helper modules (model.py, imgproc.py): {e}")
    print("Please ensure model.py and imgproc.py are in the same directory as this script, or in your PYTHONPATH.")
    exit(1)
# --- Configuration ---
# : Update these paths according to your Kaggle environment or local setup
MODEL_PATH = "fsrcnn_x4-T91-97a30bfb.pth.tar"  # Path to your .pth.tar model file
UPSCALE_FACTOR = 4  # The model filename "fsrcnn_x4" suggests x4

# Example Kaggle paths:
# LR_IMAGE_DIR = "/kaggle/input/your-dataset-name/lr_images"
# HR_IMAGE_DIR = "/kaggle/input/your-dataset-name/hr_images"
# SR_OUTPUT_DIR = "/kaggle/working/sr_output_images" # Output in Kaggle's writable directory

# Example local paths:
LR_IMAGE_DIR = "/kaggle/input/updated-dataset/dataset/test/LR" # MODIFY THIS
HR_IMAGE_DIR = "/kaggle/input/updated-dataset/dataset/test/HR"     # MODIFY THIS
SR_OUTPUT_DIR = "/kaggle/working/output_sr" # MODIFY THIS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_SR_IMAGES = True # Set to False if you don't want to save SR images

def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    if not os.path.isdir(LR_IMAGE_DIR):
        print(f"Error: LR image directory not found at {LR_IMAGE_DIR}")
        return
    if not os.path.isdir(HR_IMAGE_DIR):
        print(f"Error: HR image directory not found at {HR_IMAGE_DIR}")
        return

    os.makedirs(SR_OUTPUT_DIR, exist_ok=True)

    # Initialize the super-resolution model
    model = FSRCNN(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    print(f"Building FSRCNN model (x{UPSCALE_FACTOR}) successfully.")

    # Load the super-resolution model weights
    print(f"Loading model weights from `{os.path.abspath(MODEL_PATH)}`...")
    checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

    # Check if the checkpoint is a dict and contains 'state_dict' (common practice)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_state_dict = checkpoint["state_dict"]
        # FSRCNN might save the model directly or within a module (e.g. if DataParallel was used)
        # This attempts to load it correctly if it was saved with a 'module.' prefix
        if all(key.startswith("module.") for key in model_state_dict.keys()):
             model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        print("Model weights loaded successfully from checkpoint['state_dict'].")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint: # another common key
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model weights loaded successfully from checkpoint['model_state_dict'].")
    else: # Assume the checkpoint *is* the state_dict itself
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully (assumed checkpoint is state_dict).")


    model.eval()
    # Optional: Use half-precision for inference on CUDA for potential speedup
    # use_half_precision_inference = DEVICE.type == 'cuda'
    # if use_half_precision_inference:
    #     model.half()

    total_psnr = 0.0
    total_ssim = 0.0
    
    lr_image_files = natsorted([
        os.path.join(LR_IMAGE_DIR, f) for f in os.listdir(LR_IMAGE_DIR) 
        if os.path.isfile(os.path.join(LR_IMAGE_DIR, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])
    
    if not lr_image_files:
        print(f"No LR images found in {LR_IMAGE_DIR}")
        return

    total_files_processed = 0

    for lr_image_path in lr_image_files:
        base_name = os.path.basename(lr_image_path)
        hr_image_path = os.path.join(HR_IMAGE_DIR, base_name)
        sr_image_path = os.path.join(SR_OUTPUT_DIR, base_name)

        if not os.path.exists(hr_image_path):
            print(f"Warning: Corresponding HR image not found for {lr_image_path} at {hr_image_path}. Skipping.")
            continue

        print(f"Processing `{lr_image_path}`...")
        
        # Load images as BGR, convert to float32 [0, 1]
        lr_image_bgr = cv2.imread(lr_image_path).astype(np.float32) / 255.0
        hr_image_bgr = cv2.imread(hr_image_path).astype(np.float32) / 255.0

        if lr_image_bgr is None:
            print(f"Warning: Could not read LR image {lr_image_path}. Skipping.")
            continue
        if hr_image_bgr is None:
            print(f"Warning: Could not read HR image {hr_image_path}. Skipping.")
            continue
        
        # Extract Y channel for processing (FSRCNN typically works on Y channel)
        lr_y_image = imgproc.bgr2ycbcr(lr_image_bgr, use_y_channel=True) # Returns float32 [0,1]
        hr_y_image = imgproc.bgr2ycbcr(hr_image_bgr, use_y_channel=True) # Returns float32 [0,1]

        # Convert LR Y image to tensor
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False).to(DEVICE).unsqueeze_(0)
        # if use_half_precision_inference:
        #     lr_y_tensor = lr_y_tensor.half()
        
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor)
        
        # Ensure sr_y_tensor is float, clamped to [0, 1]
        sr_y_tensor = sr_y_tensor.float().clamp_(0.0, 1.0)

        # Prepare HR Y tensor for comparison (already float32 [0,1])
        hr_y_tensor_orig = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).to(DEVICE).unsqueeze_(0)
        
        # Crop HR to match SR spatial dimensions if necessary (FSRCNN should maintain size)
        h_sr, w_sr = sr_y_tensor.shape[2], sr_y_tensor.shape[3]
        h_hr, w_hr = hr_y_tensor_orig.shape[2], hr_y_tensor_orig.shape[3]

        # FSRCNN output size is (H_lr * upscale, W_lr * upscale).
        # If HR images are exactly that size, no cropping is needed.
        # Otherwise, crop the HR image from the center or top-left to match SR.
        # For simplicity, we'll assume HR is either same size or needs top-left cropping.
        if h_hr != h_sr or w_hr != w_sr:
            print(f"Warning: HR image dimensions ({h_hr}x{w_hr}) differ from SR image dimensions ({h_sr}x{w_sr}) for {base_name}. Cropping HR for metrics.")
            hr_y_tensor_cropped = hr_y_tensor_orig[:, :, :h_sr, :w_sr]
        else:
            hr_y_tensor_cropped = hr_y_tensor_orig
            
        # --- Calculate PSNR on Y channel ---
        mse = torch.mean((sr_y_tensor - hr_y_tensor_cropped) ** 2)
        if mse.item() == 0:
            current_psnr = float('inf')
        else:
            current_psnr = 10. * torch.log10(1.0 / mse).item()
        total_psnr += current_psnr
        
        # --- Calculate SSIM on Y channel ---
        # Convert tensors to numpy arrays, shape [H, W], range [0, 1]
        sr_y_numpy = sr_y_tensor.squeeze(0).squeeze(0).cpu().numpy()
        hr_y_numpy_cropped = hr_y_tensor_cropped.squeeze(0).squeeze(0).cpu().numpy()
        
        # Clip again just in case of numerical precision issues, though clamping sr_y_tensor helps
        sr_y_numpy = np.clip(sr_y_numpy, 0.0, 1.0)
        hr_y_numpy_cropped = np.clip(hr_y_numpy_cropped, 0.0, 1.0)
        
        win_size = min(7, sr_y_numpy.shape[0], sr_y_numpy.shape[1])
        if win_size % 2 == 0: win_size -= 1 # Ensure win_size is odd
        
        current_ssim = 0.0
        if win_size >= 1: # ssim requires win_size >= 1
            try:
                current_ssim = structural_similarity(hr_y_numpy_cropped, sr_y_numpy, 
                                                     data_range=1.0, win_size=win_size,
                                                     gaussian_weights=True, # Often recommended
                                                     sigma=1.5, # Often used with gaussian_weights
                                                     use_sample_covariance=False) # Default for skimage >= 0.16
            except ValueError as e:
                 print(f"Could not calculate SSIM for {base_name} (win_size={win_size}, shape={sr_y_numpy.shape}): {e}. Setting SSIM to 0.")
        else:
            print(f"SSIM win_size for {base_name} is < 1 ({win_size}). Setting SSIM to 0.")
        total_ssim += current_ssim
        
        total_files_processed += 1
        print(f"  Metrics for {base_name}: PSNR: {current_psnr:6.2f} dB, SSIM: {current_ssim:7.4f}")

        # --- Save SR image (reconstruct with Cb,Cr from upscaled LR or original HR) ---
        if SAVE_SR_IMAGES:
            # Convert SR Y channel tensor back to uint8 NumPy image [0, 255]
            sr_y_image_save = imgproc.tensor2image(sr_y_tensor.cpu(), range_norm=False, half=False) 
            sr_y_image_float = sr_y_image_save.astype(np.float32) / 255.0 # Back to float [0,1] for merging

            # Get Cb, Cr channels. For best color quality, use original HR image's CbCr if available and same size.
            # Otherwise, upscale LR's CbCr. Here, we use HR's CbCr.
            _, hr_cb_orig, hr_cr_orig = cv2.split(imgproc.bgr2ycbcr(hr_image_bgr, use_y_channel=False))
            
            # Resize Cb, Cr from HR to match SR Y channel's dimensions
            target_h, target_w = sr_y_image_float.shape[:2] # Should be h_sr, w_sr
            if hr_cb_orig.shape[0] != target_h or hr_cb_orig.shape[1] != target_w:
                hr_cb_resized = cv2.resize(hr_cb_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                hr_cr_resized = cv2.resize(hr_cr_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            else:
                hr_cb_resized = hr_cb_orig
                hr_cr_resized = hr_cr_orig
                
            sr_ycbcr_image = cv2.merge([sr_y_image_float, hr_cb_resized, hr_cr_resized])
            sr_bgr_image = imgproc.ycbcr2bgr(sr_ycbcr_image) # Converts YCbCr [0,1] to BGR [0,1]
            cv2.imwrite(sr_image_path, np.clip(sr_bgr_image * 255.0, 0, 255).astype(np.uint8))
            # print(f"  Saved SR image to {sr_image_path}")

    if total_files_processed > 0:
        avg_psnr = total_psnr / total_files_processed
        avg_ssim = total_ssim / total_files_processed
        print("-" * 40)
        print(f"Validation Summary ({total_files_processed} images):")
        print(f"  Average PSNR: {avg_psnr:6.2f} dB")
        print(f"  Average SSIM: {avg_ssim:7.4f}")
        print("-" * 40)
    else:
        print("\nNo images were processed successfully to calculate PSNR/SSIM.")

if __name__ == "__main__":
    # Verify FSRCNN and imgproc are available before running main
    if 'FSRCNN' not in globals() or 'imgproc' not in globals():
        print("Exiting due to missing FSRCNN or imgproc modules.")
    else:
        main()