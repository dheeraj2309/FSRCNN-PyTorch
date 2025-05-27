import os
import cv2
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from natsort import natsorted # For sorting filenames naturally

# --- Import your project-specific modules ---
try:
    from model import FSRCNN
    import imgproc
    import config # Try to import config for UPSCALE_FACTOR and DEVICE
    UPSCALE_FACTOR = config.UPSCALE_FACTOR
    DEVICE = config.DEVICE
    print(f"Loaded UPSCALE_FACTOR ({UPSCALE_FACTOR}) and DEVICE ({DEVICE}) from config.py")
except ImportError:
    print("Warning: Could not import FSRCNN, imgproc, or config.py. "
          "Ensure model.py and imgproc.py are in the current directory. "
          "UPSCALE_FACTOR and DEVICE will be set to defaults if config.py is missing.")
    UPSCALE_FACTOR = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration for Inference ---

# 1. PATH TO YOUR BEST TRAINED MODEL
MODEL_NAME_FROM_TRAINING = f"FSRCNN_x{UPSCALE_FACTOR}_custom" # Should match EXP_NAME
MODEL_WEIGHTS_PATH = f"/kaggle/working/results/{MODEL_NAME_FROM_TRAINING}/best.pth.tar"

# 2. DIRECTORIES FOR NEW DATASET
# Replace 'your-new-dataset-slug' and paths accordingly.
NEW_DATASET_BASE_DIR = "/kaggle/input/set5test/Set5/" # MODIFY THIS
NEW_LR_SUBDIR = "downx4/"  # Subdirectory for LR images within NEW_DATASET_BASE_DIR
NEW_HR_SUBDIR = "Set5/"  # Subdirectory for HR images within NEW_DATASET_BASE_DIR

NEW_LR_DATASET_DIR = os.path.join(NEW_DATASET_BASE_DIR, NEW_LR_SUBDIR)
NEW_HR_DATASET_DIR = os.path.join(NEW_DATASET_BASE_DIR, NEW_HR_SUBDIR) # Will be used if available

# 3. DIRECTORY TO SAVE SUPER-RESOLVED (SR) OUTPUT IMAGES
SR_OUTPUT_DIR = f"/kaggle/working/sr_output_on_new_dataset_{MODEL_NAME_FROM_TRAINING}/"

# 4. INFERENCE OPTIONS
USE_HALF_PRECISION = True if DEVICE.type == 'cuda' else False
EVALUATE_PERFORMANCE = True  # Set to True to calculate PSNR/SSIM against HR images
USE_HR_COLOR_CHANNELS = True # Set to True to use color from HR images (better quality)
                               # If False, will use upsampled LR color.
                               # This is only effective if HR images are found.

# --- Helper function for evaluation ---
def calculate_metrics(img_true_y, img_pred_y):
    """Calculates PSNR and SSIM on the Y channel (luminance).
    Assumes images are numpy arrays, Y channel, range [0, 1] or [0, 255].
    """
    # Ensure images are in the range [0, 255] for skimage metrics if they are not already
    if img_true_y.max() <= 1.0:
        img_true_y = (img_true_y * 255.0).astype(np.uint8)
    if img_pred_y.max() <= 1.0:
        img_pred_y = (img_pred_y * 255.0).astype(np.uint8)

    current_psnr = psnr(img_true_y, img_pred_y, data_range=255)
    current_ssim = ssim(img_true_y, img_pred_y, data_range=255, channel_axis=None, win_size=7) # Adjusted for single channel Y
    return current_psnr, current_ssim

def main():
    global EVALUATE_PERFORMANCE, USE_HR_COLOR_CHANNELS # Allow modification
    print(f"Starting inference with the following settings:")
    print(f"  Model Weights: {MODEL_WEIGHTS_PATH}")
    print(f"  New LR Dataset: {NEW_LR_DATASET_DIR}")
    print(f"  New HR Dataset (for color/eval): {NEW_HR_DATASET_DIR}")
    print(f"  SR Output Directory: {SR_OUTPUT_DIR}")
    print(f"  Upscale Factor: {UPSCALE_FACTOR}")
    print(f"  Device: {DEVICE}")
    print(f"  Use Half Precision: {USE_HALF_PRECISION}")
    print(f"  Evaluate Performance (PSNR/SSIM): {EVALUATE_PERFORMANCE}")
    print(f"  Use HR Color Channels: {USE_HR_COLOR_CHANNELS}")

    # --- Sanity Checks ---
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"ERROR: Model weights not found at '{MODEL_WEIGHTS_PATH}'")
        return
    if not os.path.isdir(NEW_LR_DATASET_DIR):
        print(f"ERROR: New LR dataset directory not found at '{NEW_LR_DATASET_DIR}'")
        return
    if (EVALUATE_PERFORMANCE or USE_HR_COLOR_CHANNELS) and not os.path.isdir(NEW_HR_DATASET_DIR):
        print(f"WARNING: New HR dataset directory not found at '{NEW_HR_DATASET_DIR}'. "
              "Will proceed without HR color channels and performance evaluation.")
        EVALUATE_PERFORMANCE = False
        USE_HR_COLOR_CHANNELS = False


    # --- Create Output Directory ---
    os.makedirs(SR_OUTPUT_DIR, exist_ok=True)
    print(f"Created output directory: {SR_OUTPUT_DIR}")

    # --- Load Model ---
    print("Loading FSRCNN model...")
    model = FSRCNN(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    try:
        checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=lambda storage, loc: storage)
        if "state_dict" in checkpoint: model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint: model.load_state_dict(checkpoint["model_state_dict"])
        else: model.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights from '{MODEL_WEIGHTS_PATH}'.")
    except Exception as e:
        print(f"ERROR: Failed to load model weights: {e}")
        return
    model.eval()
    if USE_HALF_PRECISION:
        model.half()
        print("Using half-precision for inference.")

    # --- Process Images ---
    lr_image_files = natsorted([
        f for f in os.listdir(NEW_LR_DATASET_DIR)
        if os.path.isfile(os.path.join(NEW_LR_DATASET_DIR, f))
           and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if not lr_image_files:
        print(f"No image files found in '{NEW_LR_DATASET_DIR}'.")
        return

    print(f"Found {len(lr_image_files)} LR images to process.")
    total_psnr = 0.0
    total_ssim = 0.0
    processed_count = 0

    for image_filename in lr_image_files:
        lr_image_path = os.path.join(NEW_LR_DATASET_DIR, image_filename)
        hr_image_path = os.path.join(NEW_HR_DATASET_DIR, image_filename) # Assumes HR has same name
        sr_image_path = os.path.join(SR_OUTPUT_DIR, image_filename)

        print(f"Processing: {lr_image_path} -> {sr_image_path}")

        try:
            lr_bgr_image = cv2.imread(lr_image_path)
            if lr_bgr_image is None:
                print(f"Warning: Could not read LR image {lr_image_path}. Skipping.")
                continue
            lr_bgr_image_float = lr_bgr_image.astype(np.float32) / 255.0
            lr_y_image = imgproc.bgr2ycbcr(lr_bgr_image_float, use_y_channel=True)
            lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=USE_HALF_PRECISION)
            lr_y_tensor = lr_y_tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                sr_y_tensor = model(lr_y_tensor)

            sr_y_tensor = sr_y_tensor.float().clamp_(0.0, 1.0)
            sr_y_image_numpy = imgproc.tensor2image(sr_y_tensor.cpu(), range_norm=False, half=False)
            sr_y_image_numpy_float = sr_y_image_numpy.astype(np.float32) / 255.0 # Y channel, range [0,1]

            # --- Handle HR image for color and/or evaluation ---
            hr_bgr_image_float = None
            hr_y_image_float = None
            hr_available = False

            if os.path.exists(hr_image_path):
                hr_bgr_image = cv2.imread(hr_image_path)
                if hr_bgr_image is not None:
                    hr_available = True
                    hr_bgr_image_float = hr_bgr_image.astype(np.float32) / 255.0
                    if EVALUATE_PERFORMANCE:
                        hr_y_image_float = imgproc.bgr2ycbcr(hr_bgr_image_float, use_y_channel=True)
                else:
                    print(f"Warning: Could read HR image {hr_image_path}, but it's empty. Proceeding without HR.")
            elif EVALUATE_PERFORMANCE or USE_HR_COLOR_CHANNELS:
                 print(f"Warning: Corresponding HR image not found for {image_filename} at {hr_image_path}. "
                       "Cannot use HR color or evaluate for this image.")


            # --- Reconstruct Color SR Image ---
            target_h, target_w = sr_y_image_numpy_float.shape[:2]

            if USE_HR_COLOR_CHANNELS and hr_available and hr_bgr_image_float is not None:
                print(f"  Using color channels from HR image: {hr_image_path}")
                _, hr_cb, hr_cr = cv2.split(imgproc.bgr2ycbcr(hr_bgr_image_float, use_y_channel=False))
                # Resize HR's Cb, Cr to match SR Y's dimensions if necessary (should be close)
                if hr_cb.shape[0] != target_h or hr_cb.shape[1] != target_w:
                    color_cb = cv2.resize(hr_cb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                    color_cr = cv2.resize(hr_cr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                else:
                    color_cb = hr_cb
                    color_cr = hr_cr
            else:
                if USE_HR_COLOR_CHANNELS and not hr_available:
                     print(f"  HR color requested but HR image not found. Falling back to upsampled LR color.")
                print(f"  Using upsampled color channels from LR image: {lr_image_path}")
                _, lr_cb, lr_cr = cv2.split(imgproc.bgr2ycbcr(lr_bgr_image_float, use_y_channel=False))
                color_cb = cv2.resize(lr_cb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                color_cr = cv2.resize(lr_cr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            sr_ycbcr_image = cv2.merge([sr_y_image_numpy_float, color_cb, color_cr])
            sr_bgr_output_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
            output_image_to_save = np.clip(sr_bgr_output_image * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(sr_image_path, output_image_to_save)

            # --- Evaluate Performance if HR is available ---
            if EVALUATE_PERFORMANCE and hr_available and hr_y_image_float is not None:
                # Crop HR Y to match SR Y dimensions for fair comparison
                h_sr, w_sr = sr_y_image_numpy_float.shape[:2]
                hr_y_cropped = hr_y_image_float[:h_sr, :w_sr]

                current_psnr, current_ssim = calculate_metrics(hr_y_cropped, sr_y_image_numpy_float)
                print(f"  Metrics for {image_filename}: PSNR: {current_psnr:.2f}dB, SSIM: {current_ssim:.4f}")
                total_psnr += current_psnr
                total_ssim += current_ssim
                processed_count += 1

        except Exception as e:
            print(f"Error processing image {lr_image_path}: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 30)
    print(f"Inference complete. Super-resolved images saved to '{SR_OUTPUT_DIR}'.")

    if EVALUATE_PERFORMANCE and processed_count > 0:
        avg_psnr = total_psnr / processed_count
        avg_ssim = total_ssim / processed_count
        print(f"\nAverage Performance Metrics over {processed_count} images:")
        print(f"  Average PSNR: {avg_psnr:.2f}dB")
        print(f"  Average SSIM: {avg_ssim:.4f}")
    elif EVALUATE_PERFORMANCE:
        print("\nNo images were successfully processed with corresponding HR images for evaluation.")

if __name__ == "__main__":
    # Ensure skimage is available if evaluating
    if EVALUATE_PERFORMANCE:
        try:
            import skimage
        except ImportError:
            print("ERROR: scikit-image is not installed. Please install it for PSNR/SSIM evaluation: pip install scikit-image")
            print("Alternatively, set EVALUATE_PERFORMANCE = False in the script.")
            exit()
    main()