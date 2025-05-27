# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""File description: Realize the verification function after model training."""
import os

import cv2
import numpy as np
import torch
from natsort import natsorted

import config
import imgproc
from model import FSRCNN


def main() -> None:
    # Initialize the super-resolution model
    os.makedirs(config.SR_TEST_OUTPUT_DIR, exist_ok=True)
    model = FSRCNN(config.UPSCALE_FACTOR).to(config.DEVICE)
    print("Build FSRCNN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.VALIDATE_MODEL_PATH, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load FSRCNN model weights `{os.path.abspath(config.VALIDATE_MODEL_PATH)}` successfully.")

    # Create a folder of super-resolution experiment results
    model.eval()
    use_half_precision_inference = config.DEVICE.type == 'cuda' # Can choose to use half for inference
    if use_half_precision_inference:
        model.half()

    total_psnr = 0.0
    
    # Use config.TEST_LR_DIR and config.TEST_HR_DIR
    lr_image_files = natsorted([os.path.join(config.TEST_LR_DIR, f) for f in os.listdir(config.TEST_LR_DIR) if os.path.isfile(os.path.join(config.TEST_LR_DIR, f))])
    
    if not lr_image_files:
        print(f"No LR images found in {config.TEST_LR_DIR}")
        return

    total_files_processed = 0

    for lr_image_path in lr_image_files:
        base_name = os.path.basename(lr_image_path)
        hr_image_path = os.path.join(config.TEST_HR_DIR, base_name) # Assumes HR has same name
        sr_image_path = os.path.join(config.SR_TEST_OUTPUT_DIR, base_name)

        if not os.path.exists(hr_image_path):
            print(f"Warning: Corresponding HR image not found for {lr_image_path}. Skipping.")
            continue

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_image_bgr = cv2.imread(lr_image_path).astype(np.float32) / 255.0
        hr_image_bgr = cv2.imread(hr_image_path).astype(np.float32) / 255.0

        if lr_image_bgr is None:
            print(f"Warning: Could not read LR image {lr_image_path}. Skipping.")
            continue
        if hr_image_bgr is None:
            print(f"Warning: Could not read HR image {hr_image_path}. Skipping.")
            continue
        
        # Extract Y channel for processing
        lr_y_image = imgproc.bgr2ycbcr(lr_image_bgr, use_y_channel=True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image_bgr, use_y_channel=True)

        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=use_half_precision_inference).to(config.DEVICE).unsqueeze_(0)
        
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor)
        
        # Ensure sr_y_tensor is float for clamping, PSNR, and saving operations
        sr_y_tensor = sr_y_tensor.float().clamp_(0.0, 1.0)

        # Calculate PSNR on Y channel
        hr_y_tensor_orig = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).to(config.DEVICE).unsqueeze_(0)
        
        # sr_y_tensor is the output of the model, already float and clamped.
        # Get SR spatial dimensions (H, W are at index 2 and 3 for a [N,C,H,W] tensor)
        h_sr, w_sr = sr_y_tensor.shape[2], sr_y_tensor.shape[3]

        # Crop original hr_y_tensor_orig to match sr_y_tensor's spatial dimensions
        if hr_y_tensor_orig.shape[2] != h_sr or hr_y_tensor_orig.shape[3] != w_sr:
            # print(f"Warning: Cropping HR Y-tensor ({hr_y_tensor_orig.shape}) to SR Y-tensor dimensions ({sr_y_tensor.shape}) for PSNR.")
            hr_y_tensor_cropped = hr_y_tensor_orig[:, :, :h_sr, :w_sr]
        else:
            hr_y_tensor_cropped = hr_y_tensor_orig # No cropping needed
            
        mse = torch.mean((sr_y_tensor - hr_y_tensor_cropped) ** 2) # Use cropped HR Y-tensor
        if mse.item() == 0:
            current_psnr = float('inf')
        else:
            psnr_tensor = 10. * torch.log10(1.0 / mse) # 1.0 / mse results in a tensor
            current_psnr = psnr_tensor.item()
        total_psnr += current_psnr
        total_files_processed +=1
        print(f"PSNR for {base_name}: {current_psnr:4.2f}dB")

        # Save SR image (reconstruct with Cb,Cr from HR for color output)
        sr_y_image_numpy = imgproc.tensor2image(sr_y_tensor.cpu(), range_norm=False, half=False) # Convert to numpy, ensure full precision
        sr_y_image_numpy_float = sr_y_image_numpy.astype(np.float32) / 255.0
        
        # Get Cb, Cr channels from the original HR image (or upsampled LR if preferred, but HR is usually better)
        # Resize them to match the SR Y channel's dimensions if FSRCNN output size slightly differs
        _, hr_cb_orig, hr_cr_orig = cv2.split(imgproc.bgr2ycbcr(hr_image_bgr, use_y_channel=False))
        
        target_h, target_w = sr_y_image_numpy_float.shape[:2]
        if hr_cb_orig.shape[0] != target_h or hr_cb_orig.shape[1] != target_w:
            hr_cb_resized = cv2.resize(hr_cb_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            hr_cr_resized = cv2.resize(hr_cr_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            hr_cb_resized = hr_cb_orig
            hr_cr_resized = hr_cr_orig
            
        sr_ycbcr_image = cv2.merge([sr_y_image_numpy_float, hr_cb_resized, hr_cr_resized])
        sr_bgr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_bgr_image * 255.0)

    if total_files_processed > 0:
        avg_psnr = total_psnr / total_files_processed
        print(f"\nAverage PSNR over {total_files_processed} images: {avg_psnr:4.2f}dB.")
    else:
        print("\nNo images were processed successfully to calculate PSNR.")


if __name__ == "__main__":
    main()
