# FSRCNN-PyTorch for Custom Super-Resolution Project

**This repository is a fork of [Lornatang/FSRCNN-PyTorch](https://github.com/Lornatang/FSRCNN-PyTorch) and has been modified to work with a custom dataset and experiment with the FSRCNN architecture for an image super-resolution project.**

The core FSRCNN (Fast Super-Resolution Convolutional Neural Network) implementation from the original repository has been adapted to suit specific dataset requirements, and various hyperparameter tuning and architectural adjustments were explored.

## Project Overview

The goal of this project was to:
1.  Understand and implement the FSRCNN model for image super-resolution.
2.  Adapt an existing PyTorch implementation to a custom dataset of Low-Resolution (LR) and High-Resolution (HR) image pairs.
3.  Train the model, evaluate its performance using PSNR and SSIM metrics, and analyze the results.
4.  Document the learning process, challenges faced, and potential areas for improvement.

This project served as a hands-on learning experience in deep learning for image processing, involving understanding existing codebases, debugging, and tailoring models to new data.

## Key Modifications from Original Repository

The primary changes made to the original codebase include:

*   **Dataset Handling (`dataset.py`, `config.py`):**
    *   Modified to load pre-existing LR and HR image pairs directly from specified directories, instead of the original approach which might have involved on-the-fly downsampling for some use cases.
    *   Adjusted to handle minor dimension mismatches between LR and HR pairs (e.g., by cropping HR images to ensure their dimensions are exact multiples of LR dimensions scaled by the `UPSCALE_FACTOR`). This was crucial for accurate metric calculation.
    *   Configuration paths in `config.py` were updated to reflect the custom dataset structure and output directories.
*   **Model Architecture (`model.py`):**
    *   While the core FSRCNN structure (feature extraction, shrinking, mapping, expanding, deconvolution) was retained, parameters within these layers (number of filters, channels in the shrinking/mapping layers, depth of mapping layers, kernel sizes) were experimented with. For instance, one configuration explored involved 64 filters for feature extraction/expansion, 12 channels for the shrinking/mapping stages, and a depth of 6 for the mapping layers.
*   **Training and Validation (`train.py`, `validate.py`):**
    *   Scripts were updated to align with the modified dataset loading.
    *   Calculation and logging of both PSNR and SSIM metrics (on the Y-channel) were ensured.
    *   TensorBoard logging for losses, PSNR, and SSIM was integrated for monitoring.
    *   The checkpointing strategy was maintained to save the best and last models.
*   **Inference (`inferrence.py`):**
    *   A dedicated inference script was adapted to load a trained model, process a directory of LR test images, and save the super-resolved outputs. It also calculates PSNR and SSIM against corresponding HR images.
    *   Color image reconstruction was handled by super-resolving the Y-channel and then merging it with upscaled Cb and Cr channels from the HR image.
*   **Code Refinements:** Various minor adjustments were made throughout the scripts to ensure compatibility with the custom dataset and to address issues encountered during experimentation, such as ensuring correct tensor types and handling potential edge cases in image processing.

## Project Structure

The main Python scripts involved in this modified project are:
*   `model.py`: Defines the FSRCNN neural network architecture.
*   `train.py`: Handles the model training loop, including data loading, optimization, loss calculation, metric evaluation on validation sets, and model checkpointing.
*   `validate.py`: Primarily contains the logic for model evaluation (PSNR, SSIM) which is called from `train.py`. It also forms the basis for the standalone `inferrence.py`.
*   `inferrence.py`: A standalone script for running inference with a pre-trained model on a test dataset and saving the super-resolved images.
*   `dataset.py`: Contains PyTorch `Dataset` classes for loading and preprocessing training, validation, and test image pairs.
*   `imgproc.py`: Provides utility functions for image processing tasks such as color space conversion (BGR to YCbCr and vice-versa), image-to-tensor conversions, and data augmentation (cropping, flipping, rotation).
*   `config.py`: A centralized configuration file for managing dataset paths, model hyperparameters (like `UPSCALE_FACTOR`, `IMAGE_SIZE`), training parameters (`BATCH_SIZE`, `EPOCHS`), and output directories.
*   `setup.py`: Original setup script from the forked repository (may not be directly used for project execution unless packaging the modified version).

## How to Use

1.  **Setup Environment:**
    *   Clone this repository.
    *   Install the necessary dependencies. A `requirements.txt` would be ideal, but common dependencies include:
        ```bash
        pip install torch torchvision torchaudio opencv-python scikit-image natsort numpy tensorboard
        ```

2.  **Prepare Dataset:**
    *   Organize your dataset according to the paths expected in `config.py`. Typically, this involves creating directories for training, validation, and testing, each with `HR` (High-Resolution) and `LR` (Low-Resolution) subdirectories.
    *   Example: `dataset/train/HR`, `dataset/train/LR`, `dataset/test/HR`, `dataset/test/LR`.
    *   Ensure image filenames in `HR` and `LR` correspond to each other.

3.  **Configure (`config.py`):**
    *   Modify `BASE_DATA_PATH` in `config.py` to point to your root dataset directory.
    *   Adjust other parameters like `UPSCALE_FACTOR`, `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `MODEL_LR` as per your requirements.
    *   Verify output paths (`MODEL_SAVE_DIR`, `RESULTS_SAVE_DIR`, `LOGS_DIR`, `SR_TEST_OUTPUT_DIR`).

4.  **Training:**
    *   Execute the training script:
        ```bash
        python train.py
        ```
    *   Training progress, including loss, PSNR, and SSIM, can be monitored using TensorBoard. Launch it by navigating to your project directory and running (adjust log directory if needed):
        ```bash
        tensorboard --logdir samples/logs
        ```
        (Note: The `LOGS_DIR` in `config.py` determines the actual log path.)

5.  **Inference:**
    *   After training, a model (e.g., `best.pth.tar`) will be saved in the directory specified by `RESULTS_SAVE_DIR`.
    *   Update the `MODEL_PATH` in `inferrence.py` to point to your trained model file.
    *   Specify the `LR_IMAGE_DIR`, `HR_IMAGE_DIR` (for metrics), and `SR_OUTPUT_DIR` in `inferrence.py`.
    *   Run the inference script:
        ```bash
        python inferrence.py
        ```
    *   Super-resolved images will be saved to the `SR_OUTPUT_DIR`.

## Results Achieved

Using the FSRCNN architecture, with experimental parameters (e.g., 64 feature extraction filters, 12 channels in shrink/map layers, 6 mapping layers), the following best performance metrics were observed on the Y-channel of the custom test dataset:

*   **PSNR:** 19.98 dB
*   **SSIM:** 0.457

While these results demonstrate a functioning super-resolution pipeline, they are modest compared to current state-of-the-art benchmarks, indicating potential for further improvement or dataset-specific challenges.

## Future Prospects

Based on the project's outcomes and learnings, potential directions for future work include:
*   **In-depth Data Analysis:** A more thorough examination of the custom dataset for characteristics that might hinder performance (e.g., noise, compression artifacts, specific content types).
*   **Exploring Advanced Architectures:** Experimenting with more recent and powerful super-resolution models such as EDSR, RCAN, or Transformer-based models like SwinIR.
*   **Transfer Learning:** Leveraging models pre-trained on larger, standard datasets (e.g., DIV2K) and fine-tuning them on the custom dataset.
*   **Advanced Loss Functions:** Incorporating alternative loss functions like L1 loss, perceptual loss (using features from pre-trained classification networks), or adversarial loss (if moving towards GAN-based SR).
*   **Systematic Hyperparameter Optimization:** Employing automated techniques like grid search, random search, or Bayesian optimization for more exhaustive hyperparameter tuning.

## Original Repository Credits

This work is fundamentally based on the FSRCNN-PyTorch implementation by **Lornatang**. We acknowledge and appreciate their contribution to the open-source community.
Please refer to the original repository for the base implementation and their detailed documentation:
[https://github.com/Lornatang/FSRCNN-PyTorch](https://github.com/Lornatang/FSRCNN-PyTorch)

## License

The modifications made in this fork are provided under the same license as the original repository (Apache License 2.0), unless explicitly stated otherwise.
