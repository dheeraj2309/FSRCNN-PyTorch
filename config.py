# config.py
import torch
import os

# --- Dataset Paths ---
# For Kaggle, these paths will typically start with "/kaggle/input/your-dataset-name/"
# For local development, use your local paths.
KAGGLE_INPUT_DIR = "/kaggle/input/updated-dataset/dataset" # CHANGE 'your-dataset-slug' if on Kaggle
LOCAL_BASE_DIR = "dataset" # Your local base directory for the dataset

# Determine if running on Kaggle or locally (simple check)
IS_KAGGLE = os.path.exists("/kaggle/input")
BASE_DATA_PATH = KAGGLE_INPUT_DIR if IS_KAGGLE else LOCAL_BASE_DIR

# Training datasets
TRAIN_HR_DIR = os.path.join(BASE_DATA_PATH, "train/HR")
TRAIN_LR_DIR = os.path.join(BASE_DATA_PATH, "train/LR")
# Validation datasets
VALID_HR_DIR = os.path.join(BASE_DATA_PATH, "valid/HR")
VALID_LR_DIR = os.path.join(BASE_DATA_PATH, "valid/LR")
# Test datasets
TEST_HR_DIR = os.path.join(BASE_DATA_PATH, "test/HR")
TEST_LR_DIR = os.path.join(BASE_DATA_PATH, "test/LR")

# --- Model Settings ---
UPSCALE_FACTOR = 4  # Super-resolution upscale factor
EXP_NAME = f"FSRCNN_x{UPSCALE_FACTOR}_custom" # Experiment name for outputs

# --- Output Paths ---
# For Kaggle, writable directory is /kaggle/working/
KAGGLE_OUTPUT_DIR = "/kaggle/working"
LOCAL_OUTPUT_DIR = "." # Current directory for local outputs

BASE_OUTPUT_PATH = KAGGLE_OUTPUT_DIR if IS_KAGGLE else LOCAL_OUTPUT_DIR

MODEL_SAVE_DIR = os.path.join(BASE_OUTPUT_PATH, "samples", EXP_NAME) # For epoch checkpoints
RESULTS_SAVE_DIR = os.path.join(BASE_OUTPUT_PATH, "results", EXP_NAME) # For best/last models
LOGS_DIR = os.path.join(BASE_OUTPUT_PATH, "samples", "logs", EXP_NAME) # Tensorboard logs
# For validate.py output of SR images
SR_TEST_OUTPUT_DIR = os.path.join(BASE_OUTPUT_PATH, "results", "test_output", EXP_NAME)

# Path to the model for validation/testing (update after training)
# This will typically be 'best.pth.tar' from RESULTS_SAVE_DIR
VALIDATE_MODEL_PATH = os.path.join(RESULTS_SAVE_DIR, "best.pth.tar")

# --- Training Parameters ---
IMAGE_SIZE = 64      # High-resolution image patch size for cropping.
                       # LR patch size will be IMAGE_SIZE // UPSCALE_FACTOR.
BATCH_SIZE = 16
EPOCHS = 100           # Number of epochs to train
START_EPOCH = 0        # Default starting epoch

# Optimizer parameters
MODEL_LR = 1e-3
MODEL_MOMENTUM = 0.9
MODEL_WEIGHT_DECAY = 1e-4
MODEL_NESTEROV = False

# --- Device Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2 if torch.cuda.is_available() else 0 # Number of worker threads for data loading
PIN_MEMORY = True if torch.cuda.is_available() else False

# --- Resume Training ---
# Path to checkpoint file for resuming training, e.g., os.path.join(MODEL_SAVE_DIR, "epoch_10.pth.tar")
RESUME_CHECKPOINT = ""

# --- Logging/Saving Frequency ---
PRINT_FREQUENCY = 50 # Print training stats every N batches