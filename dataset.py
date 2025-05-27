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
"""Realize the function of dataset preparation."""
import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

import imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class TrainValidImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        image_dir (str): Train/Valid dataset address.
        image_size (int): High resolution image size.
        upscale_factor (int): Image up scale factor.
        mode (str): Data set loading method, the training data set is for data enhancement, and the verification data set is not for data enhancement.
    """

    def __init__(self, hr_image_dir: str, lr_image_dir: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()

        self.hr_image_files = natsorted([os.path.join(hr_image_dir, f) for f in os.listdir(hr_image_dir) if os.path.isfile(os.path.join(hr_image_dir, f))])
        # Assuming LR images have the SAME filenames as their HR counterparts
        self.lr_image_files = [os.path.join(lr_image_dir, os.path.basename(hr_file)) for hr_file in self.hr_image_files]

        # Sanity check: ensure all LR files exist
        missing_lr = [f for f in self.lr_image_files if not os.path.exists(f)]
        if missing_lr:
            raise FileNotFoundError(f"Missing {len(missing_lr)} LR files. Example: {missing_lr[0]}")
        if not self.hr_image_files:
             raise FileNotFoundError(f"No HR images found in {hr_image_dir}")


        self.image_size = image_size  # HR patch size
        self.upscale_factor = upscale_factor
        self.mode = mode

        if self.mode not in ["Train", "Valid"]:
            raise ValueError("Mode must be 'Train' or 'Valid'.")

    def __getitem__(self, batch_index: int) -> dict[str, torch.Tensor]:
        hr_image_path = self.hr_image_files[batch_index]
        lr_image_path = self.lr_image_files[batch_index]

        try:
            hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.
            lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.
        except Exception as e:
            raise IOError(f"Error reading image: HR='{hr_image_path}', LR='{lr_image_path}'. Original error: {e}")


        if hr_image is None: raise IOError(f"Failed to load HR image: {hr_image_path}")
        if lr_image is None: raise IOError(f"Failed to load LR image: {lr_image_path}")

        if self.mode == "Train":
            lr_image, hr_image = imgproc.random_crop(lr_image, hr_image, self.image_size, self.upscale_factor)
            # Optional: Add more augmentations like flips/rotations if desired
            lr_image, hr_image = imgproc.random_rotate(lr_image, hr_image, angles=[0, 90, 180, 270])
            lr_image, hr_image = imgproc.random_horizontally_flip(lr_image, hr_image)
            # lr_image, hr_image = imgproc.random_vertically_flip(lr_image, hr_image) # Often less common for SR
        elif self.mode == "Valid":
            lr_image, hr_image = imgproc.center_crop(lr_image, hr_image, self.image_size, self.upscale_factor)

        # Convert BGR to YCbCr and extract Y channel (as FSRCNN typically works on luminance)
        lr_y_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)

        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.hr_image_files)


class TestImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        test_lr_image_dir (str): Test dataset address for low resolution image dir.
        test_hr_image_dir (str): Test dataset address for high resolution image dir.
        upscale_factor (int): Image up scale factor.
    """

    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        self.lr_image_files = natsorted([os.path.join(test_lr_image_dir, f) for f in os.listdir(test_lr_image_dir) if os.path.isfile(os.path.join(test_lr_image_dir, f))])
        # Assuming LR images have the SAME filenames as their HR counterparts
        self.hr_image_files = [os.path.join(test_hr_image_dir, os.path.basename(lr_file)) for lr_file in self.lr_image_files]

        missing_hr = [f for f in self.hr_image_files if not os.path.exists(f)]
        if missing_hr:
            raise FileNotFoundError(f"Missing {len(missing_hr)} HR files for testing. Example: {missing_hr[0]}")
        if not self.lr_image_files:
             raise FileNotFoundError(f"No LR images found in {test_lr_image_dir}")

    def __getitem__(self, batch_index: int) -> dict[str, torch.Tensor | str]:
        lr_image_path = self.lr_image_files[batch_index]
        hr_image_path = self.hr_image_files[batch_index]

        try:
            lr_image = cv2.imread(lr_image_path).astype(np.float32) / 255.
            hr_image = cv2.imread(hr_image_path).astype(np.float32) / 255.
        except Exception as e:
            raise IOError(f"Error reading image: HR='{hr_image_path}', LR='{lr_image_path}'. Original error: {e}")

        if lr_image is None: raise IOError(f"Failed to load LR image: {lr_image_path}")
        if hr_image is None: raise IOError(f"Failed to load HR image: {hr_image_path}")

        lr_y_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)

        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

        # Return filenames for reference during validation/testing if needed
        return {"lr": lr_y_tensor, "hr": hr_y_tensor, "lr_path": lr_image_path, "hr_path": hr_image_path}

    def __len__(self) -> int:
        return len(self.lr_image_files)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
