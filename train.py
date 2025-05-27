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
# ============================================================================
"""File description: Realize the model training function."""
import os
import shutil
import time
import matplotlib.pyplot as plt
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import CUDAPrefetcher
from dataset import TrainValidImageDataset, TestImageDataset
from model import FSRCNN


def main():
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    # Create output directories if they don't exist
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)


    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    model = build_model()
    print("Build FSRCNN model successfully.")

    psnr_criterion, pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the pretrained model is restored...")
    if config.RESUME_CHECKPOINT: # Use RESUME_CHECKPOINT from config
        if os.path.isfile(config.RESUME_CHECKPOINT):
            print(f"Loading checkpoint '{config.RESUME_CHECKPOINT}'")
            checkpoint = torch.load(config.RESUME_CHECKPOINT, map_location=lambda storage, loc: storage)
            config.START_EPOCH = checkpoint["epoch"]
            best_psnr = checkpoint["best_psnr"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["scheduler"]) # If you add a scheduler
            print(f"Loaded checkpoint '{config.RESUME_CHECKPOINT}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{config.RESUME_CHECKPOINT}'")
    else:
        print("No checkpoint specified. Training from scratch.")

    # Create a folder of super-resolution experiment results
    

    # Create training process log file
    writer = SummaryWriter(log_dir=config.LOGS_DIR) # Use LOGS_DIR from config

    scaler = amp.GradScaler(enabled=config.DEVICE.type == 'cuda') # Enable scaler only for CUDA

    epoch_numbers_list = []
    train_losses_list = []
    train_psnrs_list = []
    valid_psnrs_list = []
    test_psnrs_list = []

    for epoch in range(config.START_EPOCH, config.EPOCHS):
        # Train for one epoch
        avg_train_loss, avg_train_psnr = train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        
        # Validate on validation set
        valid_psnr = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        
        # Validate on test set
        test_psnr = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # ADDED: Store metrics for this epoch
        epoch_numbers_list.append(epoch + 1)
        train_losses_list.append(avg_train_loss)
        train_psnrs_list.append(avg_train_psnr)
        valid_psnrs_list.append(valid_psnr)
        test_psnrs_list.append(test_psnr)

        is_best = test_psnr > best_psnr # Using test_psnr to determine best model as per original logic
        best_psnr = max(test_psnr, best_psnr)

        checkpoint_data = {
            "epoch": epoch + 1,
            "best_psnr": best_psnr,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint_data, os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch + 1}.pth.tar"), 
                            os.path.join(config.RESULTS_SAVE_DIR, "best.pth.tar"))
        if (epoch + 1) == config.EPOCHS:
            shutil.copyfile(os.path.join(config.MODEL_SAVE_DIR, f"epoch_{epoch + 1}.pth.tar"), 
                            os.path.join(config.RESULTS_SAVE_DIR, "last.pth.tar"))
    
    writer.close()

    # ADDED: Call plotting function after training loop
    plot_training_results(epoch_numbers_list, train_losses_list, train_psnrs_list, valid_psnrs_list, test_psnrs_list)

def load_dataset() -> tuple[CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.TRAIN_HR_DIR,
                                            config.TRAIN_LR_DIR,
                                            config.IMAGE_SIZE,
                                            config.UPSCALE_FACTOR,
                                            "Train")
    valid_datasets = TrainValidImageDataset(config.VALID_HR_DIR,
                                            config.VALID_LR_DIR,
                                            config.IMAGE_SIZE, # For validation, this is HR crop size
                                            config.UPSCALE_FACTOR,
                                            "Valid")
    test_datasets = TestImageDataset(config.TEST_LR_DIR, config.TEST_HR_DIR)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=config.NUM_WORKERS,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True if config.NUM_WORKERS > 0 else False)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=config.NUM_WORKERS,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True if config.NUM_WORKERS > 0 else False)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=False)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.DEVICE)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.DEVICE)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.DEVICE)


    return train_prefetcher, valid_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = FSRCNN(config.UPSCALE_FACTOR).to(config.DEVICE)
    return model


def define_loss() -> [nn.MSELoss, nn.MSELoss]:
    psnr_criterion = nn.MSELoss().to(config.DEVICE)
    pixel_criterion = nn.MSELoss().to(config.DEVICE)

    return psnr_criterion, pixel_criterion


def define_optimizer(model : nn.Module) -> optim.SGD:
    optimizer = optim.SGD([{"params": model.feature_extraction.parameters()},
                           {"params": model.shrink.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.expand.parameters()},
                           {"params": model.deconv.parameters(), "lr": config.MODEL_LR * 0.1}],
                          lr=config.MODEL_LR,
                          momentum=config.MODEL_MOMENTUM,
                          weight_decay=config.MODEL_WEIGHT_DECAY,
                          nesterov=config.MODEL_NESTEROV)

    return optimizer


def train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer) -> tuple[float, float]:
    batches = len(train_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    model.train()
    batch_index = 0
    end = time.time()
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    while batch_data is not None:
        data_time.update(time.time() - end)
        lr = batch_data["lr"].to(config.DEVICE, non_blocking=True)
        hr = batch_data["hr"].to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast(enabled=config.DEVICE.type == 'cuda'):
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % config.PRINT_FREQUENCY == 0:
            writer.add_scalar("Train/Loss_batch", loss.item(), batch_index + epoch * batches + 1) # Renamed for clarity
            progress.display(batch_index)

        batch_data = train_prefetcher.next()
        batch_index += 1
    
    # Log epoch-level average training loss and PSNR to TensorBoard
    if writer is not None:
        writer.add_scalar("Train/EpochAvgLoss", losses.avg, epoch + 1)
        writer.add_scalar("Train/EpochAvgPSNR", psnres.avg, epoch + 1)
    
    # Return average loss and PSNR for this training epoch
    return losses.avg, psnres.avg


def validate(model, data_prefetcher, psnr_criterion, epoch, writer, mode) -> float:
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter(f"{mode} PSNR", ":4.2f", Summary.AVERAGE) # Use mode in name
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres], prefix=f"{mode}: ")

    model.eval()
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()
    batch_index = 0
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            lr = batch_data["lr"].to(config.DEVICE, non_blocking=True)
            hr = batch_data["hr"].to(config.DEVICE, non_blocking=True)

            with amp.autocast(enabled=config.DEVICE.type == 'cuda'):
                sr = model(lr)
            
            # Ensure tensors are float for PSNR calculation
            mse = psnr_criterion(sr.float(), hr.float())
            if mse.item() == 0: # Avoid log(0)
                psnr_val = float('inf')
            else:
                psnr_tensor = 10. * torch.log10(1.0 / mse)
                psnr_val = psnr_tensor.item()
            
            psnres.update(psnr_val, lr.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_index % config.PRINT_FREQUENCY == 0:
                progress.display(batch_index)
            
            batch_data = data_prefetcher.next()
            batch_index += 1

    progress.display_summary()
    writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
    return psnres.avg

def plot_training_results(epochs, train_losses, train_psnrs, valid_psnrs, test_psnrs):
    """
    Plots training and validation/test metrics over epochs and saves the plot.
    """
    plt.figure(figsize=(18, 12)) # Adjusted figure size for better layout

    # Plot 1: Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='red', linestyle='-', marker='o')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Training PSNR
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_psnrs, label='Training PSNR', color='blue', linestyle='-', marker='o')
    plt.title('Training PSNR vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    # Plot 3: Validation PSNR
    plt.subplot(2, 2, 3)
    plt.plot(epochs, valid_psnrs, label='Validation PSNR', color='green', linestyle='-', marker='s')
    plt.title('Validation PSNR vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    # Plot 4: Test PSNR
    plt.subplot(2, 2, 4) # Changed from 2,2,3 to 2,2,4
    plt.plot(epochs, test_psnrs, label='Test PSNR', color='purple', linestyle='-', marker='^')
    plt.title('Test PSNR vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(pad=3.0) # Add some padding
    
    # Save the plot to the results directory
    plot_filename = os.path.join(config.RESULTS_SAVE_DIR, "training_performance_plots.png")
    try:
        plt.savefig(plot_filename)
        print(f"Training performance plots saved to '{plot_filename}'")
    except Exception as e:
        print(f"Error saving plot: {e}")
# Copy form "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
