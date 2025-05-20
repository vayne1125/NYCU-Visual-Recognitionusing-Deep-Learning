"""
This module defines a PyTorch Lightning model (`PromptIRModel`) for image restoration.

It encapsulates:
- The `PromptIR` neural network architecture.
- Training logic using **MSE loss** and **PSNR metric**.
- A **patch-based training strategy** with fixed 2x2 patch splitting.
- A **validation strategy that stitches patches** back to full images for PSNR calculation.
- **Optimizer and learning rate scheduler** configuration.
- Setup for data loading, including **stratified train/validation splitting**
  of the `CustomDataset`.
- **PyTorch Lightning callbacks** for model checkpointing and early stopping.

This script is designed for training and validating image restoration models
using a structured, reproducible approach.
"""
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import PeakSignalNoiseRatio
import torchvision.transforms as T
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from schedulers import LinearWarmupCosineAnnealingLR
from model import PromptIR
from datasets import CustomDataset
from utils import plot_and_save_image_pairs, batch_stitch_patches_2x2

class PromptIRModel(pl.LightningModule):
    """
    A PyTorch Lightning module for training an image restoration model.

    This class encapsulates the PromptIR network, defines the training and
    validation steps using MSE loss and PSNR, and configures optimizers
    and learning rate scheduling. It's designed for patch-based training
    where validation involves stitching patches back to full images.
    """
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.MSELoss()  # MSE Loss (L2 Loss)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_psnr_patch = PeakSignalNoiseRatio(data_range=1.0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # degraded_patches_batch shape: [DataLoader_batch_size, 4, C, 128, 128]
        # clean_patches_batch shape:    [DataLoader_batch_size, 4, C, 128, 128]
        # clean_256_batch shape:    [DataLoader_batch_size, C, 256, 256]
        degraded_patches_batch, clean_patches_batch, _ = batch

        batch_size, num_patches, C, H, W = degraded_patches_batch.shape
        # Shape [batch_size * 4, C, 128, 128]
        degraded_patches_flat = degraded_patches_batch.view(-1, C, H, W)
        degraded_patches_flat = degraded_patches_flat.to(self.device)

        restored_patches_flat = self.net(
            degraded_patches_flat)  # Model outputs 128x128
        # Shape [batch_size * 4, C, 128, 128]
        clean_patches_flat = clean_patches_batch.view(
            -1, C, H, W).to(self.device)

        loss = self.loss_fn(restored_patches_flat, clean_patches_flat)
        self.log("train_loss", loss)

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                  warmup_epochs=15, max_epochs=50)
        return [optimizer], [scheduler]

    # --- validation_step method remains unchanged: calculates PSNR on 256x256 stitched results ---
    def validation_step(self, batch, batch_idx):
        # Batch data structure: (degraded_patches_batch, clean_patches_batch, clean_256_batch)
        # degraded_patches_batch shape: [DataLoader_batch_size, 4, C, 128, 128]
        # clean_patches_batch shape:    [DataLoader_batch_size, 4, C, 128, 128]
        # clean_256_batch shape:    [DataLoader_batch_size, C, 256, 256] (used as 256 PSNR target)
        # Receive all elements in the batch
        degraded_patches_batch, _, clean_256_batch = batch

        # --- Reshape degraded patches into a single large flat batch for the model ---
        batch_size, num_patches, C, H, W = degraded_patches_batch.shape
        # Shape [batch_size * 4, C, 128, 128]
        degraded_patches_flat = degraded_patches_batch.view(-1, C, H, W)

        degraded_patches_flat = degraded_patches_flat.to(self.device)

        with torch.no_grad():
            restored_patches_flat = self.net(
                degraded_patches_flat)  # Model outputs 128x128

        # --- Move model output back to the target device (same as clean_256_batch) ---
        restored_patches_flat = restored_patches_flat.to(
            clean_256_batch.device)

        # --- Reshape restored patches back to the original batch structure ---
        restored_patches_batch = restored_patches_flat.view(
            batch_size, num_patches, C, H, W)  # Shape [batch_size, 4, C, 128, 128]

        # --- Stitch restored patches back into 256x256 images ---
        # Use batch_stitch_patches_2x2 helper function
        restored_stitched_256_batch = batch_stitch_patches_2x2(
            restored_patches_batch)  # Shape [batch_size, C, 256, 256]

        # --- Update PSNR metric state ---
        # self.val_psnr will accumulate states from all validation steps
        self.val_psnr.update(restored_stitched_256_batch, clean_256_batch)
        val_loss = self.loss_fn(restored_stitched_256_batch, clean_256_batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True,
                 prog_bar=True)  # Log at the end of the Epoch

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation Epoch. Computes and logs the accumulated PSNR.
        """
        # --- Compute and log the average validation PSNR ---
        avg_val_psnr = self.val_psnr.compute()
        self.log("val_psnr", avg_val_psnr, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.val_psnr.reset()


if __name__ == '__main__':

    LOCAL_SAVE_DIR = "local/"

    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    SAVE_NAME = "part1"
    logger = TensorBoardLogger(save_dir=LOCAL_SAVE_DIR + "runs/")

    DATA_DIR = "hw4_realse_dataset/train"
    CHECKPOINT_DIR = LOCAL_SAVE_DIR + "params/" + SAVE_NAME

    BATCH_SIZE = 1
    EPOCHS = 5

    transform = T.Compose([
        T.ToTensor()
    ])

    full_dataset = CustomDataset(DATA_DIR, transform=transform)

    sample_types = full_dataset.img_type
    all_indices = list(range(len(full_dataset)))

    # train_indices, val_indices = train_test_split(
    #     all_indices,
    #     test_size=0.2,
    #     stratify=sample_types, # Pass a list containing type information for each sample here
    #     random_state=63
    # )

    tp_indices, train_indices = train_test_split(
        all_indices,
        test_size=0.1,
        stratify=sample_types,  # Pass a list containing type information for each sample here
        random_state=63
    )

    tp_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.1,
        stratify=sample_types,  # Pass a list containing type information for each sample here
        random_state=63
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        # filename= 'best_psnr-{epoch:03d}-{val_psnr:.2f}',
        filename='best_params',
        every_n_epochs=1,
        save_top_k=1,
        monitor='val_psnr',
        mode='max'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_psnr',  # Name of the metric to monitor, must precisely match in self.log
        # Minimum improvement in the monitored metric to count as an improvement (optional)
        min_delta=0.00,
        # Number of consecutive epochs without improvement after the best Epoch to stop
        patience=EPOCHS//5,
        # Whether to print messages when early stopping (recommended True)
        verbose=True,
        mode='max'          # 'val_psnr' metric: higher is better
    )
    plot_and_save_image_pairs(train_dataset, num_images_to_plot=4,
                              save_path="local/train_image_pairs_first_8.png")

    model = PromptIRModel()
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu", logger=logger, callbacks=[
                         checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
