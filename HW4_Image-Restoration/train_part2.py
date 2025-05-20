"""
This module focuses on fine-tuning an image restoration model using PyTorch Lightning.

It defines `PromptIRModel`, a LightningModule that:
- Integrates the `PromptIR` neural network.
- Uses a combined loss function (MSE, L1, and SSIM) for robust training.
- Implements a random crop strategy for training data augmentation.
- Configures an AdamW optimizer with a Linear Warmup Cosine Annealing LR scheduler.
- Handles loading pre-trained checkpoints by resetting optimizer and scheduler states 
    to allow for new training curves.
- Employs PyTorch Lightning callbacks for logging, model checkpointing (monitoring `val_psnr`),
  and early stopping to optimize the training process.

The main execution block sets up the training environment, including data loading,
stratified train/validation splitting, and trainer configuration,
making it suitable for transfer learning or continued training of image restoration models.
"""
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchmetrics.image import PeakSignalNoiseRatio
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from pytorch_msssim import ssim

from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from datasets import CustomDataset, RandomDataset
from model import PromptIR
from schedulers import LinearWarmupCosineAnnealingLR

from utils import batch_stitch_patches_2x2

class PromptIRModel(pl.LightningModule):
    """
    A PyTorch Lightning module for training and fine-tuning an image restoration model.

    This class encapsulates the PromptIR network, defines training and validation
    steps using a combined MSE, L1, and SSIM loss, and configures optimizers
    and a learning rate scheduler. It supports random cropping for training
    and handles loading pre-trained checkpoints by resetting optimizer states.
    """
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)

        self.loss_fn_mse = nn.MSELoss()  # MSE Loss (L2 Loss)
        self.loss_fn_l1 = nn.L1Loss()   # L1 Loss (MAE Loss)
        # SSIM Loss: 1 - SSIM, so minimizing this maximizes SSIM
        self.loss_fn_ssim = lambda x, y: 1 - \
            ssim(x, y, data_range=1.0, size_average=True)

        # --- Define weights for combined Loss ---
        self.loss_weight_mse = 0.2
        self.loss_weight_l1 = 0.6
        self.loss_weight_ssim = 0.2

        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_psnr_patch = PeakSignalNoiseRatio(data_range=1.0)

    def forward(self, x):
        return self.net(x)

    # --- Modified training_step method for Random Crop strategy ---
    def training_step(self, batch, batch_idx):
        # Batch data structure for Random Crop training:
        #   (degraded_patches_batch, clean_patches_batch)
        # degraded_patches_batch shape: [DataLoader_batch_size, C, 128, 128]
        # clean_patches_batch shape:    [DataLoader_batch_size, C, 128, 128]
        degraded_patches_batch, clean_patches_batch = batch

        # --- Move patch data to device (GPU) ---
        degraded_patches_batch = degraded_patches_batch.to(self.device)
        clean_patches_batch = clean_patches_batch.to(self.device)

        restored_patches_batch = self.net(degraded_patches_batch)
        mse_loss = self.loss_fn_mse(
            restored_patches_batch, clean_patches_batch)
        l1_loss = self.loss_fn_l1(restored_patches_batch, clean_patches_batch)
        ssim_loss_value = self.loss_fn_ssim(
            restored_patches_batch, clean_patches_batch)

        total_loss = self.loss_weight_mse * mse_loss + self.loss_weight_l1 * \
            l1_loss + self.loss_weight_ssim * ssim_loss_value

        self.log("train_loss", total_loss, on_step=True,
                 on_epoch=True, prog_bar=True)

        return total_loss  # Return total Loss for backpropagation

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer, warmup_epochs=8, max_epochs=80)
        return [optimizer], [scheduler]

    def on_load_checkpoint(self, checkpoint):
        """
        Called before loading the state dict from a checkpoint.
        Here, we remove optimizer and scheduler states to reinitialize them.
        """
        if 'optimizer_states' in checkpoint:
            del checkpoint['optimizer_states']
        if 'lr_schedulers' in checkpoint:
            del checkpoint['lr_schedulers']

    # --- validation_step method remains unchanged: calculates PSNR on 256x256 stitched results ---
    def validation_step(self, batch, batch_idx):
        # Batch data structure: (degraded_patches_batch, clean_patches_batch, clean_256_batch)
        # degraded_patches_batch shape: [DataLoader_batch_size, 4, C, 128, 128]
        # clean_patches_batch shape:    [DataLoader_batch_size, 4, C, 128, 128]
        # clean_256_batch shape:    [DataLoader_batch_size, C, 256, 256]
        # Receive all elements in the batch
        degraded_patches_batch, clean_patches_batch, clean_256_batch = batch
        batch_size, num_patches, C, H, W = degraded_patches_batch.shape
        # Shape [batch_size * 4, C, 128, 128]
        degraded_patches_flat = degraded_patches_batch.view(-1, C, H, W)
        degraded_patches_flat = degraded_patches_flat.to(self.device)

        with torch.no_grad():
            restored_patches_flat = self.net(
                degraded_patches_flat)  # Model outputs 128x128

        restored_patches_flat = restored_patches_flat.to(
            clean_256_batch.device)
        restored_patches_batch = restored_patches_flat.view(
            batch_size, num_patches, C, H, W)  # Shape [batch_size, 4, C, 128, 128]
        restored_stitched_256_batch = batch_stitch_patches_2x2(
            restored_patches_batch)  # Shape [batch_size, C, 256, 256]

        # --- Update PSNR metric state ---
        self.val_psnr.update(restored_stitched_256_batch, clean_256_batch)

        ssim_value_256 = ssim(restored_stitched_256_batch,
                              clean_256_batch, data_range=1.0, size_average=True)
        val_loss_mse = self.loss_fn_mse(
            restored_stitched_256_batch, clean_256_batch)
        val_loss_l1 = self.loss_fn_l1(
            restored_stitched_256_batch, clean_256_batch)
        val_loss_total = self.loss_weight_mse * val_loss_mse + self.loss_weight_l1 * \
            val_loss_l1 + self.loss_weight_ssim * (1-ssim_value_256)

        self.log("val_loss", val_loss_total, on_step=False,
                 on_epoch=True, prog_bar=True)  # Log total validation Loss

    def on_validation_epoch_end(self):
        """
        Called at the end of each validation Epoch. Computes and logs the accumulated PSNR.
        """
        avg_val_psnr = self.val_psnr.compute()
        self.log("val_psnr", avg_val_psnr, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.val_psnr.reset()


if __name__ == '__main__':
    LOCAL_SAVE_DIR = "local/"

    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    SAVE_NAME = "part2"
    logger = TensorBoardLogger(save_dir=LOCAL_SAVE_DIR + "runs/")

    DATA_DIR = "hw4_realse_dataset/train"
    CHECKPOINT_DIR = LOCAL_SAVE_DIR + "params/" + SAVE_NAME

    EPOCHS = 80
    BATCH_SIZE = 4

    transform = T.Compose([
        T.ToTensor()
    ])

    CROP_SIZE = (128, 128)

    full_dataset = CustomDataset(DATA_DIR, transform=transform)
    random_crop_dataset = RandomDataset(
        DATA_DIR, transform=transform, crop_size=CROP_SIZE)

    sample_types = full_dataset.img_type
    all_indices = list(range(len(full_dataset)))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=sample_types,  # Pass a list containing type information for each sample here
        random_state=63
    )

    # train_dataset = Subset(full_dataset, train_indices)
    train_dataset = Subset(random_crop_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Training set size: {len(random_crop_dataset)}")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "part1/best_params"  # Your checkpoint name
    # Full path to the checkpoint
    MODEL_PATH = os.path.join('local/params', MODEL_NAME + '.ckpt')

    model = PromptIRModel.load_from_checkpoint(
        checkpoint_path=MODEL_PATH,
        map_location=device,
    )
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="gpu", logger=logger, callbacks=[
                         checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
