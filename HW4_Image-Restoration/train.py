import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
import torchvision.transforms as T

# from utils.dataset_utils import PromptTrainDataset
from datasets import CustomDataset
from model import PromptIR
from schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
# from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils import plot_and_save_image_pairs
import os

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()

        # --- 在這裡初始化驗證集 PSNR 指標 ---
        # 使用 torchmetrics.regression.PeakSignalNoiseRatio
        # data_range=1.0 是因為你的影像像素值通常被正規化到 [0, 1]
        # 如果你的像素值是 [0, 255]，請移除 data_range=1.0 或設定為 data_range=255
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)

    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        degrad_patch, clean_patch = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

        # --- 在這裡添加 validation_step 方法 ---
    def validation_step(self, batch, batch_idx):
        # 從 batch 中解包數據 (假設結構和 training_step 一樣)
        degrad_patch, clean_patch = batch

        # 將有雨水影像傳入模型，得到模型預測的乾淨影像
        # 注意：在 validation_step 中，PyTorch Lightning 預設會關閉梯度計算 (torch.no_grad())
        predicted_patch = self.net(degrad_patch)

        # --- 計算驗證集 PSNR ---
        # 調用 self.val_psnr 實例，傳入模型預測結果和真實的乾淨影像
        # self.val_psnr 會在內部累積當前 Epoch 所有 batch 的 PSNR 狀態
        self.val_psnr(predicted_patch, clean_patch)

        # --- 記錄驗證集 PSNR 指標 ---
        # 使用 self.log() 方法記錄 PSNR 指標
        # 'val_psnr' 是你設定 ModelCheckpoint 監控的名稱，必須完全一致
        # on_step=False, on_epoch=True 表示不在每個 batch 結束時記錄，而是在整個 Epoch 結束時記錄該 Epoch 的平均 PSNR
        # prog_bar=True 讓指標顯示在進度條中
        self.log("val_psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)

        # 可選：計算並記錄驗證損失
        val_loss = self.loss_fn(predicted_patch, clean_patch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

        # validation_step 的返回值通常不是必須的，除非你在 on_validation_epoch_end 需要進一步處理

def main():
    # print("Options")
    # print(opt)
    # if opt.wblogger is not None:
    #     logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    # else:
    
    LOCAL_SAVE_DIR = "local/"

    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    SAVE_NAME = "test"
    logger = TensorBoardLogger(save_dir = LOCAL_SAVE_DIR + "runs/")

    DATA_DIR = "hw4_realse_dataset/train"
    CHECKPOINT_DIR = LOCAL_SAVE_DIR + "params"

    BATCH_SIZE = 4
    EPOCHS = 10
    
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(root_dir=DATA_DIR, transform=train_transform)

    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = CHECKPOINT_DIR,
        filename= SAVE_NAME + 'best_psnr-{epoch:03d}-{val_psnr:.2f}',
        every_n_epochs = 1,     # 每 Epoch 結束時檢查一次
        save_top_k=1,           # 只保留最佳的一個
        monitor='val_psnr',     # <--- 監控名為 'val_psnr' 的指標
        mode='max'              # <--- 設定模式為 'max'，表示這個指標越大越好
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    
    plot_and_save_image_pairs(train_dataset, num_pairs_to_plot=8, save_path="local/train_image_pairs_first_8.png")
    model = PromptIRModel()
    
    trainer = pl.Trainer( max_epochs=EPOCHS, accelerator="gpu", logger=logger, callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()


