import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio
import torchvision.transforms as T
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset # 確保導入

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
        # self.loss_fn = nn.MSELoss()

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
        scheduler.step()
        lr = scheduler.get_last_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]

        # --- 在這裡添加 validation_step 方法 ---
    def validation_step(self, batch, batch_idx):
        degrad_patch, clean_patch = batch

        predicted_patch = self.net(degrad_patch)
        self.val_psnr(predicted_patch, clean_patch)

        # --- 記錄驗證集 PSNR 指標 ---
        # 使用 self.log() 方法記錄 PSNR 指標
        # 'val_psnr' 是你設定 ModelCheckpoint 監控的名稱，必須完全一致
        # on_step=False, on_epoch=True 表示不在每個 batch 結束時記錄，而是在整個 Epoch 結束時記錄該 Epoch 的平均 PSNR
        # prog_bar=True 讓指標顯示在進度條中
        self.log("val_psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)

        val_loss = self.loss_fn(predicted_patch, clean_patch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

def main():
    # print("Options")
    # print(opt)
    # if opt.wblogger is not None:
    #     logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    # else:
    
    LOCAL_SAVE_DIR = "local/"

    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    SAVE_NAME = "128_to_256/"
    logger = TensorBoardLogger(save_dir = LOCAL_SAVE_DIR + "runs/")

    DATA_DIR = "hw4_realse_dataset/train"
    CHECKPOINT_DIR = LOCAL_SAVE_DIR + "params/" + SAVE_NAME

    BATCH_SIZE = 4
    EPOCHS = 50
    resize_size = (128, 128)

    transform = T.Compose([
        T.Resize(resize_size),
        T.ToTensor()
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_degraded = T.Compose([
        T.Resize(resize_size),
        T.Resize(resize_size),
        T.ToTensor()
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_clean = T.Compose([
        # T.Resize(resize_size),
        T.ToTensor()
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = CustomDataset(DATA_DIR, transform_degraded=transform_degraded, transform_clean=transform_clean)
    sample_types = full_dataset.img_type
    all_indices = list(range(len(full_dataset)))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=sample_types, # 這裡傳入包含每個樣本類型信息的列表
        random_state=63
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")

    # total_size = len(train_dataset)
    # train_size = int(0.8 * total_size)
    # val_size = total_size - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath = CHECKPOINT_DIR,
        filename= 'best_psnr-{epoch:03d}-{val_psnr:.2f}',
        every_n_epochs = 1,     # 每 Epoch 結束時檢查一次
        save_top_k=5,           # 只保留最佳的一個
        monitor='val_psnr',     # <--- 監控名為 'val_psnr' 的指標
        mode='max'              # <--- 設定模式為 'max'，表示這個指標越大越好
    )

    early_stop_callback = EarlyStopping(
        monitor='val_psnr', # 要監控的指標名稱，必須在 self.log 中精確匹配
        min_delta=0.00,     # 指標改善的最小幅度，小於此值不算改善 (可選)
        patience=EPOCHS//5, # 在最佳 Epoch 後，連續多少個 Epoch 沒有改善就停止
        verbose=True,       # 是否在早停時打印訊息 (建議設為 True)
        mode='max'          # 'val_psnr' 指標是越大越好
    )
    # 這裡的 train_dataset 和 val_dataset 是 CustomDataset 的實例    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4)
    # return 
    # plot_and_save_image_pairs(train_dataset, num_pairs_to_plot=8, save_path="local/train_image_pairs_first_8.png")
    model = PromptIRModel()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = PromptIRModel.load_from_checkpoint(
    #          checkpoint_path = 'local/params/test2/testbest_psnr-epoch=018-val_psnr=26.24.ckpt',
    #          map_location=device,
    #     )
    trainer = pl.Trainer( max_epochs=EPOCHS, accelerator="gpu", logger=logger, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()


