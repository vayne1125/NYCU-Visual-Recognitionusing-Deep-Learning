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
from datasets import CustomDataset, RandomDataset
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

import lightning.pytorch as pl
import torch.nn.functional as F # 確保導入 F

def batch_stitch_patches_2x2(patches_batch_tensor):
    """
    將一批每份 4 個非重疊的 128x128 Patch 拼接回一批 256x256 影像。
    假設 patches_batch_tensor 形狀是 [N, 4, C, 128, 128]。
    返回形狀為 [N, C, 256, 256] 的 Tensor。
    Patch 順序假定為：左上、右上、左下、右下。
    """
    N, num_patches, C, H, W = patches_batch_tensor.shape
    if num_patches != 4 or H != 128 or W != 128:
         raise ValueError(f"Expected patches_batch_tensor shape [N, 4, C, 128, 128], but got {patches_batch_tensor.shape}")

    stitched_images_list = []
    for i in range(N):
        single_image_patches = patches_batch_tensor[i] # 形狀 [4, C, 128, 128]
        
        # 按照 左上(0), 右上(1), 左下(2), 右下(3) 的順序拼接
        top_row = torch.cat((single_image_patches[0], single_image_patches[1]), dim=2) # 沿著寬度拼接，形狀 [C, 128, 256]
        bottom_row = torch.cat((single_image_patches[2], single_image_patches[3]), dim=2) # 沿著寬度拼接，形狀 [C, 128, 256]
        stitched_image = torch.cat((top_row, bottom_row), dim=1) # 沿著高度拼接，形狀 [C, 256, 256]
        stitched_images_list.append(stitched_image)

    return torch.stack(stitched_images_list, dim=0) # 形狀 [N, C, 256, 256]

from pytorch_msssim import ssim

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.net 應該是那個接收 128x128 輸入並輸出 128x128 的 PromptIR 模型
        self.net = PromptIR(decoder=True) 
        
        # self.loss_fn = nn.MSELoss() 
        self.loss_fn_mse = nn.MSELoss() # MSE Loss (L2 Loss)
        self.loss_fn_l1 = nn.L1Loss()   # L1 Loss (MAE Loss)
        self.loss_fn_ssim = lambda x, y: 1 - ssim(x, y, data_range=1.0, size_average=True)

        # --- 定義結合 Loss 的權重 ---
        # self.loss_weight_mse = 0.2 
        self.loss_weight_l1 = 0.7
        self.loss_weight_ssim = 0.3

        # PSNR 指標，用於計算拼接後的 256x256 結果與原始 256x256 clean 影像的 PSNR
        # data_range=1.0 是正確的，因為你的 Tensor 在 [0, 1] 範圍
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0) 
        self.train_psnr_patch = PeakSignalNoiseRatio(data_range=1.0) 


    # forward 方法不是 LightningModule 必須的，但保留著也無妨
    def forward(self,x):
        return self.net(x)

    # # --- 修改 training_step 方法，用於 Random Crop 策略 ---
    # # 這個 training_step 期望從 Random Crop DataLoader 獲取數據
    def training_step(self, batch, batch_idx):
        # 批次數據結構 for Random Crop 訓練: (degraded_patches_batch, clean_patches_batch)
        # degraded_patches_batch 形狀: [DataLoader_batch_size, C, 128, 128]
        # clean_patches_batch 形狀:   [DataLoader_batch_size, C, 128, 128]
        degraded_patches_batch, clean_patches_batch = batch 

        # --- 將 Patch 數據移到設備 (GPU) ---
        degraded_patches_batch = degraded_patches_batch.to(self.device)
        clean_patches_batch = clean_patches_batch.to(self.device)

        restored_patches_batch = self.net(degraded_patches_batch) 
        mse_loss = self.loss_fn_mse(restored_patches_batch, clean_patches_batch)
        l1_loss = self.loss_fn_l1(restored_patches_batch, clean_patches_batch)
        ssim_loss_value = self.loss_fn_ssim(restored_patches_batch, clean_patches_batch) 

        total_loss = self.loss_weight_mse * mse_loss + self.loss_weight_l1 * l1_loss + self.loss_weight_ssim * ssim_loss_value

        # --- 記錄個別 Loss 組件和總 Loss ---
        self.log("train_loss_mse", mse_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_l1", l1_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_ssim", ssim_loss_value, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True) 

        # self.train_psnr_patch.update(restored_patches_batch, clean_patches_batch) 
        # self.log("train_psnr_patch", self.train_psnr_patch, on_step=False, on_epoch=True, prog_bar=True) # 記錄在 Epoch 結束時

        return total_loss # 返回用於反向傳播的總 Loss
    
    # --- 修改 training_step 方法：在 128x128 Patch 尺度計算 Loss ---
    # def training_step(self, batch, batch_idx):
    #     # degraded_patches_batch 形狀: [DataLoader_batch_size, 4, C, 128, 128]
    #     # clean_patches_batch 形狀:   [DataLoader_batch_size, 4, C, 128, 128]
    #     # clean_256_batch 形狀:   [DataLoader_batch_size, C, 256, 256] (這個在 training_step 中不再用於 Loss 計算)
    #     degraded_patches_batch, clean_patches_batch, clean_256_batch = batch

    #     batch_size, num_patches, C, H, W = degraded_patches_batch.shape
    #     degraded_patches_flat = degraded_patches_batch.view(-1, C, H, W) # 形狀 [batch_size * 4, C, 128, 128]
    #     degraded_patches_flat = degraded_patches_flat.to(self.device)
        
    #     restored_patches_flat = self.net(degraded_patches_flat) # 模型輸出 128x128
    #     clean_patches_flat = clean_patches_batch.view(-1, C, H, W).to(self.device) # 形狀 [batch_size * 4, C, 128, 128]

    #     # loss = self.loss_fn(restored_patches_flat, clean_patches_flat)
    #     # self.log("train_loss", loss)
        
    #      # --- 計算個別 Loss 組件 (在 128x128 Patch 尺度) ---
    #     mse_loss = self.loss_fn_mse(restored_patches_flat, clean_patches_flat)
    #     l1_loss = self.loss_fn_l1(restored_patches_flat, clean_patches_flat)
    #     ssim_loss_value = self.loss_fn_ssim(restored_patches_flat, clean_patches_flat) 

    #     # --- 組合 Loss ---
    #     total_loss = self.loss_weight_mse * mse_loss + self.loss_weight_l1 * l1_loss + self.loss_weight_ssim * ssim_loss_value

    #     # --- 記錄個別 Loss 組件和總 Loss ---
    #     self.log("train_loss_mse", mse_loss, on_step=True, on_epoch=True, prog_bar=False)
    #     self.log("train_loss_l1", l1_loss, on_step=True, on_epoch=True, prog_bar=False)
    #     self.log("train_loss_ssim", ssim_loss_value, on_step=True, on_epoch=True, prog_bar=False) # 記錄 1-SSIM 的值
    #     self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True) # 記錄用於優化的總 Loss

    #     self.train_psnr_patch.update(restored_patches_flat, clean_patches_flat) 
    #     self.log("train_psnr_patch", self.train_psnr_patch, on_step=False, on_epoch=True, prog_bar=True) # 記錄在 Epoch 結束時

    #     return total_loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.001)
        # optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.001)
        # optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=8,max_epochs=80) 
        return [optimizer],[scheduler]

    # def on_load_checkpoint(self, checkpoint):
    #     """
    #     在從 checkpoint 載入狀態字典之前呼叫。
    #     這裡我們移除優化器和排程器的狀態，以便重新初始化。
    #     """
    #     print("從 checkpoint 載入模型權重，並重新初始化優化器和排程器。")
    #     if 'optimizer_states' in checkpoint:
    #         del checkpoint['optimizer_states']
    #     if 'lr_schedulers' in checkpoint:
    #         del checkpoint['lr_schedulers']

    # --- validation_step 方法保持不變：在 256x256 拼接結果上計算 PSNR ---
    def validation_step(self, batch, batch_idx):
        # 批次數據結構: (degraded_patches_batch, clean_patches_batch, clean_256_batch)
        # degraded_patches_batch 形狀: [DataLoader_batch_size, 4, C, 128, 128]
        # clean_patches_batch 形狀:   [DataLoader_batch_size, 4, C, 128, 128] (在這個 validation 邏輯中用不到)
        # clean_256_batch 形狀:   [DataLoader_batch_size, C, 256, 256] (用於 256 PSNR 目標)
        degraded_patches_batch, clean_patches_batch, clean_256_batch = batch # 接收批次中的所有元素

        # --- 將 degraded Patch 重塑為一個大的平坦批次送入模型 ---
        batch_size, num_patches, C, H, W = degraded_patches_batch.shape
        degraded_patches_flat = degraded_patches_batch.view(-1, C, H, W) # 形狀 [batch_size * 4, C, 128, 128]

        degraded_patches_flat = degraded_patches_flat.to(self.device)

        with torch.no_grad():
            restored_patches_flat = self.net(degraded_patches_flat) # 模型輸出 128x128

        # --- 將模型輸出移回目標設備 (與 clean_256_batch 相同) ---
        restored_patches_flat = restored_patches_flat.to(clean_256_batch.device)

        # --- 將恢復的 Patch 重塑回原來的批次結構 ---
        restored_patches_batch = restored_patches_flat.view(batch_size, num_patches, C, H, W) # 形狀 [batch_size, 4, C, 128, 128]

        # --- 將恢復的 Patch 拼接回 256x256 影像 ---
        # 使用 batch_stitch_patches_2x2 輔助函數
        restored_stitched_256_batch = batch_stitch_patches_2x2(restored_patches_batch) # 形狀 [batch_size, C, 256, 256]

        # --- 更新 PSNR 指標的狀態 ---
        # self.val_psnr 會累積所有 validation step 的狀態
        self.val_psnr.update(restored_stitched_256_batch, clean_256_batch)

        # val_loss = self.loss_fn(restored_stitched_256_batch, clean_256_batch) 
        # self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True) # 記錄在 Epoch 結束時

        ssim_value_256 = ssim(restored_stitched_256_batch, clean_256_batch, data_range=1.0, size_average=True)
        # self.log("val_ssim_256", ssim_value_256, on_step=False, on_epoch=True, prog_bar=True)

        # --- 計算並記錄驗證損失 (在 256x256 尺度，使用組合 Loss) ---
        val_loss_mse = self.loss_fn_mse(restored_stitched_256_batch, clean_256_batch)
        val_loss_l1 = self.loss_fn_l1(restored_stitched_256_batch, clean_256_batch)
        val_loss_total = self.loss_weight_mse * val_loss_mse + self.loss_weight_l1 * val_loss_l1 + self.loss_weight_ssim * (1-ssim_value_256) 
        
        # 記錄個別和總 Loss
        # self.log("val_loss_mse", val_loss_mse, on_step=False, on_epoch=True)
        # self.log("val_loss_l1", val_loss_l1, on_step=False, on_epoch=True)
        self.log("val_loss", val_loss_total, on_step=False, on_epoch=True, prog_bar=True) # 記錄總驗證 Loss

    def on_validation_epoch_end(self):
        """
        在每個驗證 Epoch 結束時呼叫。計算累積的 PSNR 並記錄。
        """
        # --- 計算並記錄平均的驗證 PSNR ---
        avg_val_psnr = self.val_psnr.compute()
        self.log("val_psnr", avg_val_psnr, on_step=False, on_epoch=True, prog_bar=True)

        # --- 重置 PSNR 指標的狀態，為下一個 Epoch 做準備 ---
        self.val_psnr.reset()

        # 如果你記錄了驗證損失並使用了指標，在這裡也重置它
        # self.val_loss_metric.reset() # 如果你沒有使用這個指標，就不需要重置

def main():
    
    LOCAL_SAVE_DIR = "../local/"

    if not os.path.exists(LOCAL_SAVE_DIR):
        os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

    SAVE_NAME = "finetune3/"
    logger = TensorBoardLogger(save_dir = LOCAL_SAVE_DIR + "runs/")

    DATA_DIR = "../hw4_realse_dataset/train"
    CHECKPOINT_DIR = LOCAL_SAVE_DIR + "params/" + SAVE_NAME

    BATCH_SIZE = 1
    EPOCHS = 80
    # resize_size = (128, 128)

    transform = T.Compose([
        # T.Resize(resize_size),
        T.ToTensor()
    ])
    
    CROP_SIZE = (128, 128)

    full_dataset = CustomDataset(DATA_DIR, transform=transform)
    random_crop_dataset = RandomDataset(DATA_DIR, transform=transform, crop_size=CROP_SIZE)

    TRAIN_BATCH_SIZE = 4 

    sample_types = full_dataset.img_type
    all_indices = list(range(len(full_dataset)))

    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=sample_types, # 這裡傳入包含每個樣本類型信息的列表
        random_state=63
    )

    # train_dataset = Subset(full_dataset, train_indices)
    train_dataset = Subset(random_crop_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"訓練集大小: {len(random_crop_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                           num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4)
    
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
    # return 
    # plot_and_save_image_pairs(train_dataset, num_pairs_to_plot=8, save_path="local/train_image_pairs_first_8.png")
    
    # finetune part
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "./finetune3/best_psnr-epoch=038-val_psnr=28.54" # 你的檢查點名稱
    MODEL_PATH = os.path.join('../local/params', MODEL_NAME + '.ckpt') # 檢查點完整路徑
    # model = PromptIRModel.load_from_checkpoint(
    #         resume_from_checkpoint=MODEL_PATH,
    #         #  checkpoint_path=MODEL_PATH,
    #          map_location=device,
    #     )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = PromptIRModel.load_from_checkpoint(
    #          checkpoint_path = 'local/params/test2/testbest_psnr-epoch=018-val_psnr=26.24.ckpt',
    #          map_location=device,
    #     )
    model = PromptIRModel()
    trainer = pl.Trainer( max_epochs=EPOCHS, accelerator="gpu", logger=logger, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model=model, ckpt_path=MODEL_PATH, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()


