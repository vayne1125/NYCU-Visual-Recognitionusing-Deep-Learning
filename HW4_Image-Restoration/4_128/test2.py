import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import re # 用於自然排序檔案名
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import math # 用於計算 Gaussian 權重

# 確保你導入了你的 LightningModule 類別 (PromptIRModel)
from train import PromptIRModel # 替換為你定義 PromptIRModel 的檔案名
# 確保你導入了你的模型架構類別 (PromptIR)，如果它在單獨的 model.py 中
# from model import PromptIR


# --- 輔助函數：自然排序鍵 ---
def natural_sort_key(s):
    """ 用於自然排序檔案名的鍵函數。 """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# --- 測試集 Dataset 類別 (修改為返回原始 256x256 影像 Tensor) ---
class TestImageDataset(Dataset):
    """
    處理測試集 degraded 影像的資料集。返回原始影像 Tensor 和原始檔案名。
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 測試集的根目錄路徑。
            transform (callable, optional): 應用於**原始影像**的轉換，應包含 ToTensor()。
        """
        self.image_dir = os.path.join(root_dir, 'degraded')
        # 應確保 transform 序列包含 T.ToTensor() 作為最後一步或在後續手動轉換
        self.transform = transform

        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        self.image_files.sort(key=natural_sort_key)
        if not self.image_files:
            print(f"警告: 在 {self.image_dir} 中未找到任何影像檔案。")
        print(f"找到 {len(self.image_files)} 張測試影像進行載入。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        img_pil = Image.open(img_path).convert('RGB') # 載入原始 256x256 影像

        if self.transform:
            # 應用 transform，應包含將 PIL Image 轉為 Tensor 的步驟
            img_transformed = self.transform(img_pil)
        else:
             # 如果沒有提供 transform，則只進行 ToTensor (返回 [0, 1] 範圍 Tensor)
             img_transformed = F.to_tensor(img_pil) # 返回 [C, 256, 256] Tensor

        if not isinstance(img_transformed, torch.Tensor):
             raise TypeError(f"Transform result for {filename} is not a Tensor.")

        return img_transformed, filename # 返回原始尺寸的影像 Tensor 和檔案名


# --- 輔助函數：生成 Gaussian 權重遮罩 (用於融合) ---
def generate_gaussian_weight_mask_3d(patch_size, sigma=None):
    """
    生成一個用於 Patch 融合的 3D Gaussian 權重遮罩 [1, patch_size, patch_size]。
    權重中心為 1，向邊緣衰減。
    """
    if sigma is None:
        sigma = patch_size / 6.0 # 默認 sigma 設置

    center = patch_size / 2.0
    y = torch.arange(patch_size, dtype=torch.float32)
    x = torch.arange(patch_size, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    distance_sq = (x_grid - center)**2 + (y_grid - center)**2
    weight_mask = torch.exp(-distance_sq / (2 * sigma**2))
    # 將權重縮放到 [0, 1] 範圍，最大值在中心
    weight_mask = weight_mask / torch.max(weight_mask)
    return weight_mask.unsqueeze(0) # 形狀 [1, patch_size, patch_size]


# --- 推斷執行函式 (基於 Patch 處理) ---
def run_inference(lightning_model, test_dataloader, output_file_path, device, patch_size, stride):
    """
    對 DataLoader 中的原始影像進行 Patch 處理推斷，將結果儲存到 .npz 檔案。
    使用重疊 Patch 和 Gaussian 融合。
    假設模型輸入輸出尺寸均為 patch_size，模型輸出在 [0, 1] 範圍。

    Args:
        lightning_model (PromptIRModel): 載入的 LightningModule 實例。
        test_dataloader (DataLoader): 返回原始尺寸影像 Tensor 和檔案名的 DataLoader。
        output_file_path (str): 儲存 .npz 檔案路徑。
        device (torch.device): 推斷設備。
        patch_size (int): 模型訓練時的輸入尺寸 (e.g., 128)。
        stride (int): 提取 Patch 時的步長 (e.g., 64)，決定重疊大小。
    """
    print(f"正在對 DataLoader 中的影像進行 Patch 推斷並儲存到: {output_file_path}")
    print(f"Patch 尺寸: {patch_size}x{patch_size}, 步長: {stride}")

    lightning_model.to(device)
    lightning_model.eval()
    inference_model = lightning_model.net # 獲取底層 PromptIR 模型實例

    results_dict = {}

    # 生成 Patch 融合用的 Gaussian 權重遮罩，並移到設備
    weight_mask = generate_gaussian_weight_mask_3d(patch_size).to(device)

    print(f"將處理 {len(test_dataloader)} 個批次，共 {len(test_dataloader.dataset)} 張影像。")

    with torch.no_grad():
        for batch_idx, (original_image_batch_256, filename_batch) in enumerate(test_dataloader):
            # original_image_batch_256: [batch_size, C, 256, 256] Tensor ([0, 1] 範圍)
            # filename_batch: list of batch_size filenames

            if (batch_idx + 1) % 20 == 0:
                print(f"正在處理批次 {batch_idx + 1}/{len(test_dataloader)}...")

            # 將原始影像批次移到設備 (雖然 Patch 提取在 CPU，但為了後續方便)
            original_image_batch_256 = original_image_batch_256.to(device)

            batch_size = original_image_batch_256.shape[0]
            # --- 處理批次中的每一張原始影像 ---
            for i in range(batch_size):
                original_image_256 = original_image_batch_256[i] # [C, 256, 256]

                # --- 提取重疊的 Patch ---
                patches_list = []
                coords_list = [] # 儲存每個 Patch 對應在原圖中的左上角座標 (y, x)

                img_h, img_w = original_image_256.shape[1:]

                # 計算 Patch 的左上角座標
                # 確保即使步長不整除，也能覆蓋到影像邊緣
                y_coords = list(range(0, img_h - patch_size + stride, stride))
                if (img_h - patch_size) % stride != 0: # 如果最後一個 Patch 的起始位置不是 stride 的倍數
                    y_coords.append(img_h - patch_size) # 添加最後一個 Patch 的起始位置

                x_coords = list(range(0, img_w - patch_size + stride, stride))
                if (img_w - patch_size) % stride != 0: # 如果最後一個 Patch 的起始位置不是 stride 的倍數
                     x_coords.append(img_w - patch_size) # 添加最後一個 Patch 的起始位置

                # 確保座標唯一且排序
                y_coords = sorted(list(set(y_coords)))
                x_coords = sorted(list(set(x_coords)))


                for y in y_coords:
                     for x in x_coords:
                         patch = original_image_256[:, y:y+patch_size, x:x+patch_size] # [C, patch_size, patch_size]
                         patches_list.append(patch)
                         coords_list.append((y, x)) # 儲存座標

                # --- 準備 Patch 批次進行推斷 ---
                if not patches_list:
                    print(f"警告: 無法為影像 {filename_batch[i]} 提取 Patch。跳過。")
                    continue

                patch_batch = torch.stack(patches_list, dim=0) # [num_patches, C, patch_size, patch_size]

                # --- 對 Patch 批次進行推斷 ---
                restored_patch_batch = inference_model(patch_batch) # [num_patches, C, patch_size, patch_size]

                # --- 重新拼接 Patch 並融合 ---
                # 將推斷結果移回 CPU (或留在 GPU 如果需要，並相應調整 canvas 的設備)
                restored_patch_batch_cpu = restored_patch_batch.cpu() # [num_patches, C, patch_size, patch_size]
                weight_mask_cpu = weight_mask.cpu() # [1, patch_size, patch_size]

                # 初始化輸出畫布和權重畫布
                # 畫布尺寸與原始影像相同 [C, 256, 256]
                output_canvas = torch.zeros_like(original_image_256).cpu() # 確保在 CPU
                weight_canvas = torch.zeros((1, img_h, img_w), dtype=torch.float32).cpu() # 確保在 CPU [1, 256, 256]

                # 遍歷每個恢復的 Patch 及其原始座標，添加到畫布上
                for j in range(restored_patch_batch_cpu.shape[0]):
                    restored_patch = restored_patch_batch_cpu[j] # [C, patch_size, patch_size]
                    y, x = coords_list[j] # Patch 左上角座標

                    # 將權重遮罩廣播到與 Patch 相同的通道數，並應用權重
                    weighted_patch = restored_patch * weight_mask_cpu # [C, patch_size, patch_size]

                    # 將加權後的 Patch 疊加到輸出畫布的對應位置
                    output_canvas[:, y:y+patch_size, x:x+patch_size] += weighted_patch

                    # 將權重遮罩（或其廣播版本）疊加到權重畫布的對應位置
                    # 這裡 weight_mask_cpu 是 [1, ps, ps]，加到 [1, H, W] 上是正確的廣播
                    weight_canvas[:, y:y+patch_size, x:x+patch_size] += weight_mask_cpu


                # 進行正規化融合：將輸出畫布除以權重畫布 (避免除以零)
                # 將權重畫布廣播到與輸出畫布相同的通道數進行除法
                blended_result_256 = output_canvas / (weight_canvas.expand_as(output_canvas) + 1e-8) # 添加 epsilon 避免除以零

                # --- 最終後處理：截斷，縮放到 [0, 255]，轉換為 uint8 NumPy ---
                # 截斷到 [0, 1] 範圍
                image_0_1_256 = torch.clamp(blended_result_256, 0.0, 1.0)

                # 縮放到 [0, 255]
                image_0_255_256 = image_0_1_256 * 255.0

                # 轉換為 uint8 NumPy 陣列
                restored_np_256 = image_0_255_256.numpy().astype(np.uint8) # [C, 256, 256], uint8

                # 將原始檔案名作為 key，256x256 結果作為 value 存入字典
                results_dict[filename_batch[i]] = restored_np_256

            # ... 批次處理進度打印 ...

    print("\n所有批次處理完成。")

    # --- 儲存結果 ---
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        np.savez(output_file_path, **results_dict)
        print(f"推斷結果已成功儲存到 {output_file_path}")
    except Exception as e:
        print(f"儲存 .npz 檔案 {output_file_path} 失敗: {e}")


# --- 繪圖函式 (保持不變) ---
def plot_restored_images(npz_file_path, num_images_to_plot=12, plot_save_path="restored_images_plot.png"):
    """ 從 .npz 結果檔案中載入影像，繪製前 num_images_to_plot 張圖片。 """
    if not os.path.exists(npz_file_path):
        print(f"錯誤: 找不到 .npz 結果檔案於 {npz_file_path}")
        return

    print(f"正在從 {npz_file_path} 載入結果並繪製前 {num_images_to_plot} 張圖片到 {plot_save_path}...")
    try:
        results_data = np.load(npz_file_path)
    except Exception as e:
        print(f"載入 .npz 檔案 {npz_file_path} 失敗: {e}")
        return

    filenames = sorted(results_data.files, key=natural_sort_key)
    images_to_plot = []
    titles = []
    actual_images_to_plot = min(num_images_to_plot, len(filenames))

    if actual_images_to_plot == 0:
        print(".npz 檔案中沒有影像可以繪製。")
        results_data.close()
        return

    for i in range(actual_images_to_plot):
        filename = filenames[i]
        image_np = results_data[filename] # (3, H, W), uint8
        image_np_hwc = image_np.transpose(1, 2, 0) # (H, W, 3), uint8
        images_to_plot.append(image_np_hwc)
        titles.append(filename)

    n_cols = 4
    n_rows = (actual_images_to_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if (n_rows > 1 or n_cols > 1) else [axes]

    for i in range(actual_images_to_plot):
        ax = axes[i]
        ax.imshow(images_to_plot[i])
        ax.set_title(titles[i], fontsize=8)
        ax.axis('off')

    for j in range(actual_images_to_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    plot_output_dir = os.path.dirname(plot_save_path)
    if plot_output_dir and not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir, exist_ok=True)

    try:
        plt.savefig(plot_save_path)
        print(f"影像結果圖已成功儲存到 {plot_save_path}")
    except Exception as e:
        print(f"儲存繪圖檔案 {plot_save_path} 失敗: {e}")

    plt.close(fig)
    results_data.close()


# --- 主執行區塊 ---
if __name__ == "__main__":
    # --- 配置 ---
    OUTPUT_DIR = '../local/results'
    TEST_DATA_ROOT = '../hw4_realse_dataset/test/' # 你的測試集根目錄
    SAVE_NAME = "./finetune3/best_psnr-epoch=037-val_psnr=28.54" # 你的檢查點名稱
    MODEL_PATH = os.path.join('../local/params', SAVE_NAME + '.ckpt') # 檢查點完整路徑
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'pred.npz') # .npz 輸出路徑
    PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, 'restored_images_grid.png') # 繪圖輸出路徑

    # --- Patch 配置 ---
    PATCH_SIZE = 128 # 模型訓練時的輸入尺寸
    STRIDE = 32   # Patch 提取的步長 (決定重疊大小)。STRIDE < PATCH_SIZE 產生重疊。

    # --- DataLoader 配置 ---
    # 推斷批次大小：這裡是指一次從 DataLoader 載入的原始影像數量
    # 處理 Patch 時，會從這些原始影像中提取所有 Patch，形成一個更大的 Patch 批次送入模型
    INFERENCE_BATCH_SIZE = 1 # 根據你的 GPU 記憶體調整，不要太大以免 Patch 批次過大
    INFERENCE_NUM_WORKERS = 4 # 數據載入工作進程數

    # --- 測試集轉換 ---
    # 應用於原始影像的轉換：只需轉為 Tensor [0, 1]
    # Patch 提取在 run_inference 內部處理
    inference_transform = T.Compose([
         T.ToTensor(), # 將 PIL Image 轉為 Tensor [0, 1]
         # 如果訓練時做了 Normalize，這裡不需要反向操作，模型輸出是正規化後的
         # 如果訓練時沒有 Normalize，模型輸入輸出都是 [0, 1]
     ])

    # --- 設備設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # --- 載入模型 (PromptIRModel) ---
    print(f"正在嘗試從 {MODEL_PATH} 載入模型...")
    try:
        # 使用 PromptIRModel.load_from_checkpoint 載入整個 LightningModule 實例
        # 如果 PromptIRModel.__init__ 需要參數 (例如 decoder_arg=True)，則在這裡傳入：
        # model = PromptIRModel.load_from_checkpoint(
        #     checkpoint_path=MODEL_PATH,
        #     map_location=device,
        #     decoder_arg=True # <--- 假設 PromptIRModel __init__ 接收這個參數
        # )
        # 假設你的 PromptIRModel __init__ 是不接收參數的，並且在裡面初始化了 PromptIR(decoder=True)
        model = PromptIRModel.load_from_checkpoint(
             checkpoint_path=MODEL_PATH,
             map_location=device,
        )
        print(f"模型載入成功從 {MODEL_PATH}")

    except FileNotFoundError:
        print(f"錯誤: 找不到模型檢查點檔案於 {MODEL_PATH}")
        exit()
    except Exception as e:
        print(f"載入模型檢查點 {MODEL_PATH} 時發生錯誤: {e}")
        exit()

    # 將載入的 PromptIRModel 實例設定為評估模式
    model.eval()

    # --- 創建測試集 Dataset 和 DataLoader ---
    test_dataset = TestImageDataset(TEST_DATA_ROOT, transform=inference_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=INFERENCE_BATCH_SIZE, # DataLoader 一次提供幾個原始影像
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS,
        persistent_workers=True
    )

    # --- 運行推斷 (Patch 處理版本) ---
    run_inference(model, test_dataloader, OUTPUT_FILE_PATH, device, PATCH_SIZE, STRIDE)

    # --- 繪製並保存結果圖片 ---
    plot_restored_images(OUTPUT_FILE_PATH, num_images_to_plot=12, plot_save_path=PLOT_SAVE_PATH)

    print("推斷腳本運行結束。")