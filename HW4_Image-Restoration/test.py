import torch
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import re # 用於自然排序檔案名
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

# 確保你導入了你的 LightningModule 類別 (PromptIRModel)
from train import PromptIRModel # 替換為你定義 PromptIRModel 的檔案路徑
# 確保你導入了你的模型架構類別 (PromptIR)，如果它在單獨的 model.py 中
# from model import PromptIR


# --- 輔助函數：自然排序鍵 ---
def natural_sort_key(s):
    """ 用於自然排序檔案名的鍵函數。 """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# --- 測試集 Dataset 類別 (保持不變，但 __getitem__ 返回的 Tensor 將是 128x128 因為 transform 的改變) ---
class TestImageDataset(Dataset):
    """
    處理測試集 degraded 影像的資料集。返回影像 Tensor 和原始檔案名。
    """
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, 'degraded')
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        self.image_files.sort(key=natural_sort_key)
        if not self.image_files:
            print(f"警告: 在 {self.image_dir} 中未找到任何影像檔案。")
        # print(f"找到 {len(self.image_files)} 張測試影像進行載入。")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        img_pil = Image.open(img_path).convert('RGB')

        if self.transform:
            img_transformed = self.transform(img_pil)
        else:
            img_transformed = F.to_tensor(img_pil) # 默認只做 ToTensor

        if not isinstance(img_transformed, torch.Tensor):
             raise TypeError(f"Transform result for {filename} is not a Tensor.")

        return img_transformed, filename


# --- 推斷執行函式 (使用 DataLoader，並添加後處理放大) ---
def run_inference(lightning_model, test_dataloader, output_file_path, device):
    """
    對指定 DataLoader 中的影像執行推斷，將結果儲存到 .npz 檔案。
    假設模型輸入為 128x128，輸出為 128x128，並將結果放大到 256x256。
    假設模型輸出在 [0, 1] 範圍。
    """
    print(f"正在對 DataLoader 中的影像執行推斷並儲存到: {output_file_path}")

    lightning_model.to(device)
    lightning_model.eval()

    results_dict = {}
    inference_model = lightning_model.net # 獲取底層 PromptIR 模型實例

    print(f"將處理 {len(test_dataloader)} 個批次，共 {len(test_dataloader.dataset)} 張影像。")

    with torch.no_grad():
        for batch_idx, (image_batch, filename_batch) in enumerate(test_dataloader):
            if (batch_idx + 1) % 50 == 0:
                print(f"正在處理批次 {batch_idx + 1}/{len(test_dataloader)}...")

            image_batch = image_batch.to(device)

            # 運行模型推斷 (輸入 128x128 批次)
            restored_batch_128 = inference_model(image_batch) # 模型輸出 128x128 批次

            # --- 後處理：假設模型輸出在 [0, 1]，縮放到 [0, 255]，轉換為 uint8 NumPy 陣列 ---
            restored_batch_cpu_128 = restored_batch_128.cpu()
            image_0_1_batch_128 = torch.clamp(restored_batch_cpu_128, 0.0, 1.0)
            image_0_255_batch_128 = image_0_1_batch_128 * 255.0
            restored_batch_np_uint8_128 = image_0_255_batch_128.numpy().astype(np.uint8) # [batch_size, 3, 128, 128], uint8

            # --- 遍歷批次中的每張圖片，放大並存入字典 ---
            current_batch_size = restored_batch_np_uint8_128.shape[0]
            for i in range(current_batch_size):
                single_image_np_128 = restored_batch_np_uint8_128[i] # [3, 128, 128], uint8
                corresponding_filename = filename_batch[i]

                # --- 將 128x128 影像放大到 256x256 ---
                # 將 (3, 128, 128) NumPy 陣列轉回 PIL Image (H, W, C)
                single_image_np_128_hwc = single_image_np_128.transpose(1, 2, 0) # [128, 128, 3]
                img_pil_128 = Image.fromarray(single_image_np_128_hwc, 'RGB')

                # 放大 PIL Image 到 256x256
                # 可以選擇不同的縮放算法，例如 Image.Resampling.BICUBIC 是常見的
                img_pil_256 = img_pil_128.resize((256, 256), Image.Resampling.BICUBIC)

                # 將 256x256 的 PIL Image 轉回 NumPy 陣列 (H, W, C)，再轉為 (C, H, W) 給 .npz
                restored_np_256_hwc = np.array(img_pil_256) # [256, 256, 3], uint8
                restored_np_256_chw = restored_np_256_hwc.transpose(2, 0, 1) # [3, 256, 256], uint8

                # 儲存 256x256 結果到字典
                results_dict[corresponding_filename] = restored_np_256_chw

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


# --- 繪圖函式 (保持不變，因為載入的 .npz 已經是 256x256 的結果了) ---
def plot_restored_images(npz_file_path, num_images_to_plot=12, plot_save_path="restored_images_plot.png"):
    """
    從 .npz 結果檔案中載入影像，繪製前 num_images_to_plot 張圖片。
    """
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
    OUTPUT_DIR = './local/results'
    SAVE_NAME = "test2/test//best_psnr-epoch=002-val_psnr=28.35" # 你的檢查點名稱
    TEST_DATA_ROOT = './hw4_realse_dataset/test/' # 你的測試集根目錄
    MODEL_PATH = os.path.join('./local/params', SAVE_NAME + '.ckpt') # 檢查點完整路徑
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'pred.npz') # .npz 輸出路徑
    PLOT_SAVE_PATH = os.path.join(OUTPUT_DIR, 'restored_images_grid.png') # 繪圖輸出路徑

    # --- DataLoader 配置 ---
    INFERENCE_BATCH_SIZE = 1 # 推斷批次大小 (根據 GPU 記憶體調整)
    INFERENCE_NUM_WORKERS = 4 # 數據載入工作進程數

    # --- 測試集轉換 ---
    # 輸入模型前的轉換：縮小到 128x128，然後轉 Tensor
    # 原始影像是 256x256
    INPUT_MODEL_SIZE = (128, 128) # 模型訓練時的輸入尺寸 (高, 寬)
    FINAL_OUTPUT_SIZE = (256, 256) # 要求的最終輸出尺寸 (高, 寬)

    inference_transform = T.Compose([
         # 前處理：將原始影像縮小到模型訓練的尺寸
        #  T.Resize(INPUT_MODEL_SIZE), # 將 PIL Image 縮放到 128x128
         T.ToTensor(), # 將 PIL Image 轉為 Tensor [0, 1]
         # 如果訓練時做了 Normalize，這裡也需要加上 (通常不需要)
         # T.Normalize(mean=[...], std=[...])
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
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS,
        persistent_workers=True
    )

    # --- 運行推斷 ---
    run_inference(model, test_dataloader, OUTPUT_FILE_PATH, device)

    # --- 繪製並保存結果圖片 ---
    plot_restored_images(OUTPUT_FILE_PATH, num_images_to_plot=12, plot_save_path=PLOT_SAVE_PATH)

    print("推斷腳本運行結束。")