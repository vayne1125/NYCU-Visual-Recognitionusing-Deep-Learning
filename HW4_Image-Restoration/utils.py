import matplotlib.pyplot as plt
import torch
import numpy as np
import os
# 假設你的 CustomDataset 類別在一個叫做 your_dataset_file.py 的檔案中
# from your_dataset_file import CustomDataset

def plot_and_save_image_pairs(dataset, num_pairs_to_plot=8, save_path="image_pairs_plot.png"):
    """
    從資料集中繪製前 num_pairs_to_plot 組影像配對 (degraded vs clean)，並存為圖片檔案。

    Args:
        dataset (torch.utils.data.Dataset): 已經載入資料的資料集物件 (你的 CustomDataset 實例)。
                                             假設 dataset[i] 返回一個 tuple (degraded_tensor, clean_tensor)。
        num_pairs_to_plot (int): 要繪製的影像配對數量 (預設為 8)。
        save_path (str): 要儲存的圖片檔案路徑 (預設為 "image_pairs_plot.png")。
    """
    
    # 確定實際要繪製的數量，不超過資料集總數
    actual_pairs_to_plot = min(num_pairs_to_plot, len(dataset))

    if actual_pairs_to_plot == 0:
        print("資料集中沒有影像配對可以繪製。")
        return

    print(f"正在繪製前 {actual_pairs_to_plot} 組影像配對並將存儲到 {save_path} ...")

    # 創建一個 Figure 和一個子圖網格
    # 每行繪製一對影像 (degraded 和 clean)，共 actual_pairs_to_plot 行，2 列
    # figsize 根據影像數量調整，以確保圖不會太小看不清
    # 這裡假設影像大小適中，可以根據需要調整 figsize 的寬度 (第一個值) 和高度 (第二個值)
    fig, axes = plt.subplots(actual_pairs_to_plot, 2, figsize=(10, actual_pairs_to_plot * 5))

    for i in range(actual_pairs_to_plot):
        # 從資料集中獲取第 i 對影像
        # 這裡獲取到的 degraded_tensor 和 clean_tensor 是已經經過正規化的數據
        degraded_tensor, clean_tensor = dataset[i]

        # --- 在這裡進行反正規化 (Denormalize) ---
        # 需要使用你在 T.Normalize() 中完全相同的 mean 和 std 值
        # 將 mean 和 std 轉換為 Tensor，並調整維度形狀以便於廣播計算 ([C, 1, 1])
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # 應用反正規化公式: 原始值 = (正規化值 * 標準差) + 均值
        degraded_tensor_denorm = degraded_tensor * std + mean
        clean_tensor_denorm = clean_tensor * std + mean

        # 反正規化後，像素值理論上會回到原始範圍 (如果是從 [0, 1] 正規化的話)。
        # 但為了安全起見，特別是在浮點數計算中，通常會將值截斷 (clamp) 到 [0, 1] 範圍，
        # 以確保符合 imshow 的預期。
        degraded_tensor_denorm = torch.clamp(degraded_tensor_denorm, 0.0, 1.0)
        clean_tensor_denorm = torch.clamp(clean_tensor_denorm, 0.0, 1.0)
        # --- 反正規化結束 ---


        # 現在將 反正規化 後的 Tensor 轉換為 Matplotlib 需要的 [H, W, C] 的 NumPy 陣列
        degraded_np = degraded_tensor_denorm.permute(1, 2, 0).numpy()
        clean_np = clean_tensor_denorm.permute(1, 2, 0).numpy()

        # --- 後面的繪圖程式碼使用 degraded_np 和 clean_np，保持不變 ---
        if actual_pairs_to_plot == 1:
            ax_deg = axes[0]
            ax_clean = axes[1]
        else:
            ax_deg = axes[i, 0]
            ax_clean = axes[i, 1]

        ax_deg.imshow(degraded_np)
        ax_deg.set_title(f'Degraded {i+1}')
        ax_deg.axis('off')

        ax_clean.imshow(clean_np)
        ax_clean.set_title(f'Clean {i+1}')
        ax_clean.axis('off')

    # 自動調整子圖參數，使之填充整個 figure 區域，並防止標題/標籤重疊
    plt.tight_layout()

    # 儲存整個 Figure 到指定的檔案路徑
    plt.savefig(save_path)

    # 關閉 Figure 以釋放記憶體，避免在程式運行中顯示出來
    plt.close(fig)

    print(f"影像配對圖已成功儲存到 {save_path}")

# --- 如何使用這個函式 ---

# 假設你已經定義並實例化了你的資料集：
# from your_dataset_file import CustomDataset # 請確保你的 CustomDataset 類別在這裡或同一個檔案中
# DATA_DIR = '/path/to/your/hw4_realse_dataset/train' # 替換為你實際的資料集路徑
# train_dataset = CustomDataset(DATA_DIR)

# 呼叫函式來繪製並儲存前 8 組影像配對
# plot_and_save_image_pairs(train_dataset, num_pairs_to_plot=8, save_path="train_image_pairs_first_8.png")

# 如果你想繪製不同數量，例如前 5 組：
# plot_and_save_image_pairs(train_dataset, num_pairs_to_plot=5, save_path="train_image_pairs_first_5.png")