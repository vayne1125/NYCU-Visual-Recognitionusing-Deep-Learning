import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
from torchvision.transforms import functional as F
# 導入 torchvision.transforms 模組，用於定義轉換序列
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    """
    處理 clean 影像和 degraded (有雨水) 影像配對的資料集類別。
    假設檔案結構為 root_dir/clean/ 和 root_dir/degraded/
    假設檔名規則為 clean: <base_name>_clean-<number>.png, degraded: <base_name>-<number>.png
    加入了影像轉換 (transform) 的功能。
    """
    def __init__(self, root_dir, transform=None):
        """
        初始化資料集。

        Args:
            root_dir (str): 資料集的根目錄路徑 (例如: 'hw4_realse_dataset/train')
            transform (callable, optional): 一個可呼叫的轉換函數，接收 PIL 影像並返回轉換後的影像。
                                            這個轉換會同時應用於 clean 和 degraded 影像。預設為 None。
        """
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.degraded_dir = os.path.join(root_dir, 'degraded')
        self.transform = transform # 儲存傳入的 transform

        # 尋找配對的影像檔案
        self.image_pairs = self._find_image_pairs()

    def _find_image_pairs(self):
        """
        遍歷 degraded 資料夾，尋找對應的 clean 影像並建立配對列表。
        """
        image_pairs = []

        # 檢查資料夾是否存在
        if not os.path.exists(self.degraded_dir):
            print(f"錯誤: 找不到 degraded 資料夾於 {self.degraded_dir}")
            return []
        if not os.path.exists(self.clean_dir):
             print(f"錯誤: 找不到 clean 資料夾於 {self.clean_dir}")
             return []

        # 獲取 degraded 資料夾中的所有 png 檔案 (忽略大小寫)
        degraded_files = [f for f in os.listdir(self.degraded_dir) if f.lower().endswith('.png')]
        # 對檔案列表進行排序，以確保不同作業系統或執行環境下檔案順序一致
        degraded_files.sort()

        # 編譯一個正規表達式，用於從 degraded 檔名中提取 <base_name> 和 <number>
        file_pattern = re.compile(r"(.*)-(\d+)\.png$", re.IGNORECASE)

        print(f"在 {self.degraded_dir} 中尋找影像配對...")

        for deg_file in degraded_files:
            match = file_pattern.match(deg_file)
            if match:
                name_prefix = match.group(1)
                number_str = match.group(2)
                expected_clean_file = f"{name_prefix}_clean-{number_str}.png"

                clean_path = os.path.join(self.clean_dir, expected_clean_file)
                degraded_path = os.path.join(self.degraded_dir, deg_file)

                if os.path.exists(clean_path):
                    image_pairs.append((clean_path, degraded_path))
                else:
                    print(f"警告: 找不到 degraded 影像 '{deg_file}' 對應的 clean 影像 '{expected_clean_file}'。跳過。")
            else:
                print(f"警告: degraded 影像檔名 '{deg_file}' 不符合預期模式 '*-N.png'。跳過。")

        if not image_pairs:
            print("錯誤: 未找到任何匹配的 clean/degraded 影像配對。請檢查資料夾結構和檔名。")
        else:
            print(f"找到 {len(image_pairs)} 對影像配對。")

        return image_pairs

    def __len__(self):
        """
        返回資料集中的影像配對數量。
        """
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        根據索引獲取一對影像。

        Args:
            idx (int): 影像對的索引。

        Returns:
            tuple: 包含兩個 Tensor 的元組 (degraded_tensor, clean_tensor)。
                   像素值範圍在 [0, 1] 之間，通道順序為 [C, H, W]。
                   在轉換後和轉換為 Tensor 之前。
        """
        clean_path, degraded_path = self.image_pairs[idx]

        # 使用 Pillow (PIL) 載入影像，並轉換為 RGB 格式
        clean_image = Image.open(clean_path).convert('RGB')
        degraded_image = Image.open(degraded_path).convert('RGB')

        # --- 在這裡應用轉換 ---
        if self.transform:
            clean_image = self.transform(clean_image)
            degraded_image = self.transform(degraded_image)

        # 使用 torchvision 的功能將 PIL 影像轉換為 PyTorch Tensor
        # F.to_tensor() 會自動將影像維度從 [H, W, C] 調整為 [C, H, W]
        # 並將像素值從 [0, 255] 縮放到 [0.0, 1.0] 的 FloatTensor
        # 注意：如果在 transform 中已經將 PIL 影像轉換為 Tensor，這裡就不需要再次使用 F.to_tensor()
        # 而是直接返回 transform 的結果。這裡假設 transform 返回的仍然是 PIL 影像。
        # 如果你的 transform 包含 ToTensor()，則需要修改這裡。
        # clean_tensor = F.to_tensor(clean_image)
        # degraded_tensor = F.to_tensor(degraded_image)


        # 通常在訓練時，我們將有雨水影像作為輸入 (input)，乾淨影像作為目標 (target)
        return degraded_image, clean_image