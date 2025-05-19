import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import re
from torchvision.transforms import functional as F
import torchvision.transforms as T

# 輔助函數：自然排序鍵 (假設可用或使用標準排序)
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


class CustomDataset(Dataset): # 用於訓練和驗證
    """
    處理 clean 影像和 degraded (有雨水) 影像配對的資料集類別。
    從每張原始影像中固定分割成四個 128x128 Patch 作為樣本。
    同時返回原始 256x256 乾淨影像。
    """
    # __init__ 方法與之前類似，transform 是應用於 256x256 影像的（例如 ToTensor, Normalize）
    def __init__(self, root_dir, transform=None):
        """
        初始化資料集。

        Args:
            root_dir (str): 資料集的根目錄路徑。
            transform (callable, optional): 應用於**載入後的 256x256 影像**的轉換 (例如 ToTensor, Normalize)。
        """
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.degraded_dir = os.path.join(root_dir, 'degraded')
        
        # 儲存應用於載入後的 256x256 影像的轉換 (例如 ToTensor, Normalize)
        self.transform = transform 
        
        # 初始化影像類型列表，用於分層分割
        self.img_type = [] 

        # 尋找配對的影像檔案
        self.image_pairs = self._find_image_pairs()

    # _find_image_pairs 方法與之前修正後的一致 (保持不變)
    def _find_image_pairs(self):
        """ 遍歷 degraded 資料夾，尋找配對並儲存類型 """
        image_pairs = []
        # 使用 natural_sort_key 進行排序 (如果可用)
        try:
             degraded_files = sorted([f for f in os.listdir(self.degraded_dir) if f.lower().endswith('.png')], key=natural_sort_key)
        except NameError:
             print("警告: natural_sort_key 未定義，使用標準排序。")
             degraded_files = sorted([f for f in os.listdir(self.degraded_dir) if f.lower().endswith('.png')])

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
                    self.img_type.append(name_prefix)
                else:
                    print(f"警告: 找不到 degraded 影像 '{deg_file}' 對應的 clean 影像 '{expected_clean_file}'。跳過。")
            else:
                print(f"警告: degraded 影像檔名 '{deg_file}' 不符合預期模式 '*-N.png'。跳過。")

        if not image_pairs:
            print("錯誤: 未找到任何匹配的 clean/degraded 影像配對。請檢查資料夾結構和檔名。")
        else:
            print(f"找到 {len(image_pairs)} 對影像配對。")
            if len(image_pairs) != len(self.img_type):
                 print("嚴重錯誤: 影像配對數量與類型數量不符！請檢查 _find_image_pairs 邏輯。") 

        return image_pairs


    def __len__(self):
        """
        返回資料集中的影像配對數量 (原始 256x256 對)。
        """
        return len(self.image_pairs)

    # 修改 __getitem__ 方法，實現固定分割成 4 個 Patch
    def __getitem__(self, idx):
        """
        根據索引載入 256x256 影像，固定分割成 4 個 128x128 Patch，
        並返回 degraded Patch 列表和原始 256x256 clean 影像。

        Args:
            idx (int): 影像對的索引 (對應原始 256x256 影像對)。

        Returns:
            tuple: 包含兩個 Tensor 的元組。
                   第一個元素是 degraded Patch 的 Tensor (形狀 [4, C, 128, 128])。
                   第二個元素是原始 256x256 clean 影像的 Tensor (形狀 [C, 256, 256])。
        """
        clean_path, degraded_path = self.image_pairs[idx]

        clean_image_256_pil = Image.open(clean_path).convert('RGB')
        degraded_image_256_pil = Image.open(degraded_path).convert('RGB')

        # --- 應用轉換到原始 256x256 影像 ---
        if self.transform:
             degraded_image_256_tensor = self.transform(degraded_image_256_pil)
             clean_image_256_tensor = self.transform(clean_image_256_pil)
        else:
             degraded_image_256_tensor = T.ToTensor()(degraded_image_256_pil)
             clean_image_256_tensor = T.ToTensor()(clean_image_256_pil)

        # --- 固定分割 256x256 degraded 影像成四個 128x128 Patch ---
        patch_size = 128
        degraded_patches_list = []
        degraded_patches_list.append(degraded_image_256_tensor[:, 0:patch_size, 0:patch_size]) # 左上
        degraded_patches_list.append(degraded_image_256_tensor[:, 0:patch_size, patch_size:256]) # 右上
        degraded_patches_list.append(degraded_image_256_tensor[:, patch_size:256, 0:patch_size]) # 左下
        degraded_patches_list.append(degraded_image_256_tensor[:, patch_size:256, patch_size:256]) # 右下
        degraded_patches_tensor = torch.stack(degraded_patches_list, dim=0) # 形狀 [4, C, 128, 128]

        # --- 添加: 固定分割 256x256 clean 影像成四個 128x128 Patch ---
        clean_patches_list = []
        clean_patches_list.append(clean_image_256_tensor[:, 0:patch_size, 0:patch_size]) # 左上
        clean_patches_list.append(clean_image_256_tensor[:, 0:patch_size, patch_size:256]) # 右上
        clean_patches_list.append(clean_image_256_tensor[:, patch_size:256, 0:patch_size]) # 左下
        clean_patches_list.append(clean_image_256_tensor[:, patch_size:256, patch_size:256]) # 右下
        clean_patches_tensor = torch.stack(clean_patches_list, dim=0) # 形狀 [4, C, 128, 128]


        # --- 返回 degraded Patch Tensor, clean Patch Tensor, 和 原始 256x256 clean 影像 Tensor ---
        return degraded_patches_tensor, clean_patches_tensor, clean_image_256_tensor


from torchvision.transforms import functional as F
import torchvision.transforms as T 

class RandomDataset(Dataset):
    """
    處理 clean 影像和 degraded (有雨水) 影像配對的資料集類別。
    從每張原始影像中隨機裁剪一個 128x128 Patch 作為訓練樣本 (Random Crop 策略)。
    """
    # 修改 __init__ 方法，接受 transform 和 crop_size 參數
    def __init__(self, root_dir, transform=None, crop_size=(128, 128)):
        """
        初始化資料集。

        Args:
            root_dir (str): 資料集的根目錄路徑。
            transform (callable, optional): 應用於**裁剪出的 128x128 Patch** 的轉換 (例如 ToTensor, Normalize)。
            crop_size (tuple): 隨機裁剪的 Patch 尺寸 (高, 寬)。
        """
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.degraded_dir = os.path.join(root_dir, 'degraded')

        # 儲存應用於**裁剪出的 Patch** 的轉換 (例如 ToTensor, Normalize)
        self.transform = transform 
        # 儲存裁剪尺寸
        self.crop_size = crop_size 

        # 初始化影像類型列表，用於分層分割 (保持不變)
        self.img_type = [] 

        # 尋找配對的影像檔案 (保持不變)
        self.image_pairs = self._find_image_pairs()

    # _find_image_pairs 方法與之前修正後的一致 (保持不變)
    def _find_image_pairs(self):
        """ 遍歷 degraded 資料夾，尋找配對並儲存類型 """
        image_pairs = []
        # 使用 natural_sort_key 進行排序 (如果可用)
        try:
             degraded_files = sorted([f for f in os.listdir(self.degraded_dir) if f.lower().endswith('.png')], key=natural_sort_key)
        except NameError:
             print("警告: natural_sort_key 未定義，使用標準排序。")
             degraded_files = sorted([f for f in os.listdir(self.degraded_dir) if f.lower().endswith('.png')])

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
                    self.img_type.append(name_prefix)
                else:
                    print(f"警告: 找不到 degraded 影像 '{deg_file}' 對應的 clean 影像 '{expected_clean_file}'。跳過。")
            else:
                print(f"警告: degraded 影像檔名 '{deg_file}' 不符合預期模式 '*-N.png'。跳過。")

        if not image_pairs:
            print("錯誤: 未找到任何匹配的 clean/degraded 影像配對。請檢查資料夾結構和檔名。")
        else:
            print(f"找到 {len(image_pairs)} 對影像配對。")
            if len(image_pairs) != len(self.img_type):
                 print("嚴重錯誤: 影像配對數量與類型數量不符！請檢查 _find_image_pairs 邏輯。") 

        return image_pairs


    def __len__(self):
        """
        返回資料集中的影像配對數量 (原始 256x256 對)。
        """
        return len(self.image_pairs)

    # --- 修改 __getitem__ 方法：實現隨機裁剪一個 128x128 Patch 對 ---
    def __getitem__(self, idx):
        """
        根據索引載入 256x256 影像，隨機裁剪一個 128x128 Patch，
        並應用隨機幾何和顏色增強。

        Args:
            idx (int): 影像對的索引 (對應原始 256x256 影像對)。

        Returns:
            tuple: 包含兩個 Tensor 的元組 (degraded_patch_tensor, clean_patch_tensor)。
                   形狀 [C, H, W] (例如 [3, 128, 128])，像素值範圍在 [0, 1] 之間。
        """
        clean_path, degraded_path = self.image_pairs[idx]

        # --- 使用 Pillow (PIL) 載入原始 256x256 影像 ---
        clean_image_256_pil = Image.open(clean_path).convert('RGB')
        degraded_image_256_pil = Image.open(degraded_path).convert('RGB')

        # --- 獲取隨機裁剪的參數 (確保兩個影像使用相同的裁剪位置) ---
        i, j, h, w = T.RandomCrop.get_params(degraded_image_256_pil, output_size=self.crop_size)

        # --- 使用 functional.crop 應用相同的裁剪到兩張影像 ---
        degraded_patch_pil = F.crop(degraded_image_256_pil, i, j, h, w)
        clean_patch_pil = F.crop(clean_image_256_pil, i, j, h, w)

        # --- 隨機幾何增強 ---
        # 注意：幾何增強應用到 PIL 影像，確保 clean 和 degraded 使用相同的變換參數

        # 隨機水平翻轉 (50% 概率)
        if torch.rand(1) < 0.5: 
            degraded_patch_pil = F.hflip(degraded_patch_pil)
            clean_patch_pil = F.hflip(clean_patch_pil)
        
        # 隨機垂直翻轉 (50% 概率，建議加入以增加多樣性)
        if torch.rand(1) < 0.5: 
            degraded_patch_pil = F.vflip(degraded_patch_pil)
            clean_patch_pil = F.vflip(clean_patch_pil)

        # 隨機固定角度旋轉 (從 0, 90, 180, 270 中隨機選一個角度)
        # 建議包含所有 90 度倍數的旋轉，以提高模型對方向變化的魯棒性
        # rotation_angles = [0, 90, 180, 270] # 包含 0 度，表示有概率不旋轉
        rotation_angles = [0, 180] # 包含 0 度，表示有概率不旋轉
        chosen_angle_idx = torch.randint(0, len(rotation_angles), (1,)).item()
        chosen_angle = rotation_angles[chosen_angle_idx]

        if chosen_angle != 0: # 只在角度不是 0 時才應用旋轉
             # F.rotate 默認 expand=False，保持原尺寸，邊緣可能會有黑邊或裁剪
             # 對於訓練 Patch 通常沒問題
             degraded_patch_pil = F.rotate(degraded_patch_pil, chosen_angle, expand=False) 
             clean_patch_pil = F.rotate(clean_patch_pil, chosen_angle, expand=False)


        # --- 隨機顏色增強 (可選，但通常有益) ---
        # 如果你希望應用顏色抖動，可以在這裡添加
        # 注意：這裡使用 ColorJitter 類別，它的隨機性是內部的
        # 如果需要確保 clean 和 degraded 應用完全相同的隨機抖動，
        # 則需要更複雜的方式 (例如手動調整參數)。
        # 通常直接應用 ColorJitter 實例到兩個 Patch 已經足夠了。
        # if torch.rand(1) < 0.8: # 例如 80% 的概率應用顏色抖動
        #     # 定義顏色抖動的範圍
        #     color_jitter_transform = T.ColorJitter(
        #         brightness=[0.8, 1.2], # 亮度範圍 [0.8, 1.2]
        #         contrast=[0.8, 1.2],   # 對比度範圍 [0.8, 1.2]
        #         saturation=[0.8, 1.2], # 飽和度範圍 [0.8, 1.2]
        #         hue=[-0.1, 0.1]        # 色調範圍 [-0.1, 0.1]
        #     )
        #     degraded_patch_pil = color_jitter_transform(degraded_patch_pil)
        #     clean_patch_pil = color_jitter_transform(clean_patch_pil)


        # --- 應用剩餘的轉換 (例如 ToTensor, Normalize) 到增強後的 Patch ---
        # self.transform 應該包含 T.ToTensor() 和可能的 Normalize()
        if self.transform:
             degraded_patch_tensor = self.transform(degraded_patch_pil)
             clean_patch_tensor = self.transform(clean_patch_pil)
        else:
             # 如果沒有提供 transform，使用默認轉換 (只做 ToTensor)
             degraded_patch_tensor = T.ToTensor()(degraded_patch_pil)
             clean_patch_tensor = T.ToTensor()(clean_patch_pil)

        # --- 返回處理後的 128x128 Patch 對 ---
        return degraded_patch_tensor, clean_patch_tensor