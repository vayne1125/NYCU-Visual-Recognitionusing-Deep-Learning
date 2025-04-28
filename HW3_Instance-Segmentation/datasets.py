import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json

from utils import decode_maskobj

class CustomDataset(Dataset):
    def __init__(self, annotation_file_path, image_dir, image_ids_subset=None, transform=None):
        """
        Args:
            annotation_file_path (str): Path to the COCO format annotation JSON file (e.g., './data/train.json').
            image_dir (str): Path to the root directory containing the image folders (e.g., './data/train/').
                             這個目錄下應該包含 generate_coco_annotations 產生的那些子資料夾。
            transform (callable, optional): Optional transform to be applied on a sample.
                                            這個 transform 應該能處理 PIL Image 並返回 Tensor。
                                            如果包含空間變換 (如 Resize)，需要額外處理 target。
        """
        self.annotation_file_path = annotation_file_path
        self.image_dir = image_dir
        self.transform = transform
        self.image_ids_subset = image_ids_subset # 儲存圖片 ID 子集
        
        self._load_annotations()

    def _load_annotations(self):
        """Loads and organizes annotations from the COCO JSON file."""
        print(f"Loading annotations from {self.annotation_file_path}...")
        with open(self.annotation_file_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        print("Annotations loaded successfully.")

        # 儲存 COCO JSON 中的主要資訊
        self.images_info = coco_data['images']
        self.annotations_info = coco_data['annotations']
        self.categories_info = coco_data['categories']

        # 建立 image_id 到 image_info 字典，方便根據 ID 查詢圖片資訊
        self.image_id_to_info = {img['id']: img for img in self.images_info}

        # 建立 image_id 到其對應的 annotations 列表的字典，方便快速獲取某圖片的所有標註
        self.image_id_to_annotations = {}
        for ann in self.annotations_info:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # 根據 image_ids_subset 篩選要使用的圖片 ID
        if self.image_ids_subset is not None:
            # 只保留在子集中的圖片 ID，並且這些 ID 必須實際存在於 annotation 文件中
            self.image_ids = sorted([img_id for img_id in self.image_ids_subset if img_id in self.image_id_to_info])
            print(f"Using a subset of {len(self.image_ids)} images.")
        else:
            # 如果沒有提供子集，使用所有有標註的圖片 ID，並排序
            self.image_ids = sorted(list(self.image_id_to_annotations.keys()))
            print(f"Using all {len(self.image_ids)} images with annotations.")

        # 建立 category_id 到 category_name 的映射字典
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories_info}
        # print(f"Found {len(self.images_info)} images, {len(self.annotations_info)} annotations, and {len(self.categories_info)} categories.")


    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding annotations for a given index.
        Args:
            idx (int): Index (0-based) of the image to retrieve from the sorted image_ids list.
        Returns:
            tuple: (image, target) where image is a Tensor and target is a dict.
                   target 包含 "boxes", "labels", "masks", "image_id" 等 tensor。
        """
        # 根據索引 idx 獲取對應的圖片 ID
        image_id = self.image_ids[idx]

        # 獲取圖片資訊字典
        image_info = self.image_id_to_info[image_id]
        # 圖片檔案路徑 (相對於 image_dir)
        image_file_name_relative = image_info['file_name']
        # 圖片的完整路徑
        img_path = os.path.join(self.image_dir, image_file_name_relative)

        # 使用 PIL 讀取圖片並轉換為 RGB 格式
        # 我們通常在 transform 中進行 ToTensor 轉換
        image = Image.open(img_path).convert("RGB")
        # 獲取原始圖片尺寸，在 target 中有用
        orig_img_w, orig_img_h = image.size


        # 獲取這張圖片的所有標註
        annotations = self.image_id_to_annotations.get(image_id, [])

        # 初始化列表來儲存提取出的 bbox, labels, masks
        boxes = []          # 儲存 [x_min, y_min, x_max, y_max] 格式的 bbox
        labels = []         # 儲存類別 ID
        masks_list = []     # 儲存二值 mask (numpy array)

        for ann in annotations:
            # 提取邊界框 (COCO 格式是 [x, y, width, height])
            bbox_coco = ann['bbox']
            # 將 COCO 格式的 bbox 轉換為 PyTorch torchvision 模型期望的 [x_min, y_min, x_max, y_max] 格式
            x_min, y_min, w, h = bbox_coco
            boxes.append([x_min, y_min, x_min + w, y_min + h])

            # 提取類別 ID
            labels.append(ann['category_id'])

            # 提取分割資訊並解碼為二值 mask (numpy array)
            segmentation = ann['segmentation']
            try:
                binary_mask = decode_maskobj(segmentation)
                masks_list.append(binary_mask) # binary_mask 是 HxW 的 numpy array (uint8)
            except ValueError as e:
                print(f"Error decoding segmentation for annotation ID {ann.get('id', 'N/A')} in image ID {image_id}: {e}. Skipping this mask.")
                pass

        # 將列表轉換為 PyTorch Tensors
        # 邊界框 Tensor，形狀 (num_instances, 4)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        # 類別標籤 Tensor，形狀 (num_instances,)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        # 將二值 mask (numpy array) 列表堆疊為一個 Tensor，形狀 (num_instances, H, W)
        # Mask R-CNN 模型通常期望 uint8 格式的 masks
        if masks_list: # 只有當 masks 列表非空時才堆疊
            masks_tensor = torch.stack([torch.as_tensor(mask, dtype=torch.uint8) for mask in masks_list])
        else:
             # 如果沒有任何 masks，返回一個空的 masks tensor
             # 需要指定正確的尺寸 (num_instances=0, H, W)
             masks_tensor = torch.empty((0, orig_img_h, orig_img_w), dtype=torch.uint8)

        # 創建訓練所需的 target 字典
        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["masks"] = masks_tensor
        # 圖片 ID Tensor，形狀 (1,)
        target["image_id"] = torch.tensor([image_id])
        # (可選) 添加面積和 iscrowd 資訊到 target 中
        # target["area"] = torch.as_tensor([ann['area'] for ann in annotations], dtype=torch.float32)
        # target["iscrowd"] = torch.as_tensor([ann['iscrowd'] for ann in annotations], dtype=torch.int64)
        # 添加原始圖片尺寸，在應用 transform 時非常有用
        target["orig_size"] = torch.as_tensor([orig_img_h, orig_img_w])


        # --- 應用 transform ---
        if self.transform is not None:
            image = self.transform(image)

        # 確保返回的圖片是 Tensor
        # 如果 transform 包含 ToTensor，這步會多餘
        if not isinstance(image, torch.Tensor):
             image = transforms.ToTensor()(image)

        return image, target
