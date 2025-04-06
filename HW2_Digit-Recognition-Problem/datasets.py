import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.transforms import functional as F
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        self.json_path = json_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.data = self._load_data()
        self.img_ids = list(self.data['images'].keys())
        self.annotations = self.data['annotations']
        self.cat_to_label = {cat_id: i for i, cat_id in enumerate(sorted(self.get_all_category_ids()))}
        self.label_to_cat = {i: cat_id for cat_id, i in self.cat_to_label.items()}

    def _load_data(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        images = {img['id']: img for img in data['images']}
        annotations = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(ann)
        return {'images': images, 'annotations': annotations}

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.data['images'][img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        width = img_info['width']
        height = img_info['height']

        annotations = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            width = bbox[2]
            height = bbox[3]
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax]) # 不再歸一化
            labels.append(self.cat_to_label[ann['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])

        if self.transform is not None:
            image = self.transform(image) # Apply torchvision transforms here

        return image, target


    def get_all_category_ids(self):
        category_ids = set()
        for image_annotations in self.data['annotations'].values():
            for ann in image_annotations:
                category_ids.add(ann['category_id'])
        return list(category_ids)

    def get_category_mapping(self):
        return self.cat_to_label

    def get_label_mapping(self):
        return self.label_to_cat