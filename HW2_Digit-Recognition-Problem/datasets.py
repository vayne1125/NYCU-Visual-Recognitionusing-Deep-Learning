"""
This module defines the CustomDataset class for loading image and annotation data.
"""
import json
import os

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class CustomDataset(Dataset):
    """
    Custom dataset class for loading data with images and annotations.
    """
    def __init__(self, json_path, img_dir, transform=None):
        """
        Initializes the CustomDataset object.

        Args:
            json_path (str): Path to the annotation JSON file.
            img_dir (str): Directory where the image files are located.
            transform (callable, optional): Optional image transformation function. 
                                            Defaults to None.
        """
        self.json_path = json_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.data = self._load_data()
        self.img_ids = list(self.data['images'].keys())
        self.annotations = self.data['annotations']

    def _load_data(self):
        """
        Loads image and annotation data from the JSON file.

        Returns:
            dict: A dictionary containing 'images' (dictionary of image info) 
                  and 'annotations' (dictionary of annotation info).
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
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
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The total number of images in the dataset.
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Retrieves the image and annotation at the specified index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the image (Tensor) and target annotations (dict).
        """
        img_id = self.img_ids[idx]
        img_info = self.data['images'][img_id]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        annotations = self.annotations.get(img_id, [])
        boxes = []
        labels = []

        for ann in annotations:
            bbox = ann['bbox']
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transform is not None:
            image = self.transform(image)  # Apply torchvision transforms here

        return image, target
