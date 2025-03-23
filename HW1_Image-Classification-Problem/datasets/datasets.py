"""
This module contains two custom dataset classes:
- CustomDataset: Loads images and labels from a directory structure where
  each subdirectory represents a class.
- TestDataset: Loads images from a directory for testing or inference,
  without requiring labels.
"""

import os

from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    """
    A custom dataset class that loads images and their corresponding labels
    from a directory where each subfolder is a class.

    Args:
        data_dir (str): Directory containing subdirectories for each class.
        transform (callable, optional): A function/transform to apply to the images.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []  # To store the paths of all images
        self.labels = []  # To store the labels of all images

        # Read the folder and sort by folder names numerically
        class_names = sorted(os.listdir(data_dir), key=int)  # Sort class folders by name

        self.class_names = class_names

        for label, class_name in enumerate(class_names):
            class_folder = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder):
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and label at the specified index.
        
        Args:
            idx (int): The index of the item to retrieve.
        
        Returns:
            tuple: (image, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')  # Open image and convert to RGB format

        if self.transform:
            img = self.transform(img)  # Apply transformation if any

        return img, label
