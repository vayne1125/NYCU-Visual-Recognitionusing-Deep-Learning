"""
This module defines PyTorch `Dataset` classes for handling image data,
specifically tailored for deep learning tasks involving image restoration
or image-to-image translation, particularly with degraded (e.g., rainy)
and clean image pairs.

It provides three main dataset implementations:

1.  **`CustomDataset` (for Training/Validation with Fixed Patches):**
    This dataset loads full-sized (256x256) clean and degraded image pairs
    and then **fixedly splits each into four non-overlapping 128x128 patches**.
    It is suitable for training or validation where a consistent, fixed patch
    extraction strategy is desired, ensuring that each original image
    contributes specific, known patches to the dataset.

2.  **`RandomDataset` (for Training with Random Patches and Augmentations):**
    Similar to `CustomDataset` in loading image pairs, but instead of fixed splits,
    it **randomly crops a single 128x128 patch pair** from each original 256x256 image.
    Crucially, it also applies **random geometric augmentations** (horizontal flip,
    vertical flip, and 0/180-degree rotation) to both the degraded and clean patches
    synchronously. This is ideal for robust training, as it increases data diversity
    and helps the model generalize better.

3.  **`TestImageDataset` (for Inference/Testing):**
    This dataset is designed for loading only the **degraded images from a test set**.
    It loads full-sized (256x256) images and returns them along with their original
    filenames, making it suitable for inference where full images need to be processed
    and results associated with their source files.

Additionally, the module includes a `natural_sort_key` helper function for
sorting filenames containing numerical parts in a human-friendly order,
which is crucial for consistent data loading.
"""
import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torchvision.transforms as T

from utils import natural_sort_key

class CustomDataset(Dataset):
    """
    Dataset class for handling pairs of clean and degraded (rainy) images.
    It fixedly splits each original 256x256 image into four 128x128 patches as samples.
    It also returns the original 256x256 clean image.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The root directory path of the dataset.
            transform (callable, optional): Transformations to apply to the
                                            **loaded 256x256 images** (e.g., ToTensor, Normalize).
        """
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.degraded_dir = os.path.join(root_dir, 'degraded')

        self.transform = transform
        self.img_type = []
        self.image_pairs = self._find_image_pairs()

    def _find_image_pairs(self):
        """
        Iterates through the degraded directory, finds paired images, and stores their types.
        """
        image_pairs = []
        # Use natural_sort_key for sorting (if available)
        try:
            degraded_files = sorted([f for f in os.listdir(
                self.degraded_dir) if f.lower().endswith('.png')], key=natural_sort_key)
        except NameError:
            print("Warning: natural_sort_key is undefined, using standard sorting.")
            degraded_files = sorted([f for f in os.listdir(
                self.degraded_dir) if f.lower().endswith('.png')])

        file_pattern = re.compile(r"(.*)-(\d+)\.png$", re.IGNORECASE)

        print(f"Searching for image pairs in {self.degraded_dir}...")

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
                    print(f"Warning: Corresponding clean image '{expected_clean_file}' "
                          f"not found for degraded image '{deg_file}'. Skipping.")
            else:
                print(f"Warning: Degraded image filename '{deg_file}' does not match "
                      f"the expected pattern '*-N.png'. Skipping.")

        if not image_pairs:
            print("Error: No matching clean/degraded image pairs found. "
                  "Please check folder structure and filenames.")
        else:
            print(f"Found {len(image_pairs)} image pairs.")
            if len(image_pairs) != len(self.img_type):
                print("Critical Error: Number of image pairs does not match number of types! "
                      "Please check _find_image_pairs logic.")

        return image_pairs

    def __len__(self):
        """
        Returns the number of image pairs (original 256x256 pairs) in the dataset.
        """
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Loads a 256x256 image pair by index, fixedly splits it into four 128x128 patches,
            and returns the degraded patches tensor, clean patches tensor, 
            and original 256x256 clean image.

        Args:
            idx (int): Index of the image pair (corresponding to an original 256x256 image pair).

        Returns:
            tuple: A tuple containing three Tensors.
                The first element is the degraded patches Tensor (shape [4, C, 128, 128]).
                The second element is the clean patches Tensor (shape [4, C, 128, 128]).
                The third element is the original 256x256 clean image Tensor (shape [C, 256, 256]).
        """
        clean_path, degraded_path = self.image_pairs[idx]

        clean_image_256_pil = Image.open(clean_path).convert('RGB')
        degraded_image_256_pil = Image.open(degraded_path).convert('RGB')

        # --- Apply transformations to the original 256x256 images ---
        if self.transform:
            degraded_image_256_tensor = self.transform(degraded_image_256_pil)
            clean_image_256_tensor = self.transform(clean_image_256_pil)
        else:
            degraded_image_256_tensor = T.ToTensor()(degraded_image_256_pil)
            clean_image_256_tensor = T.ToTensor()(clean_image_256_pil)

        # --- Fixedly split 256x256 degraded image into four 128x128 patches ---
        patch_size = 128
        degraded_patches_list = []
        degraded_patches_list.append(
            degraded_image_256_tensor[:, 0:patch_size, 0:patch_size])  # Top-left
        degraded_patches_list.append(
            degraded_image_256_tensor[:, 0:patch_size, patch_size:256])  # Top-right
        degraded_patches_list.append(
            degraded_image_256_tensor[:, patch_size:256, 0:patch_size])  # Bottom-left
        degraded_patches_list.append(
            degraded_image_256_tensor[:, patch_size:256, patch_size:256])  # Bottom-right
        degraded_patches_tensor = torch.stack(
            degraded_patches_list, dim=0)  # Shape [4, C, 128, 128]

        # --- Added: Fixedly split 256x256 clean image into four 128x128 patches ---
        clean_patches_list = []
        clean_patches_list.append(
            clean_image_256_tensor[:, 0:patch_size, 0:patch_size])  # Top-left
        clean_patches_list.append(
            clean_image_256_tensor[:, 0:patch_size, patch_size:256])  # Top-right
        clean_patches_list.append(
            clean_image_256_tensor[:, patch_size:256, 0:patch_size])  # Bottom-left
        clean_patches_list.append(
            clean_image_256_tensor[:, patch_size:256, patch_size:256])  # Bottom-right
        clean_patches_tensor = torch.stack(
            clean_patches_list, dim=0)  # Shape [4, C, 128, 128]

        return degraded_patches_tensor, clean_patches_tensor, clean_image_256_tensor


class RandomDataset(Dataset):
    """
    Dataset class for handling pairs of clean and degraded (rainy) images.
    It randomly crops one 128x128 patch from each original image as a training sample
    (Random Crop strategy).
    """

    def __init__(self, root_dir, transform=None, crop_size=(128, 128)):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The root directory path of the dataset.
            transform (callable, optional): Transformations to apply to the
                                            **cropped 128x128 patches** (e.g., ToTensor, Normalize).
            crop_size (tuple): The size of the random patch crop (height, width).
        """
        self.root_dir = root_dir
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.degraded_dir = os.path.join(root_dir, 'degraded')

        # Stores transformations applied to the **cropped patches** (e.g., ToTensor, Normalize)
        self.transform = transform
        self.crop_size = crop_size

        # Initialize list for image types, used for stratified splitting (unchanged)
        self.img_type = []
        self.image_pairs = self._find_image_pairs()

    def _find_image_pairs(self):
        """
        Iterates through the degraded directory, finds paired images, and stores their types.
        """
        image_pairs = []
        # Use natural_sort_key for sorting (if available)
        try:
            degraded_files = sorted([f for f in os.listdir(
                self.degraded_dir) if f.lower().endswith('.png')], key=natural_sort_key)
        except NameError:
            print("Warning: natural_sort_key is undefined, using standard sorting.")
            degraded_files = sorted([f for f in os.listdir(
                self.degraded_dir) if f.lower().endswith('.png')])

        file_pattern = re.compile(r"(.*)-(\d+)\.png$", re.IGNORECASE)

        print(f"Searching for image pairs in {self.degraded_dir}...")

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
                    print(f"Warning: Corresponding clean image '{expected_clean_file}' "
                          f"not found for degraded image '{deg_file}'. Skipping.")
            else:
                print(f"Warning: Degraded image filename '{deg_file}' does not match "
                      f"the expected pattern '*-N.png'. Skipping.")

        if not image_pairs:
            print("Error: No matching clean/degraded image pairs found. "
                  "Please check folder structure and filenames.")
        else:
            print(f"Found {len(image_pairs)} image pairs.")
            if len(image_pairs) != len(self.img_type):
                print("Critical Error: Number of image pairs does not match number of types! "
                      "Please check _find_image_pairs logic.")

        return image_pairs

    def __len__(self):
        """
        Returns the number of image pairs (original 256x256 pairs) in the dataset.
        """
        return len(self.image_pairs)

    # --- Modified __getitem__ method: Implements random cropping of one 128x128 patch pair ---
    def __getitem__(self, idx):
        """
        Loads a 256x256 image pair by index, randomly crops a 128x128 patch,
        and applies random geometric and color augmentations.

        Args:
            idx (int): Index of the image pair (corresponding to an original 256x256 image pair).

        Returns:
            tuple: A tuple containing two Tensors (degraded_patch_tensor, clean_patch_tensor).
                   Shape [C, H, W] (e.g., [3, 128, 128]), with pixel values in the range [0, 1].
        """
        clean_path, degraded_path = self.image_pairs[idx]

        # --- Load original 256x256 images using Pillow (PIL) ---
        clean_image_256_pil = Image.open(clean_path).convert('RGB')
        degraded_image_256_pil = Image.open(degraded_path).convert('RGB')

        # --- Get parameters for random crop (ensuring both images use the same crop location) ---
        i, j, h, w = T.RandomCrop.get_params(
            degraded_image_256_pil, output_size=self.crop_size)

        # --- Apply the same crop to both images using functional.crop ---
        degraded_patch_pil = F.crop(degraded_image_256_pil, i, j, h, w)
        clean_patch_pil = F.crop(clean_image_256_pil, i, j, h, w)

        # --- Random Geometric Augmentations ---
        # Random horizontal flip (50% probability)
        if torch.rand(1) < 0.5:
            degraded_patch_pil = F.hflip(degraded_patch_pil)
            clean_patch_pil = F.hflip(clean_patch_pil)

        # Random vertical flip (50% probability, recommended for increased diversity)
        if torch.rand(1) < 0.5:
            degraded_patch_pil = F.vflip(degraded_patch_pil)
            clean_patch_pil = F.vflip(clean_patch_pil)

        # Includes 0 degrees, meaning there's a chance of no rotation
        rotation_angles = [0, 180]
        chosen_angle_idx = torch.randint(0, len(rotation_angles), (1,)).item()
        chosen_angle = rotation_angles[chosen_angle_idx]

        if chosen_angle != 0:  # Only apply rotation if the angle is not 0
            degraded_patch_pil = F.rotate(
                degraded_patch_pil, chosen_angle, expand=False)
            clean_patch_pil = F.rotate(
                clean_patch_pil, chosen_angle, expand=False)

        if self.transform:
            degraded_patch_tensor = self.transform(degraded_patch_pil)
            clean_patch_tensor = self.transform(clean_patch_pil)
        else:
            degraded_patch_tensor = T.ToTensor()(degraded_patch_pil)
            clean_patch_tensor = T.ToTensor()(clean_patch_pil)

        # --- Return the processed 128x128 patch pair ---
        return degraded_patch_tensor, clean_patch_tensor


class TestImageDataset(Dataset):
    """
    Dataset class for handling degraded images in the test set.
    Returns the original image Tensor and the original filename.
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): The root directory path of the test set.
            transform (callable, optional): Transformations to apply to the **original image**,
                                            should include ToTensor().
        """
        self.image_dir = os.path.join(root_dir, 'degraded')
        self.transform = transform

        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower(
        ).endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        self.image_files.sort(key=natural_sort_key)
        if not self.image_files:
            print(f"Warning: No image files found in {self.image_dir}.")
        print(f"Found {len(self.image_files)} test images to load.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)
        img_pil = Image.open(img_path).convert(
            'RGB')  # Load the original 256x256 image

        if self.transform:
            img_transformed = self.transform(img_pil)
        else:
            img_transformed = F.to_tensor(img_pil)

        if not isinstance(img_transformed, torch.Tensor):
            raise TypeError(
                f"Transform result for {filename} is not a Tensor.")

        return img_transformed, filename
