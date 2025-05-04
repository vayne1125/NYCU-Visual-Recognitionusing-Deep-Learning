"""
utils.py

This module contains utility functions used throughout the object detection pipeline,
including:

- Converting model predictions to COCO JSON format for evaluation
- Saving model checkpoints and training logs
- Plotting training and validation loss/accuracy
- Visualizing predicted bounding boxes and segmentation masks

These tools are essential for training, evaluating, and debugging deep learning models
such as Faster R-CNN or Mask R-CNN.
"""
import os
import random

import numpy as np
import skimage.io as sio

import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


def convert_to_coco_format(targets, outputs):
    """
    Convert ground truth and model outputs to COCO format for evaluation.

    This function takes a batch of targets and outputs (from a Mask R-CNN model),
    converts them into the COCO-style ground truth (GT) and detection result (DT)
    dictionaries, and returns two `pycocotools.coco.COCO` objects for evaluation.

    Args:
        targets (list of dict): List of ground truth annotations per image. 
            Each dict should contain:
                - 'masks' (Tensor): Boolean mask for each object.
                - 'labels' (Tensor): Category label for each object.
                - 'boxes' (Tensor): Bounding boxes for each object.
        outputs (list of dict): List of predicted annotations per image.
            Each dict should contain:
                - 'masks' (Tensor): Predicted mask for each object.
                - 'labels' (Tensor): Predicted category label for each object.
                - 'boxes' (Tensor): Predicted bounding boxes.
                - 'scores' (Tensor): Confidence scores for each prediction.

    Returns:
        tuple:
            - coco_true (COCO): Ground truth COCO object.
            - coco_pred (COCO): Detection result COCO object, for evaluation.

    Notes:
        - The image_id is assigned using the batch index (0, 1, 2, ...).
        - COCO-style RLE encoding is applied to masks.
        - bbox is in [x, y, width, height] format as required by COCO.
    """
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []

    category_ids = set()

    for idx, (target, output) in enumerate(zip(targets, outputs)):
        image_id = idx  # 可以是 idx，或真實的 image id

        coco_gt["images"].append({
            "id": image_id,
            "width": target["masks"].shape[-1],
            "height": target["masks"].shape[-2],
        })

        masks = target["masks"].numpy()
        labels = target["labels"].numpy()
        boxes = target["boxes"].numpy()

        for i in range(len(masks)):
            rle = maskUtils.encode(np.asfortranarray(masks[i]))
            rle["counts"] = rle["counts"].decode("utf-8")  # json 可存
            coco_gt["annotations"].append({
                "id": len(coco_gt["annotations"]),
                "image_id": image_id,
                "category_id": int(labels[i]),
                "segmentation": rle,
                "bbox": list(boxes[i]),
                "area": float(maskUtils.area(rle)),
                "iscrowd": 0,
            })
            category_ids.add(int(labels[i]))

        pred_masks = output["masks"].numpy()
        pred_labels = output["labels"].numpy()
        pred_boxes = output["boxes"].numpy()
        pred_scores = output["scores"].numpy()

        for i in range(len(pred_masks)):
            rle = maskUtils.encode(np.asfortranarray(pred_masks[i, 0] > 0.5))
            rle["counts"] = rle["counts"].decode("utf-8")
            coco_dt.append({
                "image_id": image_id,
                "category_id": int(pred_labels[i]),
                "segmentation": rle,
                "bbox": list(pred_boxes[i]),
                "score": float(pred_scores[i]),
            })

    coco_gt["categories"] = [
        {"id": cid, "name": str(cid)} for cid in sorted(category_ids)]

    coco_true = COCO()
    coco_true.dataset = coco_gt
    coco_true.createIndex()

    coco_pred = coco_true.loadRes(coco_dt)

    return coco_true, coco_pred


def plot_training_history(history, save_name):
    """
    Plot the training history of loss and mAP, then save the results as an image.

    Args:
        history (dict): Training history, a dictionary containing loss, 
                        validation loss, and validation mAP.
        save_name (str): Path to save the plot image
    """
    # Set the figure size
    # Adjusted figure size for better visualization
    plt.figure(figsize=(10, 4))

    # --- Plot the Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='train loss')
    # plt.plot(history['val_loss'], label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # --- Plot the mAP ---
    plt.subplot(1, 2, 2)
    # Using a different color for clarity
    plt.plot(history['val_mask_map'], label='valid mask mAP', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.ylim([0, 1.0])  # mAP is typically between 0 and 1
    plt.grid(True)
    plt.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Ensure the directory exists for saving the image
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    # Save the plot
    plt.savefig(save_name, format='png')
    print(f"Plot saved to {save_name}")

    plt.close()


def save_model_info(params_count, trained_model_str, save_name):
    """
    Save training model information to a specified file.

    Args:
        params_count (int): Total number of model parameters
        trained_model_str (str): Model architecture description
        save_name (str): File name to save the information
    """
    # Check if the folder exists, create it if not
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    # Write the model information to a file
    with open(save_name, 'w', encoding='utf-8') as f:
        f.write("Total Parameters: ")
        f.write(str(params_count))
        f.write("\n\nTrained Model:\n")
        f.write(trained_model_str)

    print(f"Model and parameter count saved to {save_name}\n")


# Define the color map for the classes (RGB tuples, 0-255)
# Class 0: Background (White)
# Class 1: 粉色 (Pink)
# Class 2: 天空藍 (Sky Blue)
# Class 3: 淺黃色 (Light Yellow)
# Class 4: 淺綠色 (Light Green)
# Add colors for any other classes if needed, including background (class 0)
# The class IDs here should match the category_id in your COCO JSON.
# Assuming category_ids are 1, 2, 3, 4 for the cells.
CLASS_COLORS = {
    0: (255, 255, 255),  # Background - White
    1: (255, 192, 203),  # Class 1 - Pink
    2: (135, 206, 235),  # Class 2 - Sky Blue
    3: (255, 255, 224),  # Class 3 - Light Yellow
    4: (144, 238, 144)  # Class 4 - Light Green
    # Add more classes if you have them
}


def print_image_with_mask_for_segmentation(dataset, save_dir="segmentation_viz"):
    """
    This function displays a set of images from the instance segmentation dataset
    along with their corresponding segmentation masks, colored by class.
    It saves the resulting image grid to a specified directory.

    Args:
        dataset (CustomDataset): The dataset containing images and their targets.
        save_dir (str, optional): 
            Directory to save the displayed image grid. Defaults to "segmentation_viz".
    """
    # Ensure the number of images to display does not exceed the dataset size
    # Fixed number of images to display
    N = 3

    if len(dataset) == 0:
        print("Dataset is empty or N is 0. Cannot display images.")
        return

    # Determine the grid size (e.g., 1x3 for 3 images)
    rows = 1
    cols = N

    fig, axes = plt.subplots(rows, cols, figsize=(
        cols * 4, rows * 4))  # Adjust figsize as needed
    # If N=1, axes is not a numpy array, handle this case
    if N == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Mean and std for unnormalization (match your train_transform/val_transform)
    # These should match the values used in your transforms.Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # --- Generate N unique random indices ---
    # random.sample performs sampling without replacement
    random_indices = random.sample(range(len(dataset)), N)

    # Display images and their corresponding masks
    for i, idx in enumerate(random_indices):
        ax = axes[i]

        # Randomly select an image index from the dataset
        img_tensor, target = dataset[idx]

        # Unnormalize the image tensor
        # Ensure mean and std are on the same device as img_tensor if using GPU
        if img_tensor.device != mean.device:
            mean = mean.to(img_tensor.device)
            std = std.to(img_tensor.device)

        img_tensor = img_tensor * std + mean
        # Clamp values to be in [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Convert image tensor (CxHxW, float) to numpy array (HxWxC, uint8) for visualization
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy()
                  * 255).astype(np.uint8)

        # Get masks and labels from the target
        # Shape (num_instances, H, W), dtype uint8 (0 or 1)
        masks = target['masks']
        labels = target['labels']  # Shape (num_instances,), dtype int64

        # Get original image dimensions from target (useful if image was resized)
        # If not resized in dataset __getitem__, this will be original size
        if 'orig_size' in target:
            img_h, img_w = target['orig_size'].tolist()
        else:
            # Fallback if orig_size is not in target (less robust)
            img_h, img_w = img_np.shape[:2]

        # Create a composite mask image (HxWx3, uint8)
        # Initialize with background color (White)
        composite_mask_np = np.full(
            (img_h, img_w, 3), CLASS_COLORS[0], dtype=np.uint8)

        # Overlay each instance mask onto the composite mask
        # Iterate through masks and labels for this image
        for mask_tensor, label_tensor in zip(masks, labels):
            # Mask is HxW, uint8 (0 or 1)
            mask_np = mask_tensor.cpu().numpy()
            label_id = label_tensor.item()

            # Get the color for this class
            # Use CLASS_COLORS.get(label_id, CLASS_COLORS[0]) to handle unknown labels
            color = CLASS_COLORS.get(label_id, CLASS_COLORS[0])

            # Find pixels belonging to this instance mask
            # np.where returns (row_indices, col_indices)
            # Use > 0 in case mask is not strictly 0/1
            y_coords, x_coords = np.where(mask_np > 0)

            # Apply the color to these pixels in the composite mask
            if len(y_coords) > 0:  # Ensure there are foreground pixels
                composite_mask_np[y_coords, x_coords] = color

        # Blend the original image and the composite mask
        # alpha controls the transparency of the mask
        # (0.0 is fully transparent, 1.0 is fully opaque)
        alpha = 0.8  # Adjust transparency as needed
        # Ensure both arrays are float for blending
        blended_img_np = (img_np.astype(np.float32) * (1 - alpha) +
                          composite_mask_np.astype(np.float32) * alpha).astype(np.uint8)

        # Display the blended image
        ax.imshow(blended_img_np)

        # --- Set title to file_name and img_id ---
        # Get the image_id from the target
        img_id = target['image_id'].item()
        # Use the image_id to get image info from the dataset's mapping
        # Assuming dataset has image_id_to_info attribute populated from JSON
        if hasattr(dataset, 'image_id_to_info') and img_id in dataset.image_id_to_info:
            image_info = dataset.image_id_to_info[img_id]
            # Get file_name, fallback to ID if not found
            file_name = image_info.get('file_name', f"ID: {img_id}")
            # Combine file_name and img_id in the title
            # Set title to file name and ID
            ax.set_title(f"ID: {img_id}\nFile: {file_name}", fontsize=8)
        else:
            # Fallback if image_id_to_info is not available or ID not found
            ax.set_title(f"Image ID: {img_id}", fontsize=8)

        ax.axis('off')  # Turn off axes

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    temp_image_path = os.path.join(
        save_dir, f"images_with_segmentation_masks_{N}.png")
    plt.tight_layout()  # Adjust layout to prevent titles/labels overlapping
    plt.savefig(temp_image_path)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Images with segmentation masks saved to {temp_image_path}")


def split_dataset_ids(all_image_ids, train_size=0.8, random_state=42):
    """
    Splits a list of image IDs into training and validation sets.

    Args:
        all_image_ids (list): A list of all unique image IDs (integers).
        train_size (float): The proportion of the dataset to include in the train split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        tuple: (train_ids, val_ids) where train_ids and val_ids are lists of image IDs.
    """
    # train_test_split 函式會將輸入列表隨機打亂後分割
    train_ids, val_ids = train_test_split(
        all_image_ids,
        train_size=train_size,
        random_state=random_state,
        shuffle=True  # 確保數據被打亂
    )
    return train_ids, val_ids


def set_seed(seed=63):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def decode_maskobj(mask_obj):
    """
    Decode a mask object in RLE (Run-Length Encoding) format into a binary mask.

    Args:
        mask_obj (dict): A dictionary representing the RLE-encoded mask object.
    
    Returns:
        numpy.ndarray: A decoded binary mask represented as a NumPy array.
    """
    return maskUtils.decode(mask_obj)


def encode_mask(binary_mask):
    """
    Encode a binary mask into RLE (Run-Length Encoding) format.

    Args:
        binary_mask (numpy.ndarray): A binary mask represented as a NumPy array.
    
    Returns:
        dict: A dictionary containing the RLE-encoded mask, with 'counts' as a UTF-8 encoded string.
    """
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = maskUtils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    """
    Read a mask file and return the mask data as a NumPy array.

    Args:
        filepath (str): The path to the mask file to be read.
    
    Returns:
        numpy.ndarray: A NumPy array representing the mask read from the file.
    """
    mask_array = sio.imread(filepath)
    return mask_array
