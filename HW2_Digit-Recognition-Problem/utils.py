"""
utils.py

This module contains utility functions used for training, testing,
and evaluating machine learning models.
It includes functions for visualizing training progress, saving model parameters, and setting
random seeds for reproducibility.

Functions:
    - print_image_with_label: Displays images and their labels.
    - save_model_info: Saves model information such as parameters and architecture to a file.
    - plot_training_history: Plots training history, including loss and accuracy over epochs.
    - set_seed: Sets the random seed for reproducibility.
    - load_config: Loads configuration settings from a YAML file.
"""
import os
import random
import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import yaml

def visualize_test_predictions(images, predictions, image_filenames, class_names, save_dir="temp_test_predictions"):
    """
    Visualizes predictions for the test set in a grid format, similar to validation visualization.

    Args:
        images (list of torch.Tensor): List of input images (tensors).
        predictions (list of list of dict): A list where each element corresponds to an image.
                                         Each element is a list of dictionaries, where each dictionary
                                         represents a single detection with keys: 'bbox', 'category_id', 'score'.
        image_filenames (list of str): List of corresponding image filenames.
        class_names (list of str): List of class names.
        save_dir (str, optional): Directory to save the visualizations.
                                 Defaults to "temp_test_predictions".
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    num_images = len(images)
    rows = 2  # 固定顯示 2 行
    cols = (num_images + 1) // rows  # 根據圖片數量計算列數

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # 建立子圖
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)  # ImageNet mean
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)  # ImageNet std

    for i in range(num_images):
        ax = axes[i]
        img_tensor = images[i].cpu()

        # 反歸一化
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)

        ax.imshow(img_tensor.permute(1, 2, 0))
        title = ""
        if image_filenames and i < len(image_filenames):
            title = f"{image_filenames[i]}\n"
        ax.set_title(title)
        ax.axis('off')

        # 畫 prediction boxes
        image_preds = predictions[i]  # Get predictions for the current image

        for pred in image_preds:
            bbox = pred['bbox']
            category_id = pred['category_id']
            score = pred['score']

            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height

            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1,
                                     edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            if class_names is not None and len(class_names) > category_id - 1:
                label_name = class_names[category_id - 1]  # category_id is 1-indexed
                label_text = f"{label_name}({score:.2f})"
                ax.text(xmin, ymin - 5, s=label_text, color='red', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8))

    save_path = os.path.join(save_dir, "test_predictions.png")  # Save all test images to one file
    plt.tight_layout()  # 自動調整子圖參數，使之填充整個畫布
    plt.savefig(save_path)
    plt.close()
    print(f"Test predictions saved to {save_path}")

def visualize_predictions(images, preds, gt_boxes=None, gt_labels=None, class_names=None, save_dir="temp_val_predictions", epoch=0, batch_idx=0, image_filenames=None):
    """Visualizes a batch of images with predicted and ground truth bounding boxes and labels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"epoch_{epoch}_batch_{batch_idx}_predictions.png")
    num_images = len(images)
    rows = 2
    cols = (num_images + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for i in range(num_images):
        ax = axes[i]
        img_tensor = images[i].cpu()

        # 反歸一化
        img_tensor = img_tensor * std + mean

        # 將 tensor 的值限制在 0 到 1 之間
        img_tensor = torch.clamp(img_tensor, 0, 1)

        ax.imshow(img_tensor.permute(1, 2, 0))
        title = ""
        if image_filenames and i < len(image_filenames):
            title = f"{image_filenames[i]}\n"
        ax.set_title(title)
        ax.axis('off')

        # 畫 ground truth boxes
        if gt_boxes is not None and gt_labels is not None and i < len(gt_boxes):
            gt_boxes_for_image = gt_boxes[i].cpu().tolist()
            gt_labels_for_image = gt_labels[i].cpu().tolist()
            for j, box in enumerate(gt_boxes_for_image):
                xmin, ymin, xmax, ymax = box
                rect_xmin = xmin
                rect_ymin = ymin
                rect_width = xmax - xmin
                rect_height = ymax - ymin
                rect = patches.Rectangle((rect_xmin, rect_ymin), rect_width, rect_height, linewidth=1, edgecolor='blue', facecolor='none') # 使用藍色標示 GT 框
                ax.add_patch(rect)
                if class_names is not None and len(class_names) > gt_labels_for_image[j]:
                    label_index = gt_labels_for_image[j]
                    label_name = class_names[label_index]
                    label_text = f"GT:{label_name}"
                    ax.text(xmin, ymin - 15, label_text, color='blue', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        # 畫 prediction boxes
        pred = preds[i]
        pred_labels = pred['labels'].cpu().tolist()
        pred_boxes = pred['boxes'].cpu().tolist()
        pred_scores = pred['scores'].cpu().tolist()

        for j, box in enumerate(pred_boxes):
            score = pred_scores[j]
            if score > 0.5: # 可以設定一個信心度閾值
                xmin, ymin, xmax, ymax = box
                rect_xmin = xmin
                rect_ymin = ymin
                rect_width = xmax - xmin
                rect_height = ymax - ymin
                rect = patches.Rectangle((rect_xmin, rect_ymin), rect_width, rect_height, linewidth=1, edgecolor='lime', facecolor='none') # 使用不同的顏色標示預測框
                ax.add_patch(rect)

                if class_names is not None and len(class_names) > pred_labels[j]:
                    label_index = pred_labels[j]
                    label_name = class_names[label_index]
                    label_text = f"{label_name}({score:.2f})"
                    ax.text(xmin, ymin - 5, label_text, color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Batch predictions saved to {save_path}")

def print_image_with_label_for_detection(train_dataset, save_dir="temp"):
    """
    This function displays a set of images from the object detection training dataset
    along with their corresponding bounding boxes and labels.
    It saves the resulting image grid to a specified directory.

    Args:
        train_dataset (CustomDataset): The training dataset containing images and their targets.
        save_dir (str, optional): Directory to save the displayed image grid. Defaults to "temp".
    """
    # Set the number of images to display
    N = 24

    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    axes = axes.flatten()

    # 反歸一化所需的 mean 和 std (與你的 train_transform 相同)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Display images and their corresponding labels
    for i in range(N):
        ax = axes[i]

        # Randomly select an image index
        idx = random.randint(0, len(train_dataset) - 1)
        img_tensor, target = train_dataset[idx]

        # 反歸一化圖像 tensor
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Get the image information to retrieve the file name and dimensions
        img_id = target['image_id'].item()
        img_info = train_dataset.data['images'][img_id]
        img_path = os.path.join(train_dataset.img_dir, img_info['file_name'])
        width = img_info['width']
        height = img_info['height']

        # Extract labels and bounding boxes
        labels = target['labels'].tolist()
        boxes = target['boxes'].tolist()

        # 使用類別名稱 (假設 train_dataset 有 class_names 屬性)
        class_names = train_dataset.class_names
        displayed_labels = [class_names[label] for label in labels]

        # Display the image, converting format from CxHxW to HxWxC
        ax.imshow(img_tensor.permute(1, 2, 0))
        ax.set_title(f"Labels: {displayed_labels}")
        ax.axis('off')

        # Draw bounding boxes
        for box in boxes:
            # bounding box 的格式是 [x_min, y_min, x_max, y_max]，現在已經是像素坐標
            xmin, ymin, xmax, ymax = box
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            # 直接使用像素坐標
            rect_xmin = xmin
            rect_ymin = ymin
            rect_width = bbox_width
            rect_height = bbox_height

            # 創建一個矩形 patch
            rect = patches.Rectangle((rect_xmin, rect_ymin), rect_width, rect_height, linewidth=1, edgecolor='r', facecolor='none')

            # 將矩形添加到 axes
            ax.add_patch(rect)

        # Output image path and labels (and boxes)
        print(f"Image Path: {img_path}, Labels: {displayed_labels}, Boxes: {boxes}")

    # Save the images to the 'temp' directory
    temp_image_path = os.path.join(save_dir, "images_with_bboxes_and_labels.png")
    plt.tight_layout()
    plt.savefig(temp_image_path)
    plt.close()

    print(f"Images with bounding boxes saved to {temp_image_path}")


def save_model_info(params_count, trained_model_str, save_name):
    """
    Save training model information to a specified file.

    :param params_count: Total number of model parameters
    :param trained_model_str: Model architecture description
    :param save_name: File name to save the information
    """

    # Check if the folder exists, create it if not
    os.makedirs(os.path.dirname(save_name), exist_ok=True)

    # Write the model information to a file
    with open(save_name, 'w') as f:
        f.write("Total Parameters: ")
        f.write(str(params_count))
        f.write("\n\nTrained Model:\n")
        f.write(trained_model_str)

    print(f"Model and parameter count saved to {save_name}\n")


def plot_training_history(history, save_name):
    """
    Plot the training history of loss and mAP, then save the results as an image.

    :param history: Training history, a dictionary containing loss, validation loss, and validation mAP.
    :param save_name: Path to save the plot image
    """

    # Set the figure size
    plt.figure(figsize=(10, 4))  # Adjusted figure size for better visualization

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
    plt.plot(history['val_map'], label='valid mAP', color='orange') # Using a different color for clarity
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

def set_seed(seed=63):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path="config.yaml"):
    """
    Load YAML configuration file and return a configuration dictionary.

    :param config_path: Path to the configuration file
    :return: Configuration dictionary
    """

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config
