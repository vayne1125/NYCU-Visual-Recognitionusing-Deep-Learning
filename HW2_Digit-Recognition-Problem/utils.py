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
"""
import os
import random

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch


def visualize_test_predictions(images, predictions, image_filenames, class_names,
                               save_dir="temp_test_predictions"):
    """
    Visualizes predictions for the test set in a grid format, similar to validation visualization.

    Args:
        images (list of torch.Tensor): List of input images (tensors).
        predictions (list of list of dict):
                A list where each element corresponds to an image.
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
    rows = 2  # Fixed to display 2 rows
    # Calculate the number of columns based on the number of images
    cols = (num_images + 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))  # Create subplots
    axes = axes.flatten()

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
        3, 1, 1)  # ImageNet mean
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)  # ImageNet std

    for i in range(num_images):
        ax = axes[i]
        img_tensor = images[i].cpu()

        # Unnormalize the image
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)

        ax.imshow(img_tensor.permute(1, 2, 0))
        title = ""
        if image_filenames and i < len(image_filenames):
            title = f"{image_filenames[i]}\n"
        ax.set_title(title)
        ax.axis('off')

        # Draw prediction boxes
        image_preds = predictions[i]  # Get predictions for the current image

        for pred in image_preds:
            bbox = pred['bbox']
            category_id = pred['category_id']
            score = pred['score']

            xmin, ymin, width, height = bbox

            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1,
                                     edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

            if class_names is not None and len(class_names) > category_id:
                label_name = class_names[category_id]
                label_text = f"{label_name}({score:.2f})"
                ax.text(xmin, ymin - 5, s=label_text, color='red', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8))

    # Save all test images to one file
    save_path = os.path.join(save_dir, "test_predictions.png")
    plt.tight_layout()  # Automatically adjust subplot parameters for a tight layout
    plt.savefig(save_path)
    plt.close()
    print(f"Test predictions saved to {save_path}")


def visualize_predictions(images, preds, gt_boxes=None, gt_labels=None, class_names=None,
                          save_dir="temp_val_predictions", epoch=0, batch_idx=0,
                          image_filenames=None):
    """Visualizes a batch of images with predicted and ground truth bounding boxes and labels."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"epoch_{epoch}_batch_{batch_idx}_predictions.png")
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

        # Unnormalize
        img_tensor = img_tensor * std + mean

        # Clamp tensor values to be between 0 and 1
        img_tensor = torch.clamp(img_tensor, 0, 1)

        ax.imshow(img_tensor.permute(1, 2, 0))
        title = ""
        if image_filenames and i < len(image_filenames):
            title = f"{image_filenames[i]}\n"
        ax.set_title(title)
        ax.axis('off')

        # Draw ground truth boxes
        if gt_boxes is not None and gt_labels is not None and i < len(gt_boxes):
            gt_boxes_for_image = gt_boxes[i].cpu().tolist()
            gt_labels_for_image = gt_labels[i].cpu().tolist()
            for j, box in enumerate(gt_boxes_for_image):
                xmin, ymin, xmax, ymax = box
                rect_xmin = xmin
                rect_ymin = ymin
                rect_width = xmax - xmin
                rect_height = ymax - ymin
                # Mark GT boxes in blue
                rect = patches.Rectangle((rect_xmin, rect_ymin), rect_width, rect_height,
                                         linewidth=1, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)
                if class_names is not None and len(class_names) > gt_labels_for_image[j]:
                    label_index = gt_labels_for_image[j]
                    label_name = class_names[label_index]
                    label_text = f"GT:{label_name}"
                    ax.text(xmin, ymin - 15, label_text, color='blue',
                            fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

        # Draw prediction boxes
        pred = preds[i]
        pred_labels = pred['labels'].cpu().tolist()
        pred_boxes = pred['boxes'].cpu().tolist()
        pred_scores = pred['scores'].cpu().tolist()

        for j, box in enumerate(pred_boxes):
            score = pred_scores[j]
            if score > 0.5:  # You can set a confidence threshold
                xmin, ymin, xmax, ymax = box
                rect_xmin = xmin
                rect_ymin = ymin
                rect_width = xmax - xmin
                rect_height = ymax - ymin
                # Mark prediction boxes in lime
                rect = patches.Rectangle((rect_xmin, rect_ymin), rect_width, rect_height,
                                         linewidth=1, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)

                if class_names is not None and len(class_names) > pred_labels[j]:
                    label_index = pred_labels[j]
                    label_name = class_names[label_index]
                    label_text = f"{label_name}({score:.2f})"
                    ax.text(xmin, ymin - 5, label_text, color='red',
                            fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

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

    # Mean and std for unnormalization (same as your train_transform)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # Display images and their corresponding labels
    for i in range(N):
        ax = axes[i]

        # Randomly select an image index
        idx = random.randint(0, len(train_dataset) - 1)
        img_tensor, target = train_dataset[idx]

        # Unnormalize the image tensor
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1)

        # Get the image information to retrieve the file name and dimensions
        img_id = target['image_id'].item()
        img_info = train_dataset.data['images'][img_id]
        img_path = os.path.join(train_dataset.img_dir, img_info['file_name'])

        # Extract labels and bounding boxes
        labels = target['labels'].tolist()
        boxes = target['boxes'].tolist()

        # Use class names (assuming train_dataset has a class_names attribute)
        class_names = train_dataset.class_names
        displayed_labels = [class_names[label] for label in labels]

        # Display the image, converting format from CxHxW to HxWxC
        ax.imshow(img_tensor.permute(1, 2, 0))
        ax.set_title(f"Labels: {displayed_labels}")
        ax.axis('off')

        # Draw bounding boxes
        for box in boxes:
            # bounding box format is [x_min, y_min, x_max, y_max], already in pixel coordinates
            xmin, ymin, xmax, ymax = box
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            # Use pixel coordinates directly
            rect_xmin = xmin
            rect_ymin = ymin
            rect_width = bbox_width
            rect_height = bbox_height

            # Create a rectangle patch
            rect = patches.Rectangle((rect_xmin, rect_ymin), rect_width,
                                     rect_height, linewidth=1, edgecolor='r', facecolor='none')

            # Add the rectangle to the axes
            ax.add_patch(rect)

        # Output image path and labels (and boxes)
        print(
            f"Image Path: {img_path}, Labels: {displayed_labels}, Boxes: {boxes}")

    # Save the images to the 'temp' directory
    temp_image_path = os.path.join(
        save_dir, "images_with_bboxes_and_labels.png")
    plt.tight_layout()
    plt.savefig(temp_image_path)
    plt.close()

    print(f"Images with bounding boxes saved to {temp_image_path}")


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
    plt.plot(history['val_map'], label='valid mAP', color='orange')
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
