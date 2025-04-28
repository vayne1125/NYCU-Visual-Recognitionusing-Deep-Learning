import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils

import os
import random

import matplotlib.pyplot as plt
from matplotlib import patches
import torch

from sklearn.model_selection import train_test_split

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
    0: (255, 255, 255), # Background - White
    1: (255, 192, 203), # Class 1 - Pink
    2: (135, 206, 235), # Class 2 - Sky Blue
    3: (255, 255, 224), # Class 3 - Light Yellow
    4: (144, 238, 144)  # Class 4 - Light Green
    # Add more classes if you have them
}

def print_image_with_mask_for_segmentation(dataset, num_images=3, save_dir="segmentation_viz"):
    """
    This function displays a set of images from the instance segmentation dataset
    along with their corresponding segmentation masks, colored by class.
    It saves the resulting image grid to a specified directory.

    Args:
        dataset (CustomDataset): The dataset containing images and their targets.
        num_images (int, optional): The number of images to display. Defaults to 3.
        save_dir (str, optional): Directory to save the displayed image grid. Defaults to "segmentation_viz".
    """
    # Ensure the number of images to display does not exceed the dataset size
    # Fixed number of images to display
    N = 3

    if N == 0 or len(dataset) == 0:
        print("Dataset is empty or N is 0. Cannot display images.")
        return

    # Determine the grid size (e.g., 1x3 for 3 images)
    rows = 1
    cols = N

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4)) # Adjust figsize as needed
    # If N=1, axes is not a numpy array, handle this case
    if N == 1:
        axes = [axes]
    else:
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration


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
        try:
            img_tensor, target = dataset[idx]
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}. Skipping.")
            ax.set_title("Error Loading Image")
            ax.axis('off')
            continue # Skip to the next image if loading fails

        # Unnormalize the image tensor
        # Ensure mean and std are on the same device as img_tensor if using GPU
        if img_tensor.device != mean.device:
             mean = mean.to(img_tensor.device)
             std = std.to(img_tensor.device)

        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0, 1) # Clamp values to be in [0, 1]

        # Convert image tensor (CxHxW, float) to numpy array (HxWxC, uint8) for visualization
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Get masks and labels from the target
        masks = target['masks'] # Shape (num_instances, H, W), dtype uint8 (0 or 1)
        labels = target['labels'] # Shape (num_instances,), dtype int64

        # Get original image dimensions from target (useful if image was resized)
        # If not resized in dataset __getitem__, this will be original size
        if 'orig_size' in target:
             img_h, img_w = target['orig_size'].tolist()
        else:
             # Fallback if orig_size is not in target (less robust)
             img_h, img_w = img_np.shape[:2]


        # Create a composite mask image (HxWx3, uint8)
        # Initialize with background color (White)
        composite_mask_np = np.full((img_h, img_w, 3), CLASS_COLORS[0], dtype=np.uint8)

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
            y_coords, x_coords = np.where(mask_np > 0) # Use > 0 in case mask is not strictly 0/1

            # Apply the color to these pixels in the composite mask
            if len(y_coords) > 0: # Ensure there are foreground pixels
                composite_mask_np[y_coords, x_coords] = color

        # Blend the original image and the composite mask
        # alpha controls the transparency of the mask (0.0 is fully transparent, 1.0 is fully opaque)
        alpha = 0.8 # Adjust transparency as needed
        # Ensure both arrays are float for blending
        blended_img_np = (img_np.astype(np.float32) * (1 - alpha) + composite_mask_np.astype(np.float32) * alpha).astype(np.uint8)


        # Display the blended image
        ax.imshow(blended_img_np)

        # --- Set title to file_name and img_id ---
        # Get the image_id from the target
        img_id = target['image_id'].item()
        # Use the image_id to get image info from the dataset's mapping
        # Assuming dataset has image_id_to_info attribute populated from JSON
        if hasattr(dataset, 'image_id_to_info') and img_id in dataset.image_id_to_info:
            image_info = dataset.image_id_to_info[img_id]
            file_name = image_info.get('file_name', f"ID: {img_id}") # Get file_name, fallback to ID if not found
            # Combine file_name and img_id in the title
            ax.set_title(f"ID: {img_id}\nFile: {file_name}", fontsize=8) # Set title to file name and ID
        else:
             # Fallback if image_id_to_info is not available or ID not found
             ax.set_title(f"Image ID: {img_id}", fontsize=8)


        ax.axis('off') # Turn off axes

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    temp_image_path = os.path.join(
        save_dir, f"images_with_segmentation_masks_{N}.png")
    plt.tight_layout() # Adjust layout to prevent titles/labels overlapping
    plt.savefig(temp_image_path)
    plt.close(fig) # Close the figure to free up memory

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
        shuffle=True # 確保數據被打亂
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
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array