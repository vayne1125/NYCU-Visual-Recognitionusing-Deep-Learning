"""
This module provides utilities for visualizing image data, particularly for
deep learning tasks involving image restoration or processing with patches.
It includes functions for:

1.  Plotting Reconstructed Image Pairs: 
        Visualizing pairs of degraded and clean images, 
        where the degraded images are reconstructed from
        their constituent patches.
2.  Batch Patch Stitching: 
        A utility to reassemble a batch of image patches back into full-sized images.
3.  Plotting Restored Images: 
    Displaying and saving a collection of restored images from a NumPy `.npz` file.
4.  Gaussian Weight Mask Generation: 
        Creating a 3D Gaussian mask, often used for smooth blending 
        in patch-based image processing or Test-Time Augmentation (TTA).
5.  Natural Sorting: 
        A helper function for sorting filenames containing numbers in a human-friendly order.

These functions are designed to aid in the analysis and presentation of
results from image-to-image translation models, especially those operating
on image patches.
"""
import os
import re
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_and_save_image_pairs(dataset, num_images_to_plot=2,
                              save_path="reconstructed_image_pairs_plot.png"):
    """
    Plots the first num_images_to_plot pairs of reconstructed images (degraded vs clean)
    from the dataset and saves them as an image file.

    Args:
        dataset (torch.utils.data.Dataset): 
            The loaded dataset object (your CustomDataset instance).
            Assumes dataset[i] returns a tuple:
            (degraded_patches_tensor, clean_patches_tensor, original_clean_tensor).
            Where degraded_patches_tensor has shape [4, C, 128, 128],
            and original_clean_tensor has shape [C, 256, 256].
        num_images_to_plot (int): The number of full 256x256 image pairs to plot. Defaults to 2.
        save_path (str): The path to save the image file 
                         (defaults to "reconstructed_image_pairs_plot.png").
    """
    # Determine the actual number of images to plot, not exceeding the total dataset size
    actual_images_to_plot = min(num_images_to_plot, len(dataset))

    if actual_images_to_plot == 0:
        print("No image pairs available to plot in the dataset.")
        return

    print(
        f"Plotting the first {actual_images_to_plot} \
            reconstructed image pairs and saving to {save_path} ...")

    # Create a Figure and a subplot grid
    # Each row plots one pair of images (reconstructed degraded and original clean),
    # with actual_images_to_plot rows and 2 columns
    fig, axes = plt.subplots(actual_images_to_plot, 2,
                             figsize=(10, actual_images_to_plot * 5))

    for i in range(actual_images_to_plot):
        # Get the i-th image pair from the dataset
        # degraded_patches_tensor: [4, C, 128, 128]
        # original_clean_tensor: [C, 256, 256]
        # Ignoring clean_patches_tensor
        degraded_patches_tensor, _, original_clean_tensor = dataset[i]

        # Convert the 4 patches of a single image (shape [4, C, 128, 128])
        # to a batch format (shape [1, 4, C, 128, 128])
        # Then call your batch stitching function
        reconstructed_degraded_tensor_batch = batch_stitch_patches_2x2(
            degraded_patches_tensor.unsqueeze(0))
        # Extract the single image from the batch (shape [C, 256, 256])
        reconstructed_degraded_tensor = reconstructed_degraded_tensor_batch.squeeze(0)

        # For the clean image, directly use the original 256x256 clean image Tensor
        reconstructed_clean_tensor = original_clean_tensor

        # Convert the Tensors to NumPy arrays with [H, W, C] format, as required by Matplotlib
        # Since you are not normalizing, it's assumed pixel values are in [0.0, 1.0]
        reconstructed_degraded_np = reconstructed_degraded_tensor.permute(
            1, 2, 0).numpy()
        reconstructed_clean_np = reconstructed_clean_tensor.permute(
            1, 2, 0).numpy()

        # Handle indexing for single subplot when only one image pair is plotted
        if actual_images_to_plot == 1:
            ax_deg = axes[0]
            ax_clean = axes[1]
        else:
            ax_deg = axes[i, 0]
            ax_clean = axes[i, 1]

        ax_deg.imshow(reconstructed_degraded_np)
        ax_deg.set_title(f'Reconstructed Degraded {i+1}')
        ax_deg.axis('off')

        ax_clean.imshow(reconstructed_clean_np)
        ax_clean.set_title(f'Original Clean {i+1}')
        ax_clean.axis('off')

    plt.tight_layout()

    output_dir = os.path.dirname(save_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(save_path)
    plt.close(fig)

    print(f"Reconstructed image pairs plot successfully saved to {save_path}")


def batch_stitch_patches_2x2(patches_batch_tensor):
    """
    Stitches a batch of 4 non-overlapping 128x128 patches back into a batch of 256x256 images.
    Assumes patches_batch_tensor has shape [N, 4, C, 128, 128].
    Returns a Tensor of shape [N, C, 256, 256].
    Patch order is assumed to be: top-left, top-right, bottom-left, bottom-right.
    """
    N, num_patches, C, H, W = patches_batch_tensor.shape
    if num_patches != 4 or H != 128 or W != 128:
        raise ValueError(
            f"Expected patches_batch_tensor shape [N, 4, C, 128, 128], \
                but got {patches_batch_tensor.shape}")

    stitched_images_list = []
    for i in range(N):
        # Shape [4, C, 128, 128]
        single_image_patches = patches_batch_tensor[i]

        # Stitch patches in the order:
        #   top-left (0), top-right (1), bottom-left (2), bottom-right (3)
        # Concatenate along width to form the top row, shape [C, 128, 256]
        top_row = torch.cat(
            (single_image_patches[0], single_image_patches[1]), dim=2)
        # Concatenate along width to form the bottom row, shape [C, 128, 256]
        bottom_row = torch.cat(
            (single_image_patches[2], single_image_patches[3]), dim=2)
        # Concatenate along height to form the full 256x256 image, shape [C, 256, 256]
        stitched_image = torch.cat((top_row, bottom_row), dim=1)
        stitched_images_list.append(stitched_image)

    return torch.stack(stitched_images_list, dim=0)  # Shape [N, C, 256, 256]


def natural_sort_key(s):
    """ Key function for natural sorting of filenames. """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def plot_restored_images(npz_file_path, num_images_to_plot=12,
                            plot_save_path="restored_images_plot.png"):
    """
    Loads restored images from an .npz file, plots the first num_images_to_plot images,
    and saves the plot to a file.

    Args:
        npz_file_path (str): Path to the .npz file containing the restored images.
        num_images_to_plot (int): Number of images to plot from the .npz file. Defaults to 12.
        plot_save_path (str): Path to save the generated plot image.
    """
    if not os.path.exists(npz_file_path):
        print(f"Error: .npz result file not found at {npz_file_path}")
        return

    print(
        f"Loading results from {npz_file_path} and plotting the first \
            {num_images_to_plot} images to {plot_save_path}...")

    results_data = np.load(npz_file_path)

    filenames = sorted(results_data.files, key=natural_sort_key)
    images_to_plot = []
    titles = []
    actual_images_to_plot = min(num_images_to_plot, len(filenames))

    if actual_images_to_plot == 0:
        print("No images available to plot in the .npz file.")
        results_data.close()
        return

    for i in range(actual_images_to_plot):
        filename = filenames[i]
        image_np = results_data[filename]  # (C, H, W), uint8
        image_np_hwc = image_np.transpose(1, 2, 0)  # (H, W, 3), uint8
        images_to_plot.append(image_np_hwc)
        titles.append(filename)

    n_cols = 4
    n_rows = (actual_images_to_plot + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if (n_rows > 1 or n_cols > 1) else [axes]

    for i in range(actual_images_to_plot):
        ax = axes[i]
        ax.imshow(images_to_plot[i])
        ax.set_title(titles[i], fontsize=8)
        ax.axis('off')

    for j in range(actual_images_to_plot, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    plot_output_dir = os.path.dirname(plot_save_path)
    if plot_output_dir and not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir, exist_ok=True)

    plt.savefig(plot_save_path)
    print(f"Image results plot successfully saved to {plot_save_path}")

    plt.close(fig)
    results_data.close()


def generate_gaussian_weight_mask_3d(patch_size, sigma=None):
    """
    Generates a 3D Gaussian weight mask [1, patch_size, patch_size] for patch blending.
    Weights are 1 at the center and decay towards the edges.
    """
    if sigma is None:
        sigma = patch_size / 8.0  # Default sigma setting

    center = patch_size / 2.0
    y = torch.arange(patch_size, dtype=torch.float32)
    x = torch.arange(patch_size, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    distance_sq = (x_grid - center)**2 + (y_grid - center)**2
    weight_mask = torch.exp(-distance_sq / (2 * sigma**2))
    # Scale weights to be in the [0, 1] range, with max at the center
    weight_mask = weight_mask / torch.max(weight_mask)
    return weight_mask.unsqueeze(0)  # Shape [1, patch_size, patch_size]
