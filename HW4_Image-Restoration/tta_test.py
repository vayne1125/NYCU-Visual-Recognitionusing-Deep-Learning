"""
This module provides an inference script for image restoration models, focusing on
**patch-based processing** and **Test-Time Augmentation (TTA)** to enhance results
on full-sized images.

It features a core `run_inference` function that:
- Loads a trained PyTorch Lightning model.
- Processes test images by extracting overlapping patches.
- Reconstructs full images using **Gaussian blending** for seamless results.
- Applies **Test-Time Augmentation (TTA)** (flips, rotations) to input images,
  averaging inverse-transformed outputs for improved robustness and quality.

The main execution block demonstrates how to set up the inference pipeline,
including loading checkpoints, configuring patching parameters, and saving
both numerical results and visual plots. This script is ideal for evaluating
image restoration models on high-resolution test data.
"""
import os
import torch
import torchvision.transforms as T
import numpy as np
from torchvision.transforms import functional as F
from datasets import TestImageDataset

from utils import generate_gaussian_weight_mask_3d, plot_restored_images
from train_part2 import PromptIRModel


# --- Inference execution function (Patch-based processing with TTA) ---
def run_inference(lightning_model, test_dataloader, output_file_path, device, patch_size, stride):
    """
    Performs patch-based inference on raw images from a DataLoader, incorporating
    Test-Time Augmentation (TTA), and saves the results to a .npz file.
    It uses overlapping patches and Gaussian blending for seamless reconstruction.
    Assumes the model input and output dimensions are both `patch_size`, and
    the model output is in the [0, 1] range.

    Args:
        lightning_model (PromptIRModel): Loaded LightningModule instance.
        test_dataloader (DataLoader):
            DataLoader returning original-sized image Tensors and filenames.
        output_file_path (str): Path to save the .npz file.
        device (torch.device): Device for inference.
        patch_size (int): Input size for the model during training (e.g., 128).
        stride (int): Stride for extracting patches (e.g., 64), determines overlap size.
    """
    print(
        f"Performing patch inference on images from DataLoader and saving to: {output_file_path}")
    print(f"Patch size: {patch_size}x{patch_size}, Stride: {stride}")

    lightning_model.to(device)
    lightning_model.eval()
    # Get the underlying model instance and ensure it's also in eval mode
    inference_model = lightning_model.net
    inference_model.eval()

    results_dict = {}

    # Generate Gaussian weight mask for patch blending and move it to the device
    weight_mask = generate_gaussian_weight_mask_3d(
        patch_size, sigma=patch_size / 1.0).to(device)

    print(
        f"Will process {len(test_dataloader)} batches, \
            totaling {len(test_dataloader.dataset)} images.")

    with torch.no_grad():
        for batch_idx, (original_image_batch_256, filename_batch) in enumerate(test_dataloader):
            # original_image_batch_256: [batch_size, C, 256, 256] Tensor (in [0, 1] range)
            # filename_batch: list of batch_size filenames

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"Processing batch {batch_idx + 1}/{len(test_dataloader)}...")

            # Move the original image batch to the device
            original_image_batch_256 = original_image_batch_256.to(device)

            batch_size = original_image_batch_256.shape[0]

            # --- Process each original image in the batch ---
            for i in range(batch_size):
                # [C, 256, 256]
                original_image_256 = original_image_batch_256[i]
                filename = filename_batch[i]

                # --- Define TTA augmentations (using torchvision.transforms.functional) ---
                # List of (augmentation_fn, inverse_augmentation_fn) pairs
                # Use lambda for simple functions, or define separate functions.
                # Add Identity, Horizontal Flip, Vertical Flip, and 180-degree rotation as base TTA.
                tta_augmentations = [
                    # 0: Identity (no change)
                    (lambda img: img, lambda img: img),
                    # 1: Horizontal flip
                    (F.hflip, F.hflip),
                    # 2: Vertical flip
                    (F.vflip, F.vflip),
                    # 4: Rotate 180 degrees (inverse is -180 degrees, or another 180 degrees)
                    (lambda img: F.rotate(img, 180),
                     lambda img: F.rotate(img, -180)),
                ]

                # To store restored images after processing each TTA variant
                restored_versions = []

                # --- Loop through TTA augmentations ---
                for aug_fn, inverse_aug_fn in tta_augmentations:
                    # 1. Apply augmentation to the original image
                    # Apply augmentation to the 256x256 image
                    augmented_image_256 = aug_fn(original_image_256)

                    # 2. Perform patch extraction and model inference on the augmented image
                    patches_list = []
                    coords_list = []  # Top-left coordinates of the patch in the augmented image
                    # Use dimensions of the augmented image
                    img_h, img_w = augmented_image_256.shape[1:]

                    # Calculate patch extraction coordinates (ensure coverage of edges)
                    y_coords = sorted(
                        list(set(range(0, img_h - patch_size + stride, stride))))
                    if (img_h - patch_size) % stride != 0:
                        y_coords.append(img_h - patch_size)
                    y_coords = sorted(list(set(y_coords)))

                    x_coords = sorted(
                        list(set(range(0, img_w - patch_size + stride, stride))))
                    if (img_w - patch_size) % stride != 0:
                        x_coords.append(img_w - patch_size)
                    x_coords = sorted(list(set(x_coords)))

                    for y in y_coords:
                        for x in x_coords:
                            # Extract patch from the augmented image
                            patch = augmented_image_256[:,
                                                        y:y+patch_size, x:x+patch_size]
                            patches_list.append(patch)
                            coords_list.append((y, x))

                    if not patches_list:
                        print(
                            f"Warning: Could not extract patches for image {filename} \
                                (after augmentation). Skipping this augmentation.")
                        continue

                    # Stack the list of patches into a batch
                    # [num_patches, C, patch_size, patch_size]
                    patch_batch = torch.stack(patches_list, dim=0)

                    # Perform inference on the patch batch
                    # [num_patches, C, patch_size, patch_size]
                    restored_patch_batch = inference_model(patch_batch)

                    # 3. Reconstruct the restored augmented image and blend
                    restored_patch_batch_cpu = restored_patch_batch.cpu()
                    weight_mask_cpu = weight_mask.cpu()  # Gaussian weight mask

                    # Initialize canvas with the same size as the augmented image
                    output_canvas = torch.zeros_like(augmented_image_256).cpu()
                    weight_canvas = torch.zeros(
                        (1, img_h, img_w), dtype=torch.float32).cpu()

                    # Iterate through each restored patch and its coordinates
                    #   in the augmented image for overlay blending
                    for j in range(restored_patch_batch_cpu.shape[0]):
                        # [C, patch_size, patch_size]
                        restored_patch = restored_patch_batch_cpu[j]
                        # Coordinates of the patch in the augmented image
                        y, x = coords_list[j]

                        # Broadcast and apply the weight mask to the restored patch
                        weighted_patch = restored_patch * weight_mask_cpu

                        # Overlay onto the canvas
                        output_canvas[:, y:y+patch_size,
                                      x:x+patch_size] += weighted_patch
                        weight_canvas[:, y:y+patch_size,
                                      x:x+patch_size] += weight_mask_cpu

                    # Perform normalized blending (avoid division by zero)
                    blended_augmented_result_256 = output_canvas / \
                        (weight_canvas.expand_as(output_canvas) + 1e-8)

                    # 4. Apply **inverse augmentation** to the restored augmented image
                    # Move the result back to the device to apply inverse function
                    restored_original_orientation_256 = inverse_aug_fn(
                        blended_augmented_result_256.to(device))

                    # 5. Store the restored image version after inverse augmentation
                    restored_versions.append(restored_original_orientation_256)

                # --- Average the results from all TTA variants ---
                if restored_versions:
                    # Stack all versions and average along the batch dimension
                    restored_versions_stacked = torch.stack(
                        restored_versions, dim=0).to(device)
                    final_restored_image_256 = torch.mean(
                        restored_versions_stacked, dim=0)  # [C, 256, 256]
                else:
                    print(
                        f"Warning: TTA could not be performed for image {filename}. Skipping.")
                    continue  # If no successful TTA variants, skip this image

                # --- Final post-processing: Clamp, scale to [0, 255], convert to uint8 NumPy ---
                # Post-process the final averaged TTA result
                image_0_1_256 = torch.clamp(final_restored_image_256, 0.0, 1.0)
                image_0_255_256 = image_0_1_256 * 255.0
                # Move to CPU before converting to NumPy
                restored_np_256 = image_0_255_256.cpu().numpy().astype(
                    np.uint8)  # [C, 256, 256], uint8

                # Store the 256x256 TTA result with the original filename as key
                results_dict[filename] = restored_np_256

    print("\nAll images processed.")

    # --- Save results ---
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    np.savez(output_file_path, **results_dict)


if __name__ == "__main__":
    # --- Configuration ---
    OUTPUT_DIR = 'local/results'
    TEST_DATA_ROOT = 'hw4_realse_dataset/test/'  # Your test set root directory

    SAVE_NAME = "./finetune3/best_psnr-epoch=059-val_psnr=28.60"  # Your checkpoint name

    # Full path to the checkpoint
    MODEL_PATH = os.path.join('local/params', SAVE_NAME + '.ckpt')
    OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, 'pred.npz')  # .npz output path
    PLOT_SAVE_PATH = os.path.join(
        OUTPUT_DIR, 'restored_images_grid.png')  # Plot output path

    # --- Patch Configuration ---
    PATCH_SIZE = 128
    STRIDE = 32

    # --- DataLoader Configuration ---
    INFERENCE_BATCH_SIZE = 1
    INFERENCE_NUM_WORKERS = 4

    # --- Test Set Transformation ---
    inference_transform = T.Compose([
        T.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PromptIRModel.load_from_checkpoint(
        checkpoint_path=MODEL_PATH,
        map_location=device,
    )
    print(f"Model loaded successfully from {MODEL_PATH}")

    model.eval()
    if hasattr(model, 'net') and isinstance(model.net, torch.nn.Module):
        model.net.eval()

    # --- Create Test Set Dataset and DataLoader ---
    test_dataset = TestImageDataset(
        TEST_DATA_ROOT, transform=inference_transform)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=INFERENCE_BATCH_SIZE,
        shuffle=False,
        num_workers=INFERENCE_NUM_WORKERS,
        persistent_workers=True
    )

    # --- Run Inference (Patch Processing + TTA) ---
    run_inference(model, test_dataloader, OUTPUT_FILE_PATH,
                  device, PATCH_SIZE, STRIDE)

    # --- Plot and Save Result Images ---
    plot_restored_images(
        OUTPUT_FILE_PATH, num_images_to_plot=12, plot_save_path=PLOT_SAVE_PATH)

    print("Inference script finished.")
