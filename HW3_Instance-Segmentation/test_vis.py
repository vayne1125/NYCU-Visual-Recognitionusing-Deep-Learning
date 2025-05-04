"""
This script visualizes instance segmentation predictions by overlaying predicted masks
onto original images. It takes in predictions in COCO format and generates visualized
images where masks are blended with the original image.

Arguments:
    - image_dir: Directory containing the input images (.tif files).
    - image_id_map_path: Path to a JSON file mapping image filenames to image IDs.
    - predictions_json_path: Path to the JSON file containing prediction results.
    - output_dir: Directory to save the visualized images.
    - score_threshold: Minimum score threshold for displaying a mask.
    - original_image_blend_ratio: Ratio for blending original image with white background.
    - mask_alpha: Transparency level for the mask overlay.

This script requires the 'pycocotools' library for decoding segmentation masks and 
uses PIL for image processing.
"""
import os
import json
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt

def visualize_masks_only(image_dir, image_id_map_path, predictions_json_path,
                         output_dir, score_threshold=0.0, original_image_blend_ratio=0.4,
                         mask_alpha=150):
    """
    Visualizes instance segmentation predicted masks on images by blending the original image
    with a background color and then overlaying colored masks.

    Args:
        image_dir (str): Directory containing the input images (.tif files).
        image_id_map_path (str): Path to the JSON file mapping file_name to image_id.
        predictions_json_path (str): Path to the JSON file containing the predictions.
        output_dir (str): Directory to save the visualized images.
        score_threshold (float): 
            Minimum score for a prediction's mask to be displayed.
            Defaults to 0.0 to show all predictions regardless of score.
        original_image_blend_ratio (float): 
            Ratio to blend the original image with a white background.
            Value between 0.0 (fully white background) and 1.0 
            (original image fully visible).
            Lower values make the original image fainter. Defaults to 0.4.
        mask_alpha (int): Alpha value (0-255) for the mask overlay. Defaults to 150.
    """
    print(f"Visualizing predictions from: {predictions_json_path}")
    print(f"Onto images in: {image_dir}")
    print(f"Using image ID map from: {image_id_map_path}")
    print(f"Saving visualized images to: {output_dir}")
    print(f"Using score threshold for visualization: {score_threshold}")
    print(f"Using original image blend ratio: {original_image_blend_ratio}")
    print(f"Using mask alpha: {mask_alpha}")

    # Ensure original_image_blend_ratio is within a valid range
    original_image_blend_ratio = max(0.0, min(1.0, original_image_blend_ratio))
    # Ensure mask_alpha is within a valid range
    mask_alpha = max(0, min(255, mask_alpha))

    # Define colors for each category (RGB format)
    # We will apply alpha later when blending the mask overlay
    # 你指定的顏色：class1 粉色, class2 藍色, class3 黃色, class4 綠色
    # COCO category_id 通常從 1 開始，所以我們定義一個字典來對應
    category_colors_rgb = {
        4: (255, 192, 203),  # 粉色 (Pink)
        2: (0, 0, 255),     # 藍色 (Blue)
        3: (255, 255, 0),   # 黃色 (Yellow)
        1: (0, 128, 0),     # 綠色 (Green)
        # Add more categories if needed, or a default color
    }
    # Default color for categories not in the map
    default_color_rgb = (128, 128, 128)  # 灰色

    # Load the image name to ID mapping
    with open(image_id_map_path, 'r', encoding='utf-8') as f:
        image_id_map_data = json.load(f)
    # Create a dictionary for quick lookup: {image_id: file_name}
    id_to_image_name = {item['id']: item['file_name']
                        for item in image_id_map_data}
    print(f"Loaded {len(id_to_image_name)} image ID mappings.")

    # Load the prediction results
    with open(predictions_json_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from results file.")

    # Group predictions by image_id for easier access
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        # Ensure prediction has required keys before grouping
        if all(key in pred for key in ['image_id', 'category_id', 'score', 'segmentation']):
            predictions_by_image[pred['image_id']].append(pred)
        else:
            print(f"Warning: Skipping prediction with missing keys: {pred}")

    print(
        f"Grouped predictions for {len(predictions_by_image)} unique images.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image IDs that have predictions
    image_ids_to_process = sorted(list(predictions_by_image.keys()))

    # Process each image ID that has predictions
    for img_idx, img_id in enumerate(image_ids_to_process):
        image_file = id_to_image_name.get(img_id)
        if image_file is None:
            print(
                f"Warning: Image ID {img_id} found in predictions but not in ID map.\
                    Cannot visualize without image file name. Skipping.")
            continue

        image_path = os.path.join(image_dir, image_file)
        # Change output file extension to .png
        output_file_name = os.path.splitext(image_file)[0] + '.png'
        output_path = os.path.join(
            output_dir, f"visualized_{output_file_name}")

        # Skip if image file doesn't exist
        if not os.path.exists(image_path):
            print(
                f"Warning: Image file not found at {image_path}.\
                    Skipping visualization for ID {img_id} ({image_file}).")
            continue

        print(
            f"Visualizing image {img_idx + 1}/{len(image_ids_to_process)}\
                (ID: {img_id}, File: {image_file})...")

        try:
            # Read image and convert to RGB for blending
            img_original_rgb = Image.open(image_path).convert("RGB")
            img_size = img_original_rgb.size

            # Get predictions for this image ID
            image_predictions = predictions_by_image.get(img_id, [])

            # --- Blend the original image with a white background to make it fainter ---
            white_background = Image.new('RGB', img_size, (255, 255, 255))

            # Blend the original RGB image with the white background
            # Use Image.blend: result = (1-alpha)*image1 + alpha*image2
            # We want (1-original_image_blend_ratio)*white + original_image_blend_ratio*original_rgb
            blended_img_rgb = Image.blend(
                white_background, img_original_rgb, original_image_blend_ratio)

            # --- Create a composite mask image (RGB) ---
            # Initialize with black or any background color that won't interfere
            composite_mask_rgb = Image.new('RGB', img_size, (0, 0, 0))
            composite_mask_draw = ImageDraw.Draw(composite_mask_rgb)

            # Draw each prediction mask onto the composite_mask_rgb
            for pred in image_predictions:
                # Filter by score threshold
                if pred['score'] < score_threshold:
                    continue  # Skip predictions below the threshold

                category_id = pred['category_id']
                segmentation = pred['segmentation']

                # Get color for the category (RGB)
                color_rgb = category_colors_rgb.get(
                    category_id, default_color_rgb)

                # Decode RLE segmentation mask
                if isinstance(segmentation['counts'], str):
                    segmentation['counts'] = segmentation['counts'].encode(
                        'utf-8')

                # mask is a numpy array (H, W) with values 0 or 1
                mask = maskUtils.decode(segmentation)
                # Ensure mask is boolean or uint8
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)

                # Ensure mask size matches image size (PIL size is width, height)
                if mask is not None and mask.shape[:2] == img_size[::-1]:
                    # Create a binary mask image from the numpy array
                    # Convert 0/1 to 0/255 for PIL mask
                    binary_mask_img = Image.fromarray(mask * 255, 'L')
                    composite_mask_draw.bitmap(
                        (0, 0), binary_mask_img, fill=color_rgb)

                else:
                    if mask is not None:
                        print(
                            f"Warning: Mask shape {mask.shape[:2]} does not match\
                                image size {img_size[::-1]} for image ID {img_id}.\
                                Skipping mask drawing.")

            # --- Blend the composite mask image onto the faded original image ---
            # Convert the composite mask RGB image to RGBA with the desired mask_alpha
            composite_mask_rgba = composite_mask_rgb.copy()  # Create a copy to add alpha
            # Add the fixed alpha channel
            composite_mask_rgba.putalpha(Image.new('L', img_size, mask_alpha))

            # Composite the semi-transparent mask overlay onto the blended original image
            # Ensure blended_img_rgb is converted to RGBA before compositing
            blended_img_rgba = blended_img_rgb.convert("RGBA")
            img_final = Image.alpha_composite(
                blended_img_rgba, composite_mask_rgba)

            # Save the visualized image as PNG
            # Ensure output directory exists (already done before loop, but good practice)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img_final.save(output_path, format='PNG')  # Explicitly save as PNG
            print(f"Saved visualized image to: {output_path}")

        except Exception as e:
            print(f"Error visualizing image ID {img_id} ({image_file}): {e}")


def plot_and_save_first_8_images(folder, output_path='output_grid.png'):
    """
    Plot the first 8 image files in a given folder and save them as a grid image.

    Args:
        folder (str): Path to the folder containing image files.
        output_path (str): Filename for the output grid image (default: 'output_grid.png').

    Returns:
        None. The function saves the output image to the specified folder.
    """
    # 取得所有圖片檔案名稱
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    image_files = image_files[:8]

    if not image_files:
        print("No image files found in the folder.")
        return

    # 建立 plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, filename in zip(axes, image_files):
        image_path = os.path.join(folder, filename)
        image = Image.open(image_path)
        ax.imshow(image)
        ax.set_title(filename, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(folder, output_path), dpi=300)
    plt.close()
    print(f"Saved grid image to {os.path.join(folder, output_path)}")

if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_DIR = './data/test_release'  # Directory containing your .tif test images
    # Path to the image name to ID map file
    IMAGE_ID_MAP_PATH = './data/test_image_name_to_ids.json'
    # Path to the prediction results JSON file
    PREDICTIONS_JSON_PATH = './local_storage/zip/test-results.json'
    # Directory to save visualized images
    OUTPUT_DIR = './local_storage/test_visualized_predictions'

    # Minimum score to display a prediction (set to 0.0 to show all)
    SCORE_THRESHOLD = 0.0
    # Adjust this value (0.0-1.0) to make the original image fainter (lower value = fainter)
    ORIGINAL_IMAGE_BLEND_RATIO = 0.8
    # Adjust this value (0-255) to control the transparency of the masks
    MASK_ALPHA = 100

    # --- Run Visualization ---
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    visualize_masks_only(IMAGE_DIR, IMAGE_ID_MAP_PATH, PREDICTIONS_JSON_PATH,
                         OUTPUT_DIR, SCORE_THRESHOLD, ORIGINAL_IMAGE_BLEND_RATIO, MASK_ALPHA)

    print("Mask-only visualization script finished.")

    plot_and_save_first_8_images(OUTPUT_DIR)
