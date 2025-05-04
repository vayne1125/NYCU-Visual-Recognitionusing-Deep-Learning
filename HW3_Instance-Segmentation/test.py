import torch
import torchvision.transforms as T
from PIL import Image
import os
import json
import numpy as np
from model import maskrcnn_v2

from utils import encode_mask


def run_inference(model, image_dir, output_json_path, image_id_map_path, device, score_threshold=0.5, mask_threshold=0.5):
    """
    Runs inference on images in a directory and saves results in COCO Detection Results JSON format.

    Args:
        model (torch.nn.Module): The trained Mask R-CNN model.
        image_dir (str): Directory containing the input images (.tif files).
        output_json_path (str): Path to save the output JSON file.
        image_id_map_path (str): Path to the JSON file mapping file_name to image_id.
        device (torch.device): The device to run inference on (cuda or cpu).
        score_threshold (float): Confidence score threshold for predictions.
        mask_threshold (float): Threshold to binarize predicted masks.
    """
    print(f"Running inference on images in: {image_dir}")
    print(f"Saving results to: {output_json_path}")
    print(f"Using image ID map from: {image_id_map_path}")

    # Load the image name to ID mapping
    try:
        with open(image_id_map_path, 'r', encoding='utf-8') as f:
            image_id_map_data = json.load(f)
        # Create a dictionary for quick lookup: {file_name: id}
        image_name_to_id = {item['file_name']: item['id'] for item in image_id_map_data}
        print(f"Loaded {len(image_name_to_id)} image ID mappings.")
    except FileNotFoundError:
        print(f"Error: Image ID map file not found at {image_id_map_path}")
        print("Please ensure the image ID map file exists.")
        return # Exit if the map file is not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {image_id_map_path}. Please check the file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading image ID map: {e}")
        return


    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Define image transformations (should match validation transformations)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of image files (assuming .tif format)
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    image_files.sort() # Sort to ensure consistent order

    # Initialize results as a single list for all predictions
    all_predictions = []
    annotation_id_counter = 1 # Counter for unique annotation IDs (optional for this format, but good practice)


    # Process each image
    for img_idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing image {img_idx + 1}/{len(image_files)}: {image_file}")

        # Get the image ID from the map
        img_id = image_name_to_id.get(image_file)
        if img_id is None:
            print(f"Warning: Image file '{image_file}' not found in the ID map. Skipping.")
            continue # Skip this image if no ID is found in the map

        try:
            # Read image
            img = Image.open(image_path).convert("RGB") # Ensure RGB format

            # Apply transformations
            img_tensor = transform(img)
            img_tensor = img_tensor.to(device)

            # Perform inference
            with torch.no_grad(): # No need to calculate gradients during inference
                # Model expects a list of tensors, even for a single image
                predictions = model([img_tensor])

            # Process predictions (predictions is a list with one dictionary for this image)
            preds = predictions[0]

            # Filter predictions by score threshold
            keep = preds['scores'] > score_threshold
            boxes = preds['boxes'][keep]
            labels = preds['labels'][keep]
            scores = preds['scores'][keep]
            masks = preds['masks'][keep] # Predicted masks (logits or probabilities)

            # Convert predictions to COCO Detection Results format
            if boxes.numel() > 0: # Check if any detections passed the threshold
                # Binarize masks and convert to RLE
                # Masks are typically [N, 1, H, W] or [N, H, W]
                # Apply threshold, squeeze channel dim if present, move to CPU, convert to numpy
                binary_masks = (masks > mask_threshold).squeeze(1).cpu().numpy() # Shape [num_predictions, H, W]

                for j in range(boxes.shape[0]): # Iterate through each predicted instance
                    box = boxes[j].tolist() # [x_min, y_min, x_max, y_max]
                    label = labels[j].item()
                    score = scores[j].item()
                    mask_np = binary_masks[j] # Binary mask numpy array

                    # Convert bbox from [x_min, y_min, x_max, y_max] to [x, y, w, h] for COCO
                    x_min, y_min, x_max, y_max = box
                    # Calculate width and height including +1
                    width_bbox = x_max - x_min + 1
                    height_bbox = y_max - y_min + 1
                    # Ensure width and height are non-negative
                    width_bbox = max(0.0, width_bbox)
                    height_bbox = max(0.0, height_bbox)

                    bbox_coco = [x_min, y_min, width_bbox, height_bbox]

                    # Convert mask numpy array to RLE format
                    # pycocotools.mask.encode expects Fortran contiguous numpy array
                    mask_np = np.asfortranarray(mask_np.astype(np.uint8)) # Ensure uint8 and Fortran contiguous
                    segmentation_rle = encode_mask(mask_np)

                    prediction_entry = {
                        "image_id": img_id, # Use the ID from the map
                        "category_id": label,
                        "bbox": bbox_coco,
                        "score": score,
                        "segmentation": segmentation_rle,
                    }
                    all_predictions.append(prediction_entry)
                    annotation_id_counter += 1 # Increment even if ID is not used in output


        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            # Continue to the next image even if one fails

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    # Save results to JSON file
    try:
        # The desired format is a list of prediction dictionaries.
        # The `all_predictions` list already holds this structure.
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # Sort results by image_id before saving, as per common COCO format practice
            sorted_predictions = sorted(all_predictions, key=lambda x: x['image_id'])
            json.dump(sorted_predictions, f, ensure_ascii=False, indent=4) # Use indent for readability
        print("Inference results saved successfully.")
    except Exception as e:
        print(f"Error saving inference results to {output_json_path}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    SAVE_NAME = "warnUp_smallanchor_v2"
    IMAGE_DIR = './data/test_release' # Directory containing your .tif test images
    MODEL_PATH = './local_storage/params/' + SAVE_NAME + '.pt' # Path to your trained model checkpoint
    OUTPUT_JSON_PATH = './local_storage/zip/test-results.json' # Output JSON file path
    IMAGE_ID_MAP_PATH = './data/test_image_name_to_ids.json' # Path to the image name to ID map file
    NUM_CLASSES = 5 # Number of classes your model was trained on (including background)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model = maskrcnn_v2(num_classes=NUM_CLASSES) # Initialize model architecture
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please ensure the trained model file exists.")
        exit()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # --- Run Inference ---
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    run_inference(model, IMAGE_DIR, OUTPUT_JSON_PATH, IMAGE_ID_MAP_PATH, device)

    print("Inference script finished.")