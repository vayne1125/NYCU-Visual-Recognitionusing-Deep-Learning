"""
Test script for running inference on images using a trained model.
"""
import csv
import os
import json
from PIL import Image

import torch
from torchvision import transforms
from torch.backends import cudnn

from model import ftrcnn
from utils import set_seed, visualize_test_predictions


def test_task1(model, model_path, test_dir, transform, output_json_path,
                 visualize=False, visualize_path="temp_test_predictions"):
    """
    Performs object detection on test images and saves predictions to a JSON file.

    Args:
        model (torch.nn.Module): The trained object detection model.
        model_path (str): Path to the trained model's state dictionary.
        test_dir (str): Directory containing the test images.
        transform (torchvision.transforms.Compose): Image transformations to apply.
        output_json_path (str): Path to save the JSON predictions.
        visualize (bool, optional): 
            Whether to visualize predictions on a few images. Defaults to False.
        visualize_path (str, optional): 
            Path to save the visualized images. Defaults to "temp_test_predictions".
    """
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    image_files = sorted(os.listdir(test_dir))

    predictions = []
    # Initialize lists to store images, predictions, and filenames for visualization
    visualization_images = []
    visualization_preds = []
    visualization_filenames = []

    with torch.no_grad():
        # Iterate through each image file
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(test_dir, image_file)
            try:
                image = Image.open(image_path).convert("RGB")
                img_tensor = transform(image).unsqueeze(0).to(device)

                output = model(img_tensor)[0]

                # Extract the image ID from the filename
                image_id = int(os.path.splitext(image_file)[0])

                current_image_predictions = []
                # Iterate through the detected bounding boxes
                for j in range(len(output['boxes'])):
                    score = output['scores'][j].item()
                    if score > 0.5:
                        bbox = output['boxes'][j].cpu().tolist()
                        label_index = output['labels'][j].item()
                        category_id = label_index
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        bbox_wh = [bbox[0], bbox[1], width, height]

                        # Append the prediction details to the list of all predictions
                        predictions.append({
                            "image_id": image_id,
                            "bbox": bbox_wh,
                            "score": score,
                            "category_id": category_id
                        })
                        current_image_predictions.append({
                            'bbox': bbox_wh,
                            'category_id': category_id,
                            'score': score
                        })

                # Store data for visualization for the first 8 images
                if visualize and i < 8:
                    visualization_images.append(img_tensor.squeeze(0).cpu())
                    visualization_preds.append(current_image_predictions)
                    visualization_filenames.append(image_file)

                # Visualize predictions for the first 8 images
                if visualize and i == 7:
                    class_names = ["bg", "0", "1", "2",
                                   "3", "4", "5", "6", "7", "8", "9"]
                    visualize_test_predictions(images=visualization_images,
                                                            predictions=visualization_preds,
                                                            class_names=class_names,
                                                            save_dir=visualize_path,
                                                            image_filenames=visualization_filenames)
                    # Clear the lists after visualization
                    visualization_images.clear()
                    visualization_preds.clear()
                    visualization_filenames.clear()
            except FileNotFoundError:
                print(f"Error: Image file not found at {image_path}")
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")

    # Save the list of predictions to a JSON file
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Task 1 predictions saved to {output_json_path}")


# Define the image transformations for testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def test_task2(input_json_path, test_image_dir, output_csv_path):
    """
    Processes the JSON predictions to generate a CSV file with predicted numbers.

    Args:
        input_json_path (str): Path to the JSON file containing object detection predictions.
        test_image_dir (str): Directory containing the test images.
        output_csv_path (str): Path to save the CSV file with predicted numbers.
    """
    # Open and load the JSON file containing the predictions
    with open(input_json_path, 'r') as f:
        predictions = json.load(f)

    # Create a dictionary to group predictions by image ID
    image_predictions = {}
    for pred in predictions:
        image_id = pred['image_id']
        if image_id not in image_predictions:
            image_predictions[image_id] = []
        image_predictions[image_id].append(pred)

    # Get all image IDs from the test image directory
    image_files = sorted(os.listdir(test_image_dir))
    all_image_ids = set()
    for file_name in image_files:
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Adjust based on your image formats
            try:
                image_id = int(os.path.splitext(file_name)[0])
                all_image_ids.add(image_id)
            except ValueError:
                print(
                    f"Warning: Could not extract image_id from filename: {file_name}")

    results_task2 = []

    # Iterate through each image ID in sorted order
    for image_id in sorted(list(all_image_ids)):
        detections = image_predictions.get(image_id, [])
        # If no detections are found for the image, predict -1
        if not detections:
            results_task2.append([image_id, -1])
        else:
            # Sort the detections by the x-coordinate of the bounding box
            detections.sort(key=lambda x: x['bbox'][0])
            predicted_number_str = ""
            for detection in detections:
                # Assuming category_id starts from 1 (0 is background), subtract 1 to get the digit
                predicted_number_str += str(detection['category_id']-1)

            # If a predicted number string was formed
            if predicted_number_str:
                try:
                    # Convert the predicted number string to an integer
                    predicted_number = int(predicted_number_str)
                    results_task2.append([image_id, predicted_number])
                except ValueError:
                    # If the conversion fails, predict -1
                    results_task2.append([image_id, -1])
            else:
                # If no digits were detected, predict -1
                results_task2.append([image_id, -1])

    # Save the Task 2 results to a CSV file
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(results_task2)

    print(f"Task 2 predictions saved to {output_csv_path}")


if __name__ == "__main__":

    DATA_DIR = 'nycu-hw2-data'
    cudnn.benchmark = True
    set_seed(63)
    TEST_DIR = os.path.join(DATA_DIR, "test/")
    SAVE_NAME = "resnet50_fpn_stLR3_0.1_nms0.3_b8_e20_small_obj_st0.5"

    MODEL_PATH = "params/" + SAVE_NAME + ".pt"

    OUTPUT_JSON_PATH = "pred.json"
    OUTPUT_CSV_PATH = "pred.csv"

    model = ftrcnn(num_classes=11)

    print(" == Task 1 Testing started! ==")
    test_task1(model, MODEL_PATH, TEST_DIR, test_transform,
                OUTPUT_JSON_PATH, visualize=True, visualize_path="temp/" + SAVE_NAME)

    print(" == Task 2 Testing started! ==")
    test_task2(OUTPUT_JSON_PATH, TEST_DIR, OUTPUT_CSV_PATH)
