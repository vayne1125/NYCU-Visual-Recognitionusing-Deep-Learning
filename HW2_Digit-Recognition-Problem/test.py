"""
Test script for running inference on images using a trained model.

This script loads a trained deep learning model, processes test images,
performs inference, and outputs predictions to a CSV file.
"""
import csv
import os
from PIL import Image

import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

import random

import torch
from torchvision import transforms
from torch.backends import cudnn

from model import ftrcnn, ftrcnn2
from datasets import CustomDataset
from utils import load_config, visualize_predictions, set_seed, visualize_test_predictions

def test_task1(model, model_path, test_dir, transform, output_json_path, visualize=False, num_visualize=12, visualize_path = "temp_test_predictions"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    image_files = sorted(os.listdir(test_dir))
    predictions = []
    visualization_images = []
    visualization_preds = []
    visualization_filenames = []

    with torch.no_grad():
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(test_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            output = model(img_tensor)[0]

            image_id = int(os.path.splitext(image_file)[0])

            current_image_predictions = []
            for j in range(len(output['boxes'])):
                score = output['scores'][j].item()
                if score > 0.5:
                    bbox = output['boxes'][j].cpu().tolist()
                    label_index = output['labels'][j].item()
                    category_id = label_index + 1
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    bbox_wh = [bbox[0], bbox[1], width, height]

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

            if visualize and i < 8:  # Store data for the first 8 images
                visualization_images.append(img_tensor.squeeze(0).cpu())
                visualization_preds.append(current_image_predictions)
                visualization_filenames.append(image_file)

            if visualize and i == 7:
                class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                visualize_test_predictions(images=visualization_images,
                                          predictions=visualization_preds,
                                          class_names=class_names,
                                          save_dir=visualize_path,
                                          image_filenames=visualization_filenames)
                # Clear lists after visualization
                visualization_images.clear()
                visualization_preds.clear()
                visualization_filenames.clear()

    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)

    print(f"Task 1 predictions saved to {output_json_path}")


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def test_task2(input_json_path, test_image_dir, output_csv_path):
    with open(input_json_path, 'r') as f:
        predictions = json.load(f)

    image_predictions = {}
    for pred in predictions:
        image_id = pred['image_id']
        if image_id not in image_predictions:
            image_predictions[image_id] = []
        image_predictions[image_id].append(pred)

    # 獲取所有測試圖片的 image_id
    image_files = sorted(os.listdir(test_image_dir))
    all_image_ids = set()
    for file_name in image_files:
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # 根據你的圖片格式調整
            try:
                image_id = int(os.path.splitext(file_name)[0])
                all_image_ids.add(image_id)
            except ValueError:
                print(f"Warning: Could not extract image_id from filename: {file_name}")

    results_task2 = []
    for image_id in sorted(list(all_image_ids)):
        detections = image_predictions.get(image_id, []) # 使用 .get() 方法，如果 image_id 不存在則返回空列表
        if not detections:
            results_task2.append([image_id, -1])
        else:
            detections.sort(key=lambda x: x['bbox'][0])
            predicted_number_str = ""
            for detection in detections:
                predicted_number_str += str(detection['category_id']-1)
            if predicted_number_str:
                try:
                    predicted_number = int(predicted_number_str)
                    results_task2.append([image_id, predicted_number])
                except ValueError:
                    results_task2.append([image_id, -1])
            else:
                results_task2.append([image_id, -1])

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "pred_label"])
        writer.writerows(results_task2)

    print(f"Task 2 predictions saved to {output_csv_path}")

if __name__ == "__main__":

    data_dir = 'nycu-hw2-data'
    cudnn.benchmark = True
    set_seed(63)
    test_dir = os.path.join(data_dir, "test")

    save_name = "resnet50_fpn_trainall_eta_1e-6_tmax_20"

    # Define your object detection model here
    model = ftrcnn2(num_classes=10, backbone_name='resnet50_fpn', pretrained=False, train_all_layers=False)
    # model = ftrcnn(num_classes=10, backbone_name='resnet50_fpn', pretrained=True) # Adjust num_classes
    model_path = "params/" + save_name + ".pt" # Use the path to your trained object detection model

    output_json_path = save_name + "/pred.json"
    output_csv_path = save_name + "/pred.csv"

    print(" == Task 1 Testing started! ==")
    test_task1(model, model_path, test_dir, test_transform, output_json_path, visualize=True, visualize_path = "temp/" + save_name)

    print(" == Task 2 Testing started! ==")
    test_task2(output_json_path, test_dir, output_csv_path)
