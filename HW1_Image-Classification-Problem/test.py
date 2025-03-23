"""
Test script for running inference on images using a trained model.

This script loads a trained deep learning model, processes test images,
performs inference, and outputs predictions to a CSV file.
"""
import csv
import os
from PIL import Image

import torch
from torchvision import transforms
from torch.backends import cudnn

from model.resnext import build_resnext
from model.resnet import build_resnet
from datasets.datasets import CustomDataset
from utils.utils import load_config

def test(model, model_path, test_dir, transform, output_path):
    """
    Run inference on test images and save predictions to a CSV file.

    Args:
        model (torch.nn.Module): The trained model for inference.
        model_path (str): Path to the trained model parameters.
        test_dir (str): Directory containing test images.
        transform (torchvision.transforms.Compose): Transformations to apply to images.
        output_csv (str): Path to save prediction results.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    image_files = sorted(os.listdir(test_dir))  # Ensure consistent order
    results = []

    with torch.no_grad():
        for image_file in image_files:
            image_path = os.path.join(test_dir, image_file)
            image = Image.open(image_path).convert("RGB")

            preds = []
            for transform in tta_transforms:
                # Apply transformation and add batch dimension
                img_t = transform(image).unsqueeze(0).to(device)
                output = model(img_t)
                preds.append(torch.softmax(output, dim=1))  # Convert logits to probabilities

            avg_pred = torch.mean(torch.stack(preds), dim=0)  # Average predictions
            final_class = torch.argmax(avg_pred, dim=1).item()

            # Store results (remove file extension from image name)
            results.append([os.path.splitext(image_file)[0], final_class])

    # Save predictions to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(results)

    print(f"結果已輸出到 {output_path}")


if __name__ == "__main__":
    """
    Test the model's performance.

    Before running the test, ensure the model architecture matches the 
    one used for training to avoid loading errors.
    """

    # Modify parameters in config.yaml if necessary
    config = load_config("config.yaml")
    model_config = config['model']
    data_config = config['data']
    testing_config = config['testing']

    # Extract model configuration settings
    model_type = model_config['type']
    model_layer = model_config['layer']
    model_version = model_config['version']

    # Extract testing configuration settings
    model_path = testing_config['params_pth']

    # Set the data directory path
    data_dir = data_config['data_dir']

    cudnn.benchmark = True  # Enable cuDNN auto-tuner for better performance

    # Define TTA transformations
    tta_transforms = [
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),  # Standard transformation

        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=1.0),  # Flip horizontally
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomRotation(15),  # Rotate by ±15 degrees
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),

        transforms.Compose([
            transforms.Resize(256),
            # Randomly crop between 80%-100% of the original size
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

    # Set test dataset dir with data_dir
    test_dir = os.path.join(data_dir, "test")

    # Load training dataset to get the number of classes
    train_dataset = CustomDataset(data_dir=os.path.join(data_dir, "train"), transform=None)
    class_names = train_dataset.class_names  # Ensure class count consistency

    # Build model based on type
    if model_type == "resnext":
        model = build_resnext(len(class_names), model_layer, model_version, False)
    elif model_type == "resnet":
        model = build_resnet(len(class_names), model_layer, model_version, False)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Output file for predictions
    output_path = "prediction.csv"  # pylint: disable=invalid-name

    print(" == Testing started! ==")
    test(model, model_path, test_dir, tta_transforms, output_path)
