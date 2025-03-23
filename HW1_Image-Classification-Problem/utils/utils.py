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
import matplotlib.pyplot as plt
import numpy as np
import yaml

def print_image_with_label(train_dataset, save_dir="temp"):
    """
    This function displays a set of images from the training dataset
    along with their corresponding labels.
    It saves the resulting image grid to a specified directory.

    Args:
        train_dataset (Dataset): The training dataset containing images and their labels.
        save_dir (str, optional): Directory to save the displayed image grid. Defaults to "temp".
    """
    # Set the number of images to display
    N = 24

    # Create a directory to save images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a canvas to display the images
    plt.figure(figsize=(12, 8))

    # Display images and their corresponding labels
    for i in range(N):
        plt.subplot(4, 6, i + 1)

        # Randomly select an image
        id = random.randint(0, len(train_dataset) - 1)
        img_tensor, label = train_dataset[id]

        # Get the image path
        img_path = train_dataset.image_paths[id]

        # Normalize the image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img_tensor * std[:, None, None] + mean[:, None, None]

        # Output image path and label
        print(f"Image Path: {img_path}, Label: {label}")

        # Display the image, converting format from CxHxW to HxWxC
        plt.imshow(img.permute(1, 2, 0))
        plt.title(label)
        plt.axis('off')

    # Save the images to the 'temp' directory
    temp_image_path = os.path.join(save_dir, "images_with_labels.png")
    plt.tight_layout()
    plt.savefig(temp_image_path)
    plt.close()

    print(f"Images saved to {temp_image_path}")


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
    Plot the training history of loss and accuracy, then save the results as an image.

    :param history: Training history, a list of results from each fold
    :param save_name: Path to save the plot image
    """

    # Set the figure size
    plt.figure(figsize=(8, 3))

    max_loss = 0
    max_loss = max(max_loss,np.max(history['loss']),np.max(history['val_loss']))

    # Increase margin for better visualization
    max_loss *= 1.05

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='valid')
    plt.xlabel('epoch')

    step = int(np.ceil(len(history['loss']) / 5))
    x_ticks = np.arange(0, len(history['loss']), step)
    x_labels = [str(u + 1) for u in x_ticks]

    plt.xticks(x_ticks, x_labels)
    plt.ylabel('loss')
    plt.ylim([0, max_loss])
    plt.grid(True)
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='valid')

    x_ticks = np.arange(0, len(history['accuracy']), step)
    x_labels = [str(u + 1) for u in x_ticks]

    plt.xticks(x_ticks, x_labels)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
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
