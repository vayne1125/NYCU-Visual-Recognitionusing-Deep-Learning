"""
This script is for training a deep learning model using PyTorch. 

It supports:
- Loading and parsing configuration from a YAML file
- Initializing different types of models (e.g., ResNet, ResNeXt)
- Setting up data augmentation and normalization for training and validation datasets
- Configuring optimizers, schedulers, and loss functions
- Running the training loop and saving the trained model and training history

The model type, batch size, learning rate, 
and other hyperparameters are configurable through the config file.
"""
# Standard libraries
import time
import os

# Third-party libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Custom modules
from datasets.datasets import CustomDataset
from model.resnext import build_resnext
from model.resnet import build_resnet
from utils.utils import (
    save_model_info,
    plot_training_history,
    set_seed,
    load_config,
    print_image_with_label
)

def train_model(model, dataloaders, dataset_sizes, config):
    """
    Train a model using the given parameters.

    Parameters:
    - model: The neural network model to be trained.
    - dataloaders: A dictionary containing 'train' and 'val' dataloaders.
    - dataset_sizes: A dictionary containing the sizes of the datasets.
    - config: A dictionary containing configuration options like num_epochs, patience, etc.
    """

    criterion = config.get('criterion')
    optimizer = config.get('optimizer')
    num_epochs = config.get('num_epochs', 100)
    patience = config.get('patience', 20)
    scheduler = config.get('scheduler', None)
    save_pt_path = config.get('save_pt_path', './params/best_model_params')

    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()

    best_model_params_path = save_pt_path
    os.makedirs(os.path.dirname(best_model_params_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_params_path)

    best_loss = None
    best_acc  = 0
    history = {'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]}
    patience_c = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch:3d}/{num_epochs - 1}',end=' ')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}',
                    end=' ' if phase == 'train' else '\n')
            if phase == 'train':
                history['loss'].append(epoch_loss)
                history['accuracy'].append(epoch_acc.cpu().numpy())

            # deep copy the model
            if phase == 'val':
                history['val_loss'].append(epoch_loss)
                history['val_accuracy'].append(epoch_acc.cpu().numpy())

                if best_acc < epoch_acc.cpu().numpy():
                    best_acc = epoch_acc.cpu().numpy()
                if epoch == 0 or epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_c= 0
                    torch.save(model.state_dict(), best_model_params_path)
                else:
                    patience_c += 1

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        if patience_c > patience:
            print("patience break!")
            break

    return model, history

if __name__ == "__main__":

    # Load and display config settings
    # Modify parameters in config.yaml if necessary
    config = load_config("config.yaml")

    training_config = config['training']
    model_config = config['model']
    data_config = config['data']

    # Extract model configuration settings
    model_type = model_config['type']
    model_layer = model_config['layer']
    model_version = model_config['version']
    model_trainable = model_config['trainable']
    model_trainable_layers = model_config['trainable_layers']

    # Extract training configuration settings
    seed = training_config['seed']
    batch_size = training_config['batch_size']
    epochs = training_config['epochs']
    lr = training_config['lr']
    weight_decay = training_config['weight_decay']
    eta_min = training_config['eta_min']

    # Set the data directory path
    data_dir = data_config['data_dir']

    save_name = f"{model_type}{model_layer}v{model_version}_b{batch_size}_e{epochs}_lr{lr}_sd{seed}"

    # Set random seed for reproducibility
    set_seed(seed)

    # Print training configuration
    print("Training Configuration:")
    print(f"  Seed: {seed}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  eta_min: {eta_min}")

    # Print model configuration
    print("\nModel Configuration:")
    print(f"  Model Type: {model_type}")
    print(f"  Model Layers: {model_layer}")
    print(f"  Model Version: {model_version}")
    print(f"  Model Trainable: {model_trainable}")
    print(f"  Model Trainable Layers: {model_trainable_layers}")

    # Print data configuration
    print("\nData Configuration:")
    print(f"  Dataset Path: {data_dir}")

    # Print save name
    print(f"\nSave Name: {save_name}\n\n")

    # ================================ Handle dataset =========================================
    # Define data transformations for training and validation
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),  # Adjust brightness, contrast, saturation, hue
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            ),  # Normalize the image with specific mean and std
            transforms.RandomRotation(10),  # Random rotation between ±10 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Create training and validation datasets
    train_dataset = CustomDataset(data_dir= data_dir + "train", transform=train_transform)
    val_dataset = CustomDataset(data_dir= data_dir + "val", transform=val_transform)

    # Retrieve class names from the training dataset
    class_names = train_dataset.class_names

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Optionally print images with labels to verify dataset labels
    # print_image_with_label(train_dataset)

    # =============================== Check GPU availability ==============================

    # Determine the device to be used (GPU or CPU)
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Check GPU availability
    print(torch.cuda.is_available())  # True if GPU is available
    print(torch.cuda.current_device())  # Get the current GPU ID
    print(torch.cuda.get_device_name(0))  # Get the name of the GPU

    # ===================================== Training ======================================

    # Initialize model, optimizer, and scheduler
    model = None
    optimizer_ft = None
    scheduler = None

    # Choose model based on configuration
    # If 'model_trainable' is True, ignore 'model_trainable_layers'.
    # To fine-tune the last 2 layers, set 'model_trainable' to False
    # and 'model_trainable_layers' to 2.
    if model_type == "resnext":
        model = build_resnext(len(class_names), model_layer, model_version)
    elif model_type == "resnet":
        model = build_resnet(
            len(class_names),
            model_layer,
            model_version,
            model_trainable,
            model_trainable_layers
        )
    else:
        raise ValueError(f"Invalid model type {model_type}. Choose 'resnext' or 'resnet'.")

    # Set early stopping patience
    patience = epochs//5

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=eta_min)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    training_config = {
        'criterion': criterion,
        'optimizer': optimizer,
        'num_epochs': epochs,
        'patience': patience,
        'scheduler': scheduler,
        'save_pt_path': "./params/" + save_name + ".pt"
    }

    # Start training
    print(" == Training started! ==")

    # Train the model
    trained_model, history = train_model(
        model,
        dataloaders={'train': train_loader, 'val': val_loader},
        dataset_sizes={'train': len(train_dataset), 'val': len(val_dataset)},
        config=training_config
    )

    print(" == Training completed! ==")

    # ================================ Record model information =========================

    # Save model information
    trained_model_str = str(trained_model)  # Convert model to string for saving
    params_count = sum(p.numel() for p in trained_model.parameters())  # Count total parameters
    save_model_info(params_count, trained_model_str, "./info/" + save_name + ".txt")

    # Plot and save training history
    print("Drawing loss figure ...")
    plot_training_history(history, f"./info/{save_name}.png")

    # Final message
    print(f"Everything is ok, you can use ./params/{save_name}.pt to test!")