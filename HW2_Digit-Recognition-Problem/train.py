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
from torch.optim.lr_scheduler import StepLR

# Third-party libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from model import ftrcnn, ftrcnn2
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import CosineAnnealingLR

# Custom modules
from datasets import CustomDataset
from utils import (
    save_model_info,
    plot_training_history,
    set_seed,
    load_config,
    print_image_with_label_for_detection,
    visualize_predictions
)

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 可以添加其他 data augmentation transforms (例如 RandomHorizontalFlip, RandomCrop 等)
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn_custom(batch):
    return tuple(zip(*batch))

def train_model(model, dataloaders, dataset_sizes, config):
    """
    Train a model using the given parameters.

    Parameters:
    - model: The neural network model to be trained.
    - dataloaders: A dictionary containing 'train' and 'val' dataloaders.
    - dataset_sizes: A dictionary containing the sizes of the datasets.
    - config: A dictionary containing configuration options like num_epochs, patience, etc.
    """

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
    best_map = 0.0
    history = {'loss':[], 'val_loss':[], 'val_map':[]}
    patience_c = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch:3d}/{num_epochs - 1}',end='\n')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            batch_idx = -1
            running_loss = 0.0
            # running_corrects = 0 # Removed accuracy tracking

            # Initialize MeanAveragePrecision metric
            metric = MeanAveragePrecision()

            # Iterate over data.
            for images, targets in dataloaders[phase]:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # zero the parameter gradients
                optimizer.zero_grad()

                batch_idx += 1

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # for target in targets:
                        # print("Target Boxes:", target['boxes'])
                        # print("Target Boxes Shape:", target['boxes'].shape)
                        # print("Target Labels:", target['labels'])
                        # print("Target Labels Shape:", target['labels'].shape)

                    outputs = model(images, targets) # 在訓練模式下，如果提供了 targets，會返回損失字典

                    if phase == 'train':
                        losses = sum(loss for loss in outputs.values())

                        # backward + optimize only if in training phase
                        losses.backward()
                        optimizer.step()

                    else: # val 階段，模型會返回預測結果
                        preds = model(images)
                        metric.update(preds, targets)
                        

                        # 對每個圖像的預測結果應用 NMS
                        nms_preds = []
                        for i in range(len(preds)):
                            pred = preds[i]
                            boxes = pred['boxes']
                            scores = pred['scores']
                            labels = pred['labels']

                            keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5) # 你需要 import torchvision.ops
                            nms_preds.append({
                                'boxes': boxes[keep_indices],
                                'labels': labels[keep_indices],
                                'scores': scores[keep_indices]
                            })

                        # 在驗證階段繪製預測結果 (每隔一定 batch 數)
                        if batch_idx % 200 == 0: # 例如每 200 個驗證 batch 印一次
                            image_filenames = []
                            for target in targets:
                                img_id = target['image_id'].item()
                                img_info = dataloaders['val'].dataset.data['images'][img_id]
                                image_filenames.append(img_info['file_name'])
                            
                            gt_boxes = [target['boxes'] for target in targets]
                            gt_labels = [target['labels'] for target in targets]
                            
                            visualize_predictions(images, nms_preds, gt_boxes=gt_boxes, gt_labels=gt_labels, class_names=train_dataset.class_names, save_dir="temp_val_predictions", epoch=epoch, batch_idx=batch_idx, image_filenames=image_filenames)

                # statistics
                if isinstance(outputs, dict):
                    loss = sum(loss for loss in outputs.values())
                    running_loss += loss.item() * len(images) # 使用 batch size

                    # 每 10 個 batch 輸出一次
                    if phase == 'train':
                        if batch_idx % 100 == 0:
                            print(f'  Batch {batch_idx:5d}: Loss = {loss.item():.4f}')

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase] # Removed accuracy calculation

            if phase == 'train':
                history['loss'].append(epoch_loss)
                print(f'{phase} Loss: {epoch_loss:.4f}', end=' ')
                # history['accuracy'].append(epoch_acc.cpu().numpy()) # Removed accuracy recording
            else:
                # Compute the mAP
                metric_dict = metric.compute()
                epoch_map = metric_dict['map'].item() # Overall mAP
                history['val_loss'].append(epoch_loss)
                history['val_map'].append(epoch_map)
                print(f'{phase} Loss: {epoch_loss:.4f} mAP: {epoch_map:.4f}')

                # Save best model based on mAP
                if epoch_map > best_map:
                    best_map = epoch_map
                    patience_c = 0
                    torch.save(model.state_dict(), best_model_params_path)
                else:
                    patience_c += 1


        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val mAP: {best_map:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        if patience_c > patience:
            print("patience break!")
            break

    return model, history

if __name__ == "__main__":

    # Set the data directory path
    # data_dir = data_config['data_dir']

    data_dir = 'nycu-hw2-data'
    # save_name = f"{model_type}{model_layer}v{model_version}_b{batch_size}_e{epochs}_lr{lr}_sd{seed}"
    save_name = "resnet50_fpn_trainall_eta_1e-6_tmax_20"
    # Set random seed for reproducibility
    set_seed(63)

    # ================================ Handle dataset =========================================
    # Define data transformations for training and validation


    # Create training and validation datasets
    train_dataset = CustomDataset(
        json_path = data_dir + '/train.json',
        img_dir = data_dir+'/train',
        transform = train_transform
    )
    val_dataset = CustomDataset(
        json_path = data_dir + '/valid.json',
        img_dir = data_dir +'/valid',
        transform = val_transform
    )
    # 設定類別名稱
    train_dataset.class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Retrieve class names from the training dataset
    # class_names = train_dataset.class_names

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn_custom)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_custom)

    # Optionally print images with labels to verify dataset labels
    print_image_with_label_for_detection(train_dataset)
    # exit()
    # =============================== Check GPU availability ==============================

    # Determine the device to be used (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    num_classes = 10
    # model = ftrcnn(num_classes=10, backbone_name='resnet50_fpn', pretrained=True)

    # test in 3090
    model = ftrcnn2(num_classes=10, backbone_name='resnet50_fpn', pretrained=True, train_all_layers=True)
    
    # Set early stopping patience
    # patience = epochs//5

    # Define optimizer and scheduler
    # learning_rate = 0.001  # 可以嘗試 0.001, 0.005, 0.01 等
    # momentum = 0.9
    # weight_decay = 0.0005 # L2 正則化
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    learning_rate = 0.0001 # 可以嘗試 1e-4, 3e-4 等
    weight_decay = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.000001)

    training_config = {
        'optimizer': optimizer,
        'num_epochs': 20,
        'patience': 5,
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

    print("\nRemember to consider enabling data augmentation in train_transform for better generalization.")