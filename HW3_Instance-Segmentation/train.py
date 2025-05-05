"""
This script trains an object detection model (Faster R-CNN) for digit recognition.
"""
# Standard libraries
import math
import time
import os
import json
import gc

# Third-party libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
from collections import defaultdict

from pycocotools.cocoeval import COCOeval

# Custom modules
from model import light_maskrcnn_v2

from datasets import CustomDataset
from utils import (
    split_dataset_ids,
    set_seed,
    save_model_info,
    plot_training_history,
    convert_to_coco_format
)


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    This function is used to group a list of samples into batches. It separates inputs and targets
    by unzipping the batch and returning them as tuples. This is especially useful for models like
    Faster R-CNN and Mask R-CNN, where each image has its own annotations (targets).

    Args:
        batch (list): A list of (image, target) tuples.

    Returns:
        tuple: A tuple of two lists - (images, targets), where:
            - images: List of image tensors.
            - targets: List of corresponding annotation dictionaries.
    """
    return tuple(zip(*batch))


def train_model(model, dataloaders, dataset_sizes, config, writer):
    """
    Trains a Mask R-CNN model using a training/validation loop with support for early stopping and 
    learning rate scheduling.

    This function handles the training and validation of a PyTorch model using provided dataloaders,
    optimizer, learning rate scheduler, and TensorBoard logging. It supports early stopping based on
    validation mAP (mean Average Precision), and saves the best-performing model weights.

    Args:
        model (torch.nn.Module): The Mask R-CNN model to train.
        dataloaders (dict): A dictionary containing 'train' and 'val' DataLoader objects.
        dataset_sizes (dict): A dictionary with the sizes (number of samples) of the training and 
                              validation datasets.
        config (dict): Configuration dictionary containing training parameters such as:
            - 'optimizer': The optimizer used for training.
            - 'num_epochs' (int, optional): Total number of training epochs. Default is 100.
            - 'patience' (int, optional): Early stopping patience in epochs. Default is 20.
            - 'scheduler' (optional): Learning rate scheduler object.
            - 'save_pt_path' (str, optional): Path to save the best model parameters.
        writer (SummaryWriter): 
            TensorBoard SummaryWriter for logging training and validation metrics.

    Returns:
        model (torch.nn.Module): The trained model loaded with the best weights.
        history (defaultdict): A dictionary recording training loss and validation metrics 
                               (e.g., 'loss', 'val_mask_map').
    """
    optimizer = config.get('optimizer')
    num_epochs = config.get('num_epochs', 100)
    patience = config.get('patience', 20)
    scheduler = config.get('scheduler', None)
    save_pt_path = config.get(
        'save_pt_path', './local_storage/params/best_model_params.pt')

    # --- Device Setup ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()  # Start time

    # --- Checkpoint Setup ---
    best_model_params_path = save_pt_path
    os.makedirs(os.path.dirname(best_model_params_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_params_path)

    best_map = 0.0
    # Use defaultdict to store history (loss, val_loss, val_box_map, val_mask_map etc.)
    history = defaultdict(list)
    patience_c = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch:3d}/{num_epochs - 1}', end='\n')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # --- For val phase: collect outputs and targets for COCO evaluation ---
            outputs_list = []
            targets_list = []

            # Iterate over data.
            print(f"Phase: {phase}")
            for batch_idx, (images, targets) in enumerate(dataloaders[phase]):
                # Move data to the device
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Enable gradient calculation only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # In training mode, if targets are provided, returns a dictionary of losses
                    outputs = model(images, targets)

                    if phase == 'train':
                        losses = sum(loss for loss in outputs.values())
                        loss_value = losses.item()
                        running_loss += loss_value * len(images)

                        # Print training loss periodically
                        if batch_idx % 5 == 0:
                            print(
                                f'  Batch {batch_idx + 1:5d}: Loss = {loss_value:.4f}')
                            if writer:
                                writer.add_scalar(
                                    'Loss/train_batch', loss_value, 
                                    epoch * len(dataloaders[phase]) + batch_idx)

                        # backward + optimize only if in training phase
                        losses.backward()
                        optimizer.step()
                        scheduler.step()

                    else:  # val phase, the model returns predictions
                        # outputs is a list of dictionaries, one dictionary per image in the batch
                        # Each dictionary contains 'boxes', 'labels', 'scores', and 'masks'
                        preds = outputs
                        # Validation phase: save predictions and ground truth for COCO evaluation
                        for pred, target in zip(preds, targets):
                            pred = {k: v.detach().cpu()
                                    for k, v in pred.items()}
                            target = {k: v.detach().cpu()
                                      for k, v in target.items()}
                            outputs_list.append(pred)
                            targets_list.append(target)

            # --- End of Batch Loop ---

            # Calculate epoch loss
            # if phase == 'train' and scheduler:
            #     scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                history['loss'].append(epoch_loss)
                writer.add_scalar('Loss/train', epoch_loss,
                                  epoch)  # Log training loss
                print(f'{phase} Loss: {epoch_loss:.4f}', end='\n')
            else:
                print("Computing COCO metrics...")

                # === 這邊要做成 COCO 格式 ===
                coco_true, coco_pred = convert_to_coco_format(
                    targets_list, outputs_list)

                coco_eval = COCOeval(coco_true, coco_pred,
                                     iouType='segm')  # 'bbox' or 'segm'
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Extract mAP
                # mAP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
                epoch_mask_map = coco_eval.stats[0]
                epoch_mask_map_50 = coco_eval.stats[1]  # mAP @IoU=0.5

                history['val_mask_map'].append(epoch_mask_map)

                if writer:
                    writer.add_scalar('mAP/val_mask', epoch_mask_map, epoch)
                    writer.add_scalar('mAP_50/val_mask',
                                      epoch_mask_map_50, epoch)

                print(f'{phase} Mask mAP: {epoch_mask_map:.4f}')

                current_best_metric = epoch_mask_map

                # Save best model based on the chosen metric (e.g., Mask mAP)
                if current_best_metric > best_map:
                    best_map = current_best_metric
                    patience_c = 0  # Reset patience counter
                    print(f"Saving best model with Mask mAP: {best_map:.4f}")
                    try:
                        torch.save(model.state_dict(), best_model_params_path)
                    except Exception as e:
                        print(f"Error saving model checkpoint: {e}")
                else:
                    patience_c += 1  # Increment patience counter

        # --- End of Phase Loop ---
        time_elapsed = time.time() - since
        print(
            f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val mAP: {best_map:4f}\n\n')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        gc.collect()
        torch.cuda.empty_cache()

        if patience_c > patience:
            print("patience break!")
            break

    return model, history


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with a warmup phase followed by cosine annealing.

    This scheduler first linearly increases the learning rate from 0 to the base learning rate
    over `warmup_iters` steps. After that, it applies cosine annealing to gradually decrease the
    learning rate from the base value to 0 over the remaining `total_iters - warmup_iters` steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): Number of warm-up iterations with linear learning rate increase.
        total_iters (int): Total number of training iterations (warm-up + cosine annealing).
        last_epoch (int, optional): The index of the last epoch. Default: -1.

    Returns:
        List[float]: A list of learning rates for each parameter group.
    """
    def __init__(self, optimizer, warmup_iters, total_iters, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup階段：線性上升
            return [
                base_lr * float(self.last_epoch + 1) / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            # CosineAnnealing階段
            cos_iter = self.last_epoch - self.warmup_iters
            cos_total = self.total_iters - self.warmup_iters
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * cos_iter / cos_total))
                for base_lr in self.base_lrs
            ]


if __name__ == "__main__":
    DATA_DIR = 'data'
    SAVE_NAME = "best"

    annotation_file_path = DATA_DIR + '/train.json'
    image_directory = DATA_DIR + '/train'

    writer = SummaryWriter(f'local_storage/runs/{SAVE_NAME}')

    set_seed(63)

    # ================================ Handle dataset =========================================
    # Create training and validation datasets
    # 載入完整的 annotation 檔案以獲取所有圖片 ID
    print(
        f"Loading full annotation file from {annotation_file_path} to get all image IDs...")
    with open(annotation_file_path, 'r', encoding='utf-8') as f:
        full_coco_data = json.load(f)
    all_image_ids = sorted([img['id'] for img in full_coco_data['images']])
    print(f"Found {len(all_image_ids)} total images in the annotation file.")

    # 分割圖片 ID 列表為訓練集和驗證集
    train_ids, val_ids = split_dataset_ids(
        all_image_ids, train_size=0.8, random_state=63)
    print(
        f"Split into {len(train_ids)} training images and {len(val_ids)} validation images.")

    # Define data transformations
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 創建訓練集和驗證集的 CustomDataset 實例
    train_dataset = CustomDataset(
        annotation_file_path=annotation_file_path,
        image_dir=image_directory,
        image_ids_subset=train_ids,
        transform=train_transform
    )
    val_dataset = CustomDataset(
        annotation_file_path=annotation_file_path,
        image_dir=image_directory,
        image_ids_subset=val_ids,
        transform=val_transform
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=2, collate_fn=collate_fn)

    # Optionally print images with labels to verify dataset labels
    # if you want to use, you should import form utils
    # print_image_with_mask_for_segmentation(train_dataset,
    #                                        save_dir="local_storage/segmentation_viz")
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
    optimizer = None
    scheduler = None

    NUM_CLASSES = 5
    
    # model = maskrcnn_v2(num_classes=NUM_CLASSES)
    model = light_maskrcnn_v2(num_classes=NUM_CLASSES)

    # Define optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,               # 最終要到達的起始 lr
        momentum=0.9,
        weight_decay=0.0005
    )

    num_epochs = 50
    total_iters = len(train_loader) * num_epochs
    warmup_iters = total_iters // 20
    # total_iters = 2000
    # warmup_iters = 100
    scheduler = WarmupCosineLR(
        optimizer, warmup_iters=warmup_iters, total_iters=total_iters)

    training_config = {
        'optimizer': optimizer,
        'num_epochs': num_epochs,
        'patience': num_epochs//5,
        'scheduler': scheduler,
        'save_pt_path': "./local_storage/params/" + SAVE_NAME + ".pt",
    }

    # Start training
    print(" == Training started! ==")
    # Train the model
    trained_model, history = train_model(
        model,
        dataloaders={'train': train_loader, 'val': val_loader},
        dataset_sizes={'train': len(train_dataset), 'val': len(val_dataset)},
        config=training_config,
        writer=writer
    )

    print(" == Training completed! ==")
    writer.close()
    # ================================ Record model information =========================

    # Save model information
    # Convert model to string for saving
    trained_model_str = str(trained_model)

    # Count total parameters
    params_count = sum(p.numel() for p in trained_model.parameters())
    save_model_info(params_count, trained_model_str,
                    "./local_storage/info/" + SAVE_NAME + ".txt")

    # Plot and save training history
    print("Drawing loss figure ...")
    plot_training_history(history, f"./local_storage/info/{SAVE_NAME}.png")

    # Final message
    print(
        f"Everything is ok, you can use ./local_storage/params/{SAVE_NAME}.pt to test!")
