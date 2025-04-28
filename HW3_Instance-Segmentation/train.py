"""
This script trains an object detection model (Faster R-CNN) for digit recognition.
"""
# Standard libraries
import time
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# Third-party libraries
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchmetrics.detection import MeanAveragePrecision
import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Custom modules
from model import maskrcnn_v2
from datasets import CustomDataset
from utils import (
    split_dataset_ids,
    set_seed,
    print_image_with_mask_for_segmentation,
    save_model_info,
    plot_training_history,
    encode_mask
)


def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(model, dataloaders, dataset_sizes, config, writer):

    optimizer = config.get('optimizer')
    num_epochs = config.get('num_epochs', 100)
    patience = config.get('patience', 20)
    scheduler = config.get('scheduler', None)
    save_pt_path = config.get('save_pt_path', './local_storage/params/best_model_params.pt')
    # val_visualize_freq = config.get('val_visualize_freq', 200)
    # iou_threshold_accuracy = config.get('iou_threshold_accuracy', 0.5)

    # --- Device Setup ---
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    since = time.time()  # Start time

    # --- Checkpoint Setup ---
    best_model_params_path = save_pt_path
    os.makedirs(os.path.dirname(best_model_params_path), exist_ok=True)
    torch.save(model.state_dict(), best_model_params_path)

    best_map = 0.0
    history = defaultdict(list) # Use defaultdict to store history (loss, val_loss, val_box_map, val_mask_map etc.)
    patience_c = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch:3d}/{num_epochs - 1}',end='\n')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Lists to store COCO-formatted predictions and ground truths for the entire validation epoch
            # Note: Accumulating for the whole epoch is the standard COCO eval approach
            # If validation set is very large, this might consume significant memory.
            # For smaller datasets, this is fine.
            coco_gt_annotations_epoch = []
            coco_dt_annotations_epoch = []
            image_ids_epoch = [] # Keep track of image IDs in the validation set

            # Get image info and category info from the validation dataset instance
            if phase == 'val':
                val_dataset_instance = dataloaders[phase].dataset
                # Assuming dataset has images_info and categories_info attributes from JSON loading
                images_info_val = val_dataset_instance.images_info
                categories_info_val = val_dataset_instance.categories_info
                # Create a mapping for quick access to image info by ID
                image_id_to_info_val = {img['id']: img for img in images_info_val}

            # Iterate over data.
            print(f"Phase: {phase}")
            for batch_idx, (images, targets) in enumerate(dataloaders[phase]):
                # Move data to the device
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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
                            print(f'  Batch {batch_idx + 1:5d}: Loss = {loss_value:.4f}')
                            if writer:
                                writer.add_scalar('Loss/train_batch', loss_value, epoch * len(dataloaders[phase]) + batch_idx)

                        # backward + optimize only if in training phase
                        losses.backward()
                        optimizer.step()

                    else: # val phase, the model returns predictions
                        # outputs is a list of dictionaries, one dictionary per image in the batch
                        # Each dictionary contains 'boxes', 'labels', 'scores', and 'masks'
                        preds = outputs

                        # --- Convert predictions and targets to COCO format for COCOeval ---
                        for i in range(len(preds)): # Iterate through each image in the batch
                            img_id = targets[i]['image_id'].item()
                            image_ids_epoch.append(img_id) # Collect image IDs for epoch evaluation

                            # Convert Ground Truth to COCO format
                            # Assuming targets[i] contains 'boxes', 'labels', 'masks', 'image_id', 'area', 'iscrowd'
                            # 'iscrowd' might not be in your targets, default to 0
                            for j in range(len(targets[i]['boxes'])):
                                gt_ann = {
                                    'image_id': img_id,
                                    'category_id': targets[i]['labels'][j].item(),
                                    'bbox': targets[i]['boxes'][j].tolist(), # [x_min, y_min, x_max, y_max] -> COCO [x, y, w, h]
                                    'segmentation': targets[i]['masks'][j].cpu().numpy(), # Mask tensor -> numpy array
                                    'area': (targets[i]['boxes'][j][2] - targets[i]['boxes'][j][0]) * (targets[i]['boxes'][j][3] - targets[i]['boxes'][j][1]), # Calculate area
                                    'iscrowd': targets[i].get('iscrowd', torch.tensor([0])).tolist()[0], # Default iscrowd to 0
                                    'id': len(coco_gt_annotations_epoch) + 1 # Assign a unique ID for the annotation
                                }
                                # Convert bbox from [x_min, y_min, x_max, y_max] to [x, y, w, h] for COCO
                                x_min, y_min, x_max, y_max = gt_ann['bbox']
                                gt_ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]

                                # Convert mask numpy array to RLE format
                                # pycocotools.mask.encode expects Fortran contiguous numpy array
                                # Need to squeeze the channel dimension if it exists (shape [H, W])
                                mask_np = gt_ann['segmentation']
                                if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                                     mask_np = mask_np.squeeze(0)

                                # Ensure mask is boolean or uint8
                                if mask_np.dtype != np.uint8:
                                     mask_np = mask_np.astype(np.uint8)

                                # Ensure Fortran contiguous
                                mask_np = np.asfortranarray(mask_np)

                                gt_ann['segmentation'] = encode_mask(mask_np)
                                # Ensure counts in segmentation is bytes
                                if isinstance(gt_ann['segmentation']['counts'], str):
                                     gt_ann['segmentation']['counts'] = gt_ann['segmentation']['counts'].encode('utf-8')


                                coco_gt_annotations_epoch.append(gt_ann)


                            # Convert Predictions to COCO format
                            # Apply a threshold to the predicted masks (e.g., 0.5)
                            mask_threshold = 0.5 # Common threshold for binary mask
                            pred_masks_binary = (preds[i]['masks'] > mask_threshold).squeeze(1).cpu().numpy() # Shape [num_predictions, H, W], boolean or uint8

                            for j in range(len(preds[i]['boxes'])):
                                pred_ann = {
                                    'image_id': img_id,
                                    'category_id': preds[i]['labels'][j].item(),
                                    'bbox': preds[i]['boxes'][j].tolist(), # [x_min, y_min, x_max, y_max] -> COCO [x, y, w, h]
                                    'score': preds[i]['scores'][j].item(),
                                    'segmentation': pred_masks_binary[j], # Binary mask numpy array
                                    'area': (preds[i]['boxes'][j][2] - preds[i]['boxes'][j][0]) * (preds[i]['boxes'][j][3] - preds[i]['boxes'][j][1]), # Calculate area
                                    'id': len(coco_dt_annotations_epoch) + 1 # Assign a unique ID for the prediction
                                }
                                # Convert bbox from [x_min, y_min, x_max, y_max] to [x, y, w, h] for COCO
                                x_min, y_min, x_max, y_max = pred_ann['bbox']
                                pred_ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]

                                # Convert mask numpy array to RLE format
                                mask_np = pred_ann['segmentation']
                                # Ensure Fortran contiguous
                                mask_np = np.asfortranarray(mask_np)
                                pred_ann['segmentation'] = encode_mask(mask_np)
                                # Ensure counts in segmentation is bytes
                                if isinstance(pred_ann['segmentation']['counts'], str):
                                     pred_ann['segmentation']['counts'] = pred_ann['segmentation']['counts'].encode('utf-8')

                                coco_dt_annotations_epoch.append(pred_ann)

                        # --- End Conversion ---
            # --- End of Batch Loop ---

            # Calculate epoch loss
            if phase == 'train' and scheduler:
                scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / dataset_sizes[phase]
                history['loss'].append(epoch_loss)
                writer.add_scalar('Loss/train', epoch_loss, epoch) # Log training loss
                print(f'{phase} Loss: {epoch_loss:.4f}', end='\n')
            else:
                print("Computing validation metrics using COCOeval...")
                try:
                    # Create dummy COCO objects for evaluation
                    # Ground truth COCO object
                    cocoGt = COCO()
                    # Need to add image info and category info to the dataset
                    # Assuming val_dataset_instance has images_info and categories_info
                    cocoGt.dataset = {
                        'images': [image_id_to_info_val[img_id] for img_id in sorted(list(set(image_ids_epoch)))],
                        'annotations': coco_gt_annotations_epoch,
                        'categories': categories_info_val
                    }
                    cocoGt.createIndex() # Create index for faster lookup

                    # Predicted COCO object
                    cocoDt = COCO()
                    cocoDt.dataset = {
                        'images': [image_id_to_info_val[img_id] for img_id in sorted(list(set(image_ids_epoch)))],
                        'annotations': coco_dt_annotations_epoch,
                        'categories': categories_info_val # Include categories in detections as well
                    }
                    cocoDt.createIndex() # Create index

                    # Initialize COCOeval object
                    coco_eval = COCOeval(cocoGt, cocoDt, iouType='segm') # Specify iouType='segm' for instance segmentation

                    # Set parameters (optional, default is usually fine)
                    # coco_eval.params.imgIds = sorted(list(set(image_ids_epoch))) # Evaluate only images present in this epoch's validation batch
                    # coco_eval.params.catIds = sorted([cat['id'] for cat in categories_info_val]) # Evaluate all categories

                    # Run evaluation
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    # coco_eval.summarize() # Summarize prints to console, we'll extract stats instead

                    # Extract metrics from coco_eval.stats
                    # The indices for stats are fixed for COCO evaluation
                    # See COCOeval documentation for full list
                    # stats[0]: AP at IoU=[.5:.05:.95] for all areas and maxDets=100 (Overall Mask AP)
                    # stats[1]: AP at IoU=.50 for all areas and maxDets=100 (Mask AP@50)
                    # stats[2]: AP at IoU=.75 for all areas and maxDets=100 (Mask AP@75)
                    # stats[3-5]: AP for small, medium, large objects
                    # stats[6-8]: AR for maxDets 1, 10, 100
                    # stats[9-11]: AR for small, medium, large objects

                    epoch_mask_map = coco_eval.stats[0] # Overall Mask AP
                    epoch_mask_map_50 = coco_eval.stats[1] # Mask AP@50
                    # You can extract other metrics as needed, e.g., coco_eval.stats[2] for Mask AP@75

                    # Note: Box mAP metrics are also available if iouType was 'bbox' or if the metric was initialized for both.
                    # With iouType='segm', stats[0] is Mask AP.

                    history['val_mask_map'].append(epoch_mask_map)
                    history['val_mask_map_50'].append(epoch_mask_map_50)
                    # Add other metrics to history if extracted

                except Exception as e:
                     print(f"Error computing validation metrics with COCOeval: {e}")
                     # If metric computation fails, don't update best_map or patience_c based on it
                     # You might want to log this error
                     pass
                
                # Log validation metrics to TensorBoard
                if writer:
                    writer.add_scalar('mAP/val_mask', epoch_mask_map, epoch)
                    writer.add_scalar('mAP_50/val_mask', epoch_mask_map_50, epoch)


                print(f'{phase} Mask mAP (IoU=0.5:0.95): {epoch_mask_map:.4f}')
                print(f'{phase} Mask mAP@50: {epoch_mask_map_50:.4f}')

                # Decide which metric to use for saving the best model and patience
                # Mask mAP (especially mask_map or mask_map_50) is usually the primary metric for segmentation
                current_best_metric = epoch_mask_map # Using overall Mask mAP for saving

                # Save best model based on the chosen metric (e.g., Mask mAP)
                if current_best_metric > best_map:
                    best_map = current_best_metric
                    patience_c = 0 # Reset patience counter
                    print(f"Saving best model with Mask mAP: {best_map:.4f}")
                    try:
                        torch.save(model.state_dict(), best_model_params_path)
                    except Exception as e:
                        print(f"Error saving model checkpoint: {e}")
                else:
                    patience_c += 1 # Increment patience counter

        # --- End of Phase Loop ---
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val mAP: {best_map:4f}\n\n')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

        if patience_c > patience:
            print("patience break!")
            break

    return model, history

if __name__ == "__main__":
    DATA_DIR = 'data'
    SAVE_NAME = "test"
    
    annotation_file_path = DATA_DIR + '/train.json'
    image_directory = DATA_DIR + '/train'
    
    writer = SummaryWriter(f'local_storage/runs/{SAVE_NAME}')

    set_seed(63)

    # ================================ Handle dataset =========================================
    # Create training and validation datasets
    # 載入完整的 annotation 檔案以獲取所有圖片 ID
    print(f"Loading full annotation file from {annotation_file_path} to get all image IDs...")
    with open(annotation_file_path, 'r', encoding='utf-8') as f:
        full_coco_data = json.load(f)
    all_image_ids = sorted([img['id'] for img in full_coco_data['images']])
    print(f"Found {len(all_image_ids)} total images in the annotation file.")

    # 分割圖片 ID 列表為訓練集和驗證集
    train_ids, val_ids = split_dataset_ids(all_image_ids, train_size=0.2, random_state=63)
    print(f"Split into {len(train_ids)} training images and {len(val_ids)} validation images.")

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
        annotation_file_path = annotation_file_path,
        image_dir = image_directory,
        image_ids_subset=train_ids,
        transform = train_transform
    )
    val_dataset = CustomDataset(
        annotation_file_path = annotation_file_path,
        image_dir = image_directory,
        image_ids_subset=val_ids,
        transform = val_transform
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    # Optionally print images with labels to verify dataset labels
    # print_image_with_mask_for_segmentation(train_dataset,save_dir="local_storage/segmentation_viz")
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
    model = maskrcnn_v2(num_classes=NUM_CLASSES)
    
    # Define optimizer and scheduler
    LEARNING_RATE = 0.0001 # Can try 1e-4, 3e-4 etc.
    WEIGHT_DECAY = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Use StepLR, you need to set step_size and gamma
    GAMMA = 0.1     # For example, multiply the learning rate by 0.1 at each step
    STEP_SIZE = 3
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # os.makedirs(os.path.dirname("./local_storage/params/x.cpp" + SAVE_NAME + ".pt"), exist_ok=True)

    training_config = {
        'optimizer': optimizer,
        'num_epochs': 20,
        'patience': 5,
        'scheduler': scheduler,
        'save_pt_path': "./local_storage/params/" + SAVE_NAME + ".pt",
        'val_visualize_freq': 200,
        'iou_threshold_accuracy': 0.5
    }

    # Start training
    print(" == Training started! ==")

    # Train the model
    trained_model, history = train_model(
        model,
        dataloaders = {'train': train_loader, 'val': val_loader},
        dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)},
        config = training_config,
        writer = writer
    )

    print(" == Training completed! ==")
    writer.close()
    # ================================ Record model information =========================

    # Save model information
    trained_model_str = str(trained_model)  # Convert model to string for saving

    # Count total parameters
    params_count = sum(p.numel() for p in trained_model.parameters())
    save_model_info(params_count, trained_model_str, "./local_storage/info/" + SAVE_NAME + ".txt")

    # Plot and save training history
    print("Drawing loss figure ...")
    plot_training_history(history, f"./local_storage/info/{SAVE_NAME}.png")

    # Final message
    print(f"Everything is ok, you can use ./local_storage/params/{SAVE_NAME}.pt to test!")
