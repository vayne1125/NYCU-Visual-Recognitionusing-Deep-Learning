"""
This script trains an object detection model (Faster R-CNN) for digit recognition.
"""
# Standard libraries
import time
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

# Third-party libraries
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchmetrics.detection import MeanAveragePrecision

# Custom modules
from model import ftrcnn
from datasets import CustomDataset
from utils import (
    save_model_info,
    plot_training_history,
    set_seed,
    print_image_with_label_for_detection,
    visualize_predictions
)

# Define data transformations
train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def collate_fn_custom(batch):
    return tuple(zip(*batch))

def train_model(model, dataloaders, dataset_sizes, config, writer):

    optimizer = config.get('optimizer')
    num_epochs = config.get('num_epochs', 100)
    patience = config.get('patience', 20)
    scheduler = config.get('scheduler', None)
    save_pt_path = config.get('save_pt_path', './params/best_model_params')
    val_visualize_freq = config.get('val_visualize_freq', 200)
    iou_threshold_accuracy = config.get('iou_threshold_accuracy', 0.5)

    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    since = time.time()

    best_model_params_path = save_pt_path
    os.makedirs(os.path.dirname(best_model_params_path), exist_ok=True)
    # 初始模型參數不一定最好，可以考慮在第一次驗證後儲存
    # torch.save(model.state_dict(), best_model_params_path)

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
            correct_predictions = 0
            total_predictions = 0

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
                    # In training mode, if targets are provided, returns a dictionary of losses
                    outputs = model(images, targets)

                    if phase == 'train':
                        losses = sum(loss for loss in outputs.values())
                        running_loss += losses.item() * len(images)

                        if batch_idx % 200 == 0:
                            print(f'  Batch {batch_idx:5d}: Loss = {losses.item():.4f}')

                        # backward + optimize only if in training phase
                        losses.backward()
                        optimizer.step()

                    else: # val phase, the model returns predictions
                        preds = model(images)
                        metric.update(preds, targets)

                        # Apply NMS to predictions for each image
                        nms_preds = []
                        for i in range(len(preds)):
                            pred = preds[i]
                            boxes = pred['boxes']
                            scores = pred['scores']
                            labels = pred['labels']

                            keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
                            nms_preds.append({
                                'boxes': boxes[keep_indices],
                                'labels': labels[keep_indices],
                                'scores': scores[keep_indices]
                            })

                        iou_threshold = iou_threshold_accuracy

                        for i in range(len(targets)): # Iterate through each image
                            gt_boxes = targets[i]['boxes']
                            gt_labels = targets[i]['labels']
                            pred = nms_preds[i]
                            pred_boxes = pred['boxes']
                            pred_labels = pred['labels']

                            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                                # Calculate IoU between predicted and ground truth boxes
                                iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)

                                # For each predicted box, find the GT box with the highest IoU
                                pred_to_gt_assignment = iou_matrix.max(dim=1)
                                max_iou_values = pred_to_gt_assignment.values
                                assigned_gt_indices = pred_to_gt_assignment.indices

                                for j in range(len(pred_boxes)):
                                    # If IoU is above the threshold, consider it a match
                                    if max_iou_values[j] > iou_threshold:
                                        total_predictions += 1
                                        predicted_label = pred_labels[j]
                                        ground_truth_label = gt_labels[assigned_gt_indices[j]]
                                        if predicted_label == ground_truth_label:
                                            correct_predictions += 1
                            elif len(pred_boxes) > 0 and len(gt_boxes) == 0:
                                # Predicted an object but no ground truth, count as incorrect
                                total_predictions += len(pred_boxes)
                            elif len(pred_boxes) == 0 and len(gt_boxes) > 0:
                                # Did not predict an object, not counted as incorrect classification
                                pass

                        # Visualize predictions during validation (every few batches)
                        if batch_idx % val_visualize_freq == 0:
                            image_filenames = []
                            for target in targets:
                                img_id = target['image_id'].item()
                                img_info = dataloaders['val'].dataset.data['images'][img_id]
                                image_filenames.append(img_info['file_name'])

                            gt_boxes = [target['boxes'] for target in targets]
                            gt_labels = [target['labels'] for target in targets]

                            visualize_predictions(images, nms_preds, gt_boxes=gt_boxes,
                                                  gt_labels=gt_labels,
                                                  class_names=train_dataset.class_names,
                                                  save_dir="temp_val_predictions",
                                                  epoch=epoch, batch_idx=batch_idx,
                                                  image_filenames=image_filenames)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                history['loss'].append(epoch_loss)
                writer.add_scalar('Loss/train', epoch_loss, epoch) # Log training loss
                print(f'{phase} Loss: {epoch_loss:.4f}', end='\n')
            else:
                # Compute the mAP
                metric_dict = metric.compute()
                epoch_map = metric_dict['map'].item() # Overall mAP
                history['val_loss'].append(epoch_loss)
                history['val_map'].append(epoch_map)

                if total_predictions > 0:
                    classification_accuracy = correct_predictions / total_predictions
                    print(f'Validation Classification Accuracy: {classification_accuracy:.4f}')
                    writer.add_scalar('Classification Accuracy/val', classification_accuracy, epoch)
                else:
                    print('No matching predictions found for classification accuracy calculation.')

                writer.add_scalar('mAP/val', epoch_map, epoch)      # Log validation mAP

                print(f'{phase} mAP: {epoch_map:.4f}')

                # Save best model based on mAP
                if epoch_map > best_map:
                    best_map = epoch_map
                    patience_c = 0
                    torch.save(model.state_dict(), best_model_params_path)
                else:
                    patience_c += 1


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
    DATA_DIR = 'nycu-hw2-data'
    SAVE_NAME = "resnet50_fpn_stLR3_0.1_nms0.3_b8_e20_small_obj_st0.5_ver"
    writer = SummaryWriter(f'runs/{SAVE_NAME}')

    set_seed(63)

    # ================================ Handle dataset =========================================
    # Create training and validation datasets
    train_dataset = CustomDataset(
        json_path = DATA_DIR + '/train.json',
        img_dir = DATA_DIR+'/train',
        transform = train_transform
    )
    val_dataset = CustomDataset(
        json_path = DATA_DIR + '/valid.json',
        img_dir = DATA_DIR +'/valid',
        transform = val_transform
    )
    # Set class names
    train_dataset.class_names = ["bg", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Create DataLoader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              num_workers=4, collate_fn=collate_fn_custom)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, collate_fn=collate_fn_custom)

    # Optionally print images with labels to verify dataset labels
    # print_image_with_label_for_detection(train_dataset)
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

    NUM_CLASSES = 11
    model = ftrcnn(num_classes=NUM_CLASSES)

    # Define optimizer and scheduler
    LEARNING_RATE = 0.0001 # Can try 1e-4, 3e-4 etc.
    WEIGHT_DECAY = 0.0001
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Use StepLR, you need to set step_size and gamma
    GAMMA = 0.1     # For example, multiply the learning rate by 0.1 at each step
    STEP_SIZE = 3
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    training_config = {
        'optimizer': optimizer,
        'num_epochs': 20,
        'patience': 5,
        'scheduler': scheduler,
        'save_pt_path': "./params/" + SAVE_NAME + ".pt",
        'val_visualize_freq': 200,
        'iou_threshold_accuracy': 0.5
    }

    # Start training
    print(" == Training started! ==")

    # Train the model
    trained_model, history = train_model(
        model,
        dataloaders={'train': train_loader, 'val': val_loader},
        dataset_sizes={'train': len(train_dataset), 'val': len(val_dataset)},
        config=training_config,
        writer = writer
    )

    print(" == Training completed! ==")
    writer.close()
    # ================================ Record model information =========================

    # Save model information
    trained_model_str = str(trained_model)  # Convert model to string for saving

    # Count total parameters
    params_count = sum(p.numel() for p in trained_model.parameters())
    save_model_info(params_count, trained_model_str, "./info/" + SAVE_NAME + ".txt")

    # Plot and save training history
    print("Drawing loss figure ...")
    plot_training_history(history, f"./info/{SAVE_NAME}.png")

    # Final message
    print(f"Everything is ok, you can use ./params/{SAVE_NAME}.pt to test!")
