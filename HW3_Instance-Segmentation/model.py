"""
This module defines the maskrcnn and light_maskrcnn functions for creating Mask R-CNN V2 models.
"""
import torchvision
# AnchorGenerator is still needed if you want to define the sizes/aspect_ratios variables,
# but we won't instantiate it directly when calling the builder function.
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# Import Mask R-CNN V2 specific components and weights
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# Use the V2 weights class
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
# Import MultiScaleRoIAlign
from torchvision.ops import MultiScaleRoIAlign
import torch
import torch.nn.functional as F
import torch.nn as nn
# 加在 model.backbone.body 後面，提升 pyramid feature 表達能力
# 小型 SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EnhancedFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, dropout_rate=0.2):
        """
        Args:
            in_channels_list (list[int]): [C2, C3, C4, C5] 的輸入 channel 數量
            out_channels (int): 最後輸出的 feature channel 數
            dropout_rate (float): Dropout rate，預設是 0.2
        """
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.dropout_rate = dropout_rate

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout_rate),  # Dropout加在這
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout_rate),  # Dropout再加一次
            ))
        
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout_rate),  # 出來後也丟一點
                SEBlock(out_channels)
            )
            for _ in range(len(in_channels_list))
        ])

    def forward(self, inputs):
        lateral_features = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]

        for i in range(len(lateral_features) - 1, 0, -1):
            upsample = F.interpolate(lateral_features[i], size=lateral_features[i-1].shape[-2:], mode="nearest")
            lateral_features[i-1] += upsample
        
        outputs = [output_conv(f) for output_conv, f in zip(self.output_convs, lateral_features)]
        return outputs
    
def maskrcnn_v2(num_classes):
    """
    Returns a Mask R-CNN V2 model with ResNet50-FPN backbone,
    adjusted for a specified number of output classes, with options for
    custom anchor sizes. Uses the V2 version of the model.

    Args:
        num_classes (int): The number of output classes, including the background class.
                           For digit detection (0-9), this should be 11.
    Returns:
        torch.nn.Module: The Mask R-CNN V2 model.
    """
    # Custom Anchor settings (suitable for small objects)
    # Based on previous analysis, these anchors are a good starting point for your data
    # anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    anchor_sizes = ((2,), (4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    # REMOVED: Explicit creation of anchor_generator instance
    # anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Create the Mask R-CNN V2 model with ResNet50 backbone and FPN
    # Use the maskrcnn_resnet50_fpn_v2 function
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        # Use Mask R-CNN V2 specific default weights
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=800,  # Image minimum size (using default value)
        max_size=1333, # Image maximum size (using default value)
        # Pass anchor sizes and aspect ratios directly to the builder function
        rpn_anchor_sizes=anchor_sizes,
        rpn_aspect_ratios=aspect_ratios
        # REMOVED: rpn_anchor_generator=anchor_generator
    )

    # model.fpn = EnhancedFPN(in_channels_list=[256, 512, 1024, 2048], out_channels=256)

    # --- Modify the prediction heads for the custom number of classes ---

    # Replace the Box Predictor (classification and bounding box regression)
    # Get the number of input features for the box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the default Box Predictor with a new one for our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the Mask Predictor (segmentation mask prediction)
    # Get the number of input channels for the mask predictor
    # This is typically the output channels of the Mask Head (before the final prediction layer)
    # The attribute name might be slightly different in V2, but conv5_mask is common
    try:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    except AttributeError:
        # Fallback if conv5_mask attribute is not found (check model structure)
        # You might need to inspect the model structure if this fails
        print("Warning: 'conv5_mask' not found in mask_predictor. Attempting to find input features differently.")
        # A common alternative is the first layer of the mask_predictor if it's sequential
        if isinstance(model.roi_heads.mask_predictor, torch.nn.Sequential):
             in_features_mask = model.roi_heads.mask_predictor[0].in_channels
        else:
             # If it's not sequential, you might need to inspect the specific V2 structure
             raise AttributeError("Could not find input features for mask_predictor. Please inspect the model structure.")


    # The Mask Predictor usually has a hidden layer with 256 channels
    hidden_layer = 256
    # Replace the default Mask Predictor with a new one for our number of classes
    # MaskRCNNPredictor outputs num_classes masks, one for each class (including background)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    # You can also modify RoIHeads parameters like score_thresh, nms_thresh etc. here if needed
    # model.roi_heads.score_thresh = 0.5 # Example: Adjust score threshold
    # model.roi_heads.nms_thresh = 0.3   # Example: Adjust NMS threshold
    # model.rpn.nms_thresh = 0.7
    # model.roi_heads.score_thresh = 0.05
    # model.rpn.pre_nms_top_n_test = 1000
    # model.rpn.post_nms_top_n_test = 300

    return model

    
# --- Example Usage ---
if __name__ == '__main__':
    NUM_CLASSES = 5 # Example: 4 cell types + background

    print("Creating standard Mask R-CNN V2 model:")
    standard_mask_rcnn_v2_model = maskrcnn_v2(num_classes=NUM_CLASSES)
    print(standard_mask_rcnn_v2_model)

    # Remember to use num_classes = 5 for your 4 cell types + background
    # For your actual training, use:
    # YOUR_NUM_CLASSES = 5
    # model_to_train = maskrcnn_v2(num_classes=YOUR_NUM_CLASSES)
    # or
    # model_to_train = light_maskrcnn_v2(num_classes=YOUR_NUM_CLASSES)
