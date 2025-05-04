"""
This module defines the maskrcnn functions for creating Mask R-CNN V2 models.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.ops import MultiScaleRoIAlign

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
    # anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    # Create the Mask R-CNN V2 model with ResNet50 backbone and FPN
    # Use the maskrcnn_resnet50_fpn_v2 function
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        # Use Mask R-CNN V2 specific default weights
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=800,  # Image minimum size (using default value)
        max_size=1333,  # Image maximum size (using default value)
        rpn_anchor_sizes=anchor_sizes,
        rpn_aspect_ratios=aspect_ratios
    )

    # --- Modify the prediction heads for the custom number of classes ---

    # Replace the Box Predictor (classification and bounding box regression)
    # Get the number of input features for the box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the default Box Predictor with a new one for our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box, num_classes)

    # Replace the Mask Predictor (segmentation mask prediction)
    # Get the number of input channels for the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # The Mask Predictor usually has a hidden layer with 256 channels
    hidden_layer = 256
    # Replace the default Mask Predictor with a new one for our number of classes
    # MaskRCNNPredictor outputs num_classes masks, one for each class
    # (including background)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    # You can also modify RoIHeads parameters like score_thresh, nms_thresh etc.
    # model.rpn.nms_thresh = 0.7
    # model.roi_heads.score_thresh = 0.05
    # model.rpn.pre_nms_top_n_test = 1000
    # model.rpn.post_nms_top_n_test = 300

    return model


def light_maskrcnn_v2(num_classes):
    """
    Returns a Mask R-CNN V2 model with ResNet50-FPN backbone,
    adjusted for a specified number of output classes, with modifications
    for potential performance or efficiency improvements (light version).

    Args:
        num_classes (int): The number of output classes, including the background class.
                           For example, 5 for 4 cell types + background.
    Returns:
        torch.nn.Module: The modified Mask R-CNN V2 model.
    """
    anchor_sizes = ((4,), (8,), (16,), (32,), (64,)) # Example sizes
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) # Example aspect ratios

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        min_size=800,  # Image minimum size (using default value from maskrcnn_v2)
        max_size=1333, # Image maximum size (using default value from maskrcnn_v2)
        rpn_anchor_sizes=anchor_sizes,
        rpn_aspect_ratios=aspect_ratios
    )

    model.roi_heads.box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1'],  # Reduced number of feature maps for Box Head (P2, P3)
        output_size=7,             # Output size for Box RoI Pool (standard)
        sampling_ratio=2           # Sampling ratio for Box RoI Pool (standard)
    )

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# --- Example Usage ---
if __name__ == '__main__':
    NUM_CLASSES = 5  # Example: 4 cell types + background

    print("Creating standard Mask R-CNN V2 model:")
    standard_mask_rcnn_v2_model = maskrcnn_v2(num_classes=NUM_CLASSES)
    print(standard_mask_rcnn_v2_model)
