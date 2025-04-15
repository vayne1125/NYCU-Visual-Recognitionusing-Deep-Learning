"""
This module defines the ftrcnn function for creating a Faster R-CNN model.
"""
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def ftrcnn(num_classes=11):
    """
    Returns a Faster R-CNN model with specified backbone,
    adjusted for a specified number of output classes, with options for
    custom anchor sizes and training all layers.

    Args:
        num_classes (int): The number of output classes, including the background class.
                             For digit detection (0-9), this should be 11.
    Returns:
        torch.nn.Module: The Faster R-CNN model.
    """
    # Custom Anchor settings (suitable for small objects)
    # anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    anchor_sizes = ((4,), (8,), (12,), (24,), (48,))
    aspect_ratios = ((1, 1.5, 2.0),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=512,  # Default: 600
        max_size=1024, # Default: 1000
        rpn_anchor_generator=anchor_generator
    )

    model.roi_heads.positive_fraction = 0.25  # Default 0.5
    model.roi_heads.batch_size_per_image = 512  # Default 512
    model.roi_heads.box_nms_thresh = 0.3  # Default 0.5
    model.roi_heads.box_score_thresh = 0.5  # Default 0.05
    model.roi_heads.box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    # Get the number of input features of the box_predictor in the ROI head
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features

    # Create a new BoxPredictor with the number of output classes matching your needs
    new_box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # Replace the pre-trained model's BoxPredictor with the new one
    model.roi_heads.box_predictor = new_box_predictor

    return model
