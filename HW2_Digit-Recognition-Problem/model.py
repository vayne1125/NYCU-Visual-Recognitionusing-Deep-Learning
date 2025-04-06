import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.models.detection as detection

def ftrcnn(num_classes=10, backbone_name='mobilenet_v3_large_fpn', pretrained=True):
    """
    Returns a pre-trained Faster R-CNN model with specified backbone,
    adjusted for a specified number of output classes, with backbone and RPN frozen.

    Args:
        num_classes (int): The number of output classes, including the background class.
                                        For digit detection (0-9), this should be 11.
        backbone_name (str): The name of the backbone to use ('resnet50_fpn' or 'mobilenet_v3_large_fpn').
        pretrained (bool): Whether to load pre-trained weights.

    Returns:
        torch.nn.Module: The pre-trained Faster R-CNN model with frozen backbone and RPN.
    """
    if backbone_name == 'resnet50_fpn':
        model = detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    elif backbone_name == 'mobilenet_v3_large_fpn':
        model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")

    # 凍結 backbone 的參數
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 凍結 RPN 的參數
    for param in model.rpn.parameters():
        param.requires_grad = False

    # 獲取 backbone 的輸出通道數
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 創建一個新的 BoxPredictor，其輸出類別數量符合你的需求
    new_box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 將新的 BoxPredictor 替換掉預訓練模型的
    model.roi_heads.box_predictor = new_box_predictor

    return model

# 範例使用 (如果你想使用 ResNet-50-FPN)：
# model = ftrcnn(num_classes=11, backbone_name='resnet50_fpn', pretrained=True)

# 範例使用 (如果你想繼續使用 MobileNetV3-Large-FPN)：
# model = ftrcnn(num_classes=11, backbone_name='mobilenet_v3_large_fpn', pretrained=True)


def ftrcnn2(num_classes=11, backbone_name='resnet50_fpn', pretrained=True, train_all_layers=False):
    """
    Returns a Faster R-CNN model with specified backbone,
    adjusted for a specified number of output classes, with options for
    custom anchor sizes and training all layers.

    Args:
        num_classes (int): The number of output classes, including the background class.
                                         For digit detection (0-9), this should be 11.
        backbone_name (str): The name of the backbone to use ('resnet50_fpn' or 'mobilenet_v3_large_fpn').
        pretrained (bool): Whether to load pre-trained weights.
        train_all_layers (bool): Whether to allow training of all layers (backbone and RPN included).

    Returns:
        torch.nn.Module: The Faster R-CNN model.
    """
    # 自定義 Anchor 設定 (適合小物件)
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    if backbone_name == 'resnet50_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained)
        in_features = backbone.out_channels
    elif backbone_name == 'mobilenet_v3_large_fpn':
        backbone = torchvision.models.detection.backbone_utils.mobilenet_v3_large_fpn(pretrained=pretrained)
        in_features = backbone.out_channels
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       min_size=512,  # 可以根據你的需求調整
                       max_size=1024) # 可以根據你的需求調整

    # 獲取 ROI head 中 box_predictor 的輸入特徵數
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    # 創建一個新的 BoxPredictor，其輸出類別數量符合你的需求
    new_box_predictor = FastRCNNPredictor(in_features_box, num_classes)
    # 將新的 BoxPredictor 替換掉預訓練模型的
    model.roi_heads.box_predictor = new_box_predictor

    # 決定是否凍結 backbone 和 RPN 的參數
    if not train_all_layers:
        # 凍結 backbone 的參數
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 凍結 RPN 的參數
        for param in model.rpn.parameters():
            param.requires_grad = False

    return model