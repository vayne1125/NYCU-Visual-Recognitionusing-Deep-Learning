"""
This module defines a function to build a ResNet model with customizable layers 
and versions. The function allows for selecting ResNet-50 or ResNet-101 architectures 
with different configurations for fine-tuning.

The function supports controlling the number of trainable layers and different 
architectural versions with varied fully connected layer designs.
"""
import torchvision
from torch import nn

def build_resnet(class_number, layer=50, version=1, trainable=True, trainable_layers=2):
    """
    Build a ResNet model with specified layers and configurations.

    :param class_number: Number of output classes.
    :param layer: Number of layers in the ResNet model (50 or 101).
    :param version: The version of the ResNet model architecture.
    :param trainable: Whether the model should be trainable.
    :param trainable_layers: Number of layers that are trainable.
    :return: The built ResNet model.
    """
    resnet = _get_resnet_layer(layer)

    for param in resnet.parameters():
        param.requires_grad = trainable

    if not trainable:
        # Unfreeze specific layers based on trainable_layers parameter
        trainable_names = _get_trainable_layers(trainable_layers)

        # Make specific layers trainable
        for name, param in resnet.named_parameters():
            if any(layer in name for layer in trainable_names):
                param.requires_grad = True

    # Adjust the fully connected layer according to the version
    resnet.fc = _get_fc_layer(version, resnet.fc.in_features, class_number)

    return resnet

def _get_resnet_layer(layer):
    """
    Returns the ResNet model based on the specified layer.
    """
    if layer == 50:
        return torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                )

    if layer == 101:
        return torchvision.models.resnet101(
                    weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
                )

    raise ValueError(f"Invalid layer {layer}. Choose a layer 50/101.")


def _get_trainable_layers(trainable_layers):
    """
    Returns the list of layers to be unfrozen based on the trainable_layers parameter.
    """
    trainable_names = []
    if trainable_layers >= 1:
        trainable_names.append("layer4")
    if trainable_layers >= 2:
        trainable_names.append("layer3")
    if trainable_layers >= 3:
        trainable_names.append("layer2")
    return trainable_names


def _get_fc_layer(version, num_ftrs, class_number):
    """
    Returns the appropriate fully connected layer based on the specified version.
    """
    # Modify the fully connected layer based on the version parameter
    if version == 0:
        # resnet50_w_dafault_drop0.35_b64_e100                  / 0.85 / 0.81
        # resnet50_w_dafault_drop0.5_lr0.0001_b64_e100          / 0.89 / 0.92
        # resnet101_w_imagenet1K2V_drop0.5_lr0.0001_b32_e100    / 0.88 / 0.92
        # resnet50_w_imagenet1K2V_drop0.5_lr0.0001_b64_e100_v1  / 0.89 / 0.93
        # resnet50_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100     / 0.92 / 0.93

        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, class_number)
        )

    if version == 1:
        # test1:  50, AdamW(0.0001, 0.001), cosAnneal(80, 1e-5)    [0.90 / 0.94]
        # test2:  50, AdamW(0.0001, 0.001), cosAnneal(50, 1e-5)    [0.90 / 0.93]
        # resnet101_w_imagenet1K2V_drop0.5_lr0.0001_tlr2_b64_e100   / 0.89 / 0.92
        # resnet101_w_imagenet1K2V_drop0.5_lrdynimc_tlr2_b64_e100   / 0.89 / 0.93
        # resnet101_w_imagenet1K2V_drop0.5_lrdynimc_tlr1_b64_e100   / 0.89 / 0.92
        return nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, class_number)
        )

    if version == 2:
        # resnet101_w_imagenet1K2V_drop0.5_lrdynimc_tlr2_b64_e100_v1
        return nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(250, class_number)
        )

    if version == 3:
        # resnet50_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100_v3    /  0.89  / 0.93
        # resnet50_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100_v2    /  0.90  / 0.95
        # resnet101_w_imagenet1K2V_drop0.5_lrdynimc_tlr2_b64_v2   /  0.88  / 0.94
        return nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(250, class_number)
        )

    if version == 4:
        # resnet101_w_imagenet1K2V_drop0.5_lrdynimc_tlr2_b64_e100_v4     - / 0.93
        return nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, class_number)
        )

    if version == 5:
        # resnet50_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100_v5          0.88 / 0.93
        # resnet101_w_imagenet1K2V_drop0.5_lrdynimc_tlr2_b64_e100_v3    0.89 / 0.93
        return nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, class_number)
        )

    raise ValueError(f"Invalid version {version}. Choose a version 0-5.")
