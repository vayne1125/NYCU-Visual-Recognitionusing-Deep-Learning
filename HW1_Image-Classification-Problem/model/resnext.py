"""
This module provides functions for building and configuring ResNeXt models
with various layers and fully connected layers for classification tasks.
It supports ResNeXt-50 and ResNeXt-101 architectures, allowing customization
of the fully connected layer based on the desired version.
"""

import torchvision
from torch import nn

def build_resnext(class_number, layer=50, version=0, trainable=True):
    """
    Builds a ResNeXt model with the specified parameters.

    Args:
        class_number (int): The number of output classes.
        layer (int): The ResNeXt architecture to use. Can be 50 or 101.
        version (int): The version of the fully connected layers. Range from 0 to 5.
        trainable (bool): If True, all layers are trainable. If False, no layers are trainable.

    Returns:
        nn.Module: A ResNeXt model with the specified configuration.
    """

    # Get the ResNeXt model based on the specified layer
    resnext = _get_resnext_layer(layer)

    # Set all parameters to be trainable or not based on the `trainable` flag
    for param in resnext.parameters():
        param.requires_grad = trainable

    # Get the number of input features to the fully connected layer
    num_ftrs = resnext.fc.in_features

    # Adjust the fully connected layer based on the version
    resnext.fc = _get_fc_layer(version, num_ftrs, class_number)

    return resnext

def _get_resnext_layer(layer):
    """
    Returns the ResNeXt model based on the specified layer.

    Args:
        layer (int): The ResNeXt architecture to use (50 or 101).

    Returns:
        nn.Module: A ResNeXt model.
    """
    if layer == 50:
        return torchvision.models.resnext50_32x4d(
                    weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
                )

    if layer == 101:
        return torchvision.models.resnext101_32x8d(
                    weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
                )

    raise ValueError(f"Invalid layer {layer}. Choose a layer 50/101.")

def _get_fc_layer(version, num_ftrs, class_number):
    """
    Returns the fully connected layer for the ResNeXt model based on the version.

    Args:
        version (int): The version of the fully connected layer.
        num_ftrs (int): The number of input features to the fully connected layer.
        class_number (int): The number of output classes.

    Returns:
        nn.Sequential: The fully connected layer.
    """
    if version == 0:
        # resneXt50_w_imagenet1K2V_drop0.5_lrdynimc_b32_e100_v4            0.92 / 0.94
        # resneXt101_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100              0.92 /
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, class_number)
        )

    if version == 1:
        # resneXt50_w_imagenet1K2V_drop0.5_lrdynimc_b32_e100_v1            0.92 / 0.94    
        # resneXt50_w_imagenet1K2V_drop0.25_lrdynimc_b32_e100 d = 0.25     0.91 / 0.94
        # resneXt101_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100              0.93 / 0.95
        return nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, class_number)
        )

    if version == 2:
        # resneXt50_w_imagenet1K2V_drop0.5_lrdynimc_b32_e100_v6             /  
        return nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(250, class_number)
        )

    if version == 3:
        # resneXt50_w_imagenet1K2V_drop0.5_lrdynimc_b32_e100_v3    0.91 / 0.94
        # resneXt101_w_imagenet1K2V_drop0.5_lrdynimc_b64_e100_v1   0.93 / 0.94
        return nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.BatchNorm1d(500),  # Add BatchNorm
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(250, class_number)
        )
    
    raise ValueError(f"Invalid version {version}. Choose a version 0-3.")
