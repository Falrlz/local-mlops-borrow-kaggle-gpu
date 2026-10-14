"""
Model factory: a torchvision backbone with a new last layer.

Used by the training and evaluation pipelines.
"""

from torch import nn
from torchvision import models


def replace_last_layer(model: nn.Module, num_classes: int) -> None:
    """
    Swap the final Linear layer for one with num_classes outputs.

    Covers the two common torchvision layouts:
    - model.fc (ResNet, RegNet)
    - model.classifier[-1] (EfficientNet, MobileNet, ConvNeXt)
    """

    if isinstance(getattr(model, "fc", None), nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
        return

    raise ValueError(
        f"Unsupported backbone {type(model).__name__}: no 'fc' or 'classifier[-1]' Linear layer"
    )


def build_model(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Build a torchvision classification model by name.

    pretrained=True downloads the default ImageNet weights once into the torch
    hub folder (the training pipeline points it at outputs/pretrained/).
    """

    weights = "DEFAULT" if pretrained else None
    model = models.get_model(backbone, weights=weights)
    replace_last_layer(model, num_classes)
    return model
