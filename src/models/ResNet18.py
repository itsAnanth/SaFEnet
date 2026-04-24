
import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=2):
    """
    Standard ResNet-18 baseline with the same backbone configuration
    (frozen trunk, unfrozen layer4) for an apples-to-apples comparison.
    """
    weights = models.ResNet18_Weights.DEFAULT
    model   = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return model

