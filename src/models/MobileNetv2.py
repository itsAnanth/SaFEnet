import torch.nn as nn
from torchvision.models import (mobilenet_v2, MobileNet_V2_Weights)

def get_MobileNetV2(num_classes=2):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model