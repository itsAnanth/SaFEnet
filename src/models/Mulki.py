import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2Classifier, self).__init__()

        # load pretrained MobileNetV2 backbone
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # extract only the feature layers (exclude classifier)
        self.features = mobilenet.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def get_mulki(num_classes=2):
    return MobileNetV2Classifier(num_classes=num_classes)