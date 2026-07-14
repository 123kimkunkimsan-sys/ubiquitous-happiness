"""ResNet18 回帰モデル定義。"""
import torch.nn as nn
from torchvision.models import resnet18


def build_model(pretrained=False):
    model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
