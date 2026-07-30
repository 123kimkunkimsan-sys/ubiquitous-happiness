"""vscode_project の model.py / dataset.py にある前処理・モデル定義を、
推論専用に切り出したもの（学習コードには依存しない）。"""
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

PAD_SIZE = 512


def build_model():
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def pad_to_square(img, pad_size=PAD_SIZE):
    """画像を中央配置で pad_size × pad_size にパディングする。
    pad_size より大きい場合はアスペクト比を保ったまま縮小してから収める。"""
    w, h = img.size
    if w > pad_size or h > pad_size:
        scale = pad_size / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        w, h = img.size
    padded = Image.new(img.mode, (pad_size, pad_size), 0)
    padded.paste(img, ((pad_size - w) // 2, (pad_size - h) // 2))
    return padded


def make_resnet_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Lambda(pad_to_square),
        transforms.ToTensor(),
    ])
