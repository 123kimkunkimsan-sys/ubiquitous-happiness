"""crop画像データセットの読み込みと前処理。"""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CROPS_DIR, PAD_SIZE, SEED, TRAIN_RATIO


def pad_to_square(img, pad_size=PAD_SIZE):
    w, h = img.size
    if w > pad_size:
        left = (w - pad_size) // 2
        img = img.crop((left, 0, left + pad_size, h))
        w = pad_size
    if h > pad_size:
        top = (h - pad_size) // 2
        img = img.crop((0, top, w, top + pad_size))
        h = pad_size
    padded = Image.new(img.mode, (pad_size, pad_size), 0)
    padded.paste(img, ((pad_size - w) // 2, (pad_size - h) // 2))
    return padded


train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Lambda(pad_to_square),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Lambda(pad_to_square),
    transforms.ToTensor(),
])


class CrystalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(row["label_um"], dtype=torch.float32)


def load_valid_dataset(csv_path, crops_dir=CROPS_DIR):
    """CSVを読み込み、破損画像を除いた有効データのみを返す。

    CSVの``filepath``は crop 画像のファイル名（相対）を想定し、``crops_dir``と
    結合して実パスに解決する。すでに絶対パスの場合はそのまま使う。
    """
    raw_df = pd.read_csv(csv_path)
    raw_df["filepath"] = raw_df["filepath"].apply(
        lambda p: p if Path(p).is_absolute() else str(Path(crops_dir) / p)
    )
    valid_idx = []
    for i, fp in enumerate(raw_df["filepath"]):
        try:
            with Image.open(fp) as im:
                im.verify()
            valid_idx.append(i)
        except Exception:
            pass
    removed = len(raw_df) - len(valid_idx)
    if removed:
        print(f"壊れた画像を除外: {removed}件")
    df_all = raw_df.iloc[valid_idx].reset_index(drop=True)
    print(f"有効データ: {len(df_all)}件")
    return df_all


def split_dataset(df_all, train_ratio=TRAIN_RATIO, seed=SEED):
    """再現可能な8:2分割。"""
    torch.manual_seed(seed)
    perm = torch.randperm(len(df_all)).tolist()
    n_train = int(len(df_all) * train_ratio)
    train_df = df_all.iloc[perm[:n_train]].reset_index(drop=True)
    test_df = df_all.iloc[perm[n_train:]].reset_index(drop=True)
    print(f"train: {len(train_df)}件 / test: {len(test_df)}件")
    return train_df, test_df
