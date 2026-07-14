"""評価スクリプト（R²・MAE・RMSE）。train.py が保存した test_split.csv を使う。

使い方:
    python evaluate.py
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from dataset import CrystalDataset, test_transform
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default=str(config.MODEL_PATH))
    p.add_argument("--test-split-csv", type=str, default=str(config.TEST_SPLIT_CSV))
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE_TEST)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    test_df = pd.read_csv(args.test_split_csv)
    test_loader = DataLoader(CrystalDataset(test_df, test_transform),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_model().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device)).squeeze(1).cpu().numpy()
            preds.extend(out.tolist())
            trues.extend(y.numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)

    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(trues - preds))
    rmse = np.sqrt(np.mean((trues - preds) ** 2))

    print(f"R²   = {r2:.4f}")
    print(f"MAE  = {mae:.4f} µm")
    print(f"RMSE = {rmse:.4f} µm")


if __name__ == "__main__":
    main()
