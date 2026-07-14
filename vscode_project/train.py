"""学習スクリプト。

使い方:
    python train.py
    python train.py --epochs 30 --lr 5e-5
"""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from dataset import CrystalDataset, load_valid_dataset, split_dataset, test_transform, train_transform
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-csv", type=str, default=str(config.DATASET_CSV))
    p.add_argument("--model-path", type=str, default=str(config.MODEL_PATH))
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--batch-size-train", type=int, default=config.BATCH_SIZE_TRAIN)
    p.add_argument("--batch-size-test", type=int, default=config.BATCH_SIZE_TEST)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return p.parse_args()


def main():
    args = parse_args()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    df_all = load_valid_dataset(args.dataset_csv)
    train_df, test_df = split_dataset(df_all)
    train_df.to_csv(config.TRAIN_SPLIT_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(config.TEST_SPLIT_CSV, index=False, encoding="utf-8-sig")

    train_loader = DataLoader(CrystalDataset(train_df, train_transform),
                               batch_size=args.batch_size_train, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(CrystalDataset(test_df, test_transform),
                              batch_size=args.batch_size_test, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    model = build_model().to(device)
    criterion = nn.HuberLoss(delta=1.0)
    mse_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_mse = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_mse = total_loss / len(train_df)

        model.eval()
        t_mse = t_mae = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                out = model(x)
                t_mse += mse_fn(out, y).item() * x.size(0)
                t_mae += (out - y).abs().sum().item()
        test_mse = t_mse / len(test_df)
        test_mae = t_mae / len(test_df)

        scheduler.step(test_mse)
        tag = ""
        if test_mse < best_mse:
            best_mse = test_mse
            torch.save(model.state_dict(), args.model_path)
            tag = "  ← best 保存"
        print(f"Epoch {ep:2d}: train={train_mse:.4f}  test={test_mse:.4f}  MAE={test_mae:.4f} µm{tag}")

    print(f"\n学習完了。best test MSE = {best_mse:.4f}")


if __name__ == "__main__":
    main()
