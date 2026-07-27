"""大きい結晶を含むtrainパッチを複製し、学習時の登場頻度を上げる。

検出率がサイズ帯によって偏っている（大きい結晶ほど検出率が低い）場合の対策。
val（検証用）データには一切手を加えない（評価が歪むため）。

再実行しても安全（前回作った複製ファイルを削除してから作り直す）。

使い方:
    python oversample_large.py
    python oversample_large.py --threshold-px 120 --factor 10
"""
import argparse
import shutil
from pathlib import Path

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--threshold-px", type=float, default=100,
                   help="この値(px)以上の矩形を含むパッチを「大きい」とみなす")
    p.add_argument("--factor", type=int, default=8,
                   help="該当パッチを何倍に増やすか（1なら複製なし）")
    return p.parse_args()


def has_large_box(label_path, img_w, img_h, threshold_px):
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            size_px = max(bw * img_w, bh * img_h)
            if size_px >= threshold_px:
                return True
    return False


def main():
    args = parse_args()
    train_img_dir = Path(args.data_dir) / "images" / "train"
    train_lbl_dir = Path(args.data_dir) / "labels" / "train"

    # 再実行しても安全なように、前回作った複製ファイルを先に削除
    removed = 0
    for p in list(train_img_dir.glob("*_dup*.png")) + list(train_lbl_dir.glob("*_dup*.txt")):
        p.unlink()
        removed += 1
    if removed:
        print(f"前回の複製ファイルを削除: {removed}件")

    original_images = sorted(train_img_dir.glob("*.png"))
    targets = []
    for img_path in original_images:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        if has_large_box(lbl_path, w, h, args.threshold_px):
            targets.append(img_path.stem)

    print(f"元のtrainパッチ数: {len(original_images)}件")
    print(f"大きい結晶(>={args.threshold_px:.0f}px)を含むパッチ: {len(targets)}件")

    for stem in targets:
        img_path = train_img_dir / f"{stem}.png"
        lbl_path = train_lbl_dir / f"{stem}.txt"
        for i in range(1, args.factor):
            shutil.copy(img_path, train_img_dir / f"{stem}_dup{i}.png")
            shutil.copy(lbl_path, train_lbl_dir / f"{stem}_dup{i}.txt")

    print(f"複製後のtrainパッチ数: {len(list(train_img_dir.glob('*.png')))}件")
    print("\n※ valデータは変更していません（評価の公平性のため）")


if __name__ == "__main__":
    main()
