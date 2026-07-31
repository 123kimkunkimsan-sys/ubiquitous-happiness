"""複数の元画像に対してYOLO+ResNetパイプラインを一括実行し、手作業CSVとの
誤差を検証してCSVにまとめる。

colab/crystal_dataset_pipeline.ipynb と同じフォルダ構成
（IMAGE_ROOTの各サブフォルダに画像、CSV_ROOTの対応するサブフォルダにCSV）を想定。

使い方:
    python validate_batch.py --image-root "path/to/画像データ" --csv-root "path/to/Excelデータ"
"""
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from resnet_utils import build_model, make_resnet_transform

COL_DIAMETER = "円相当径"
IMG_EXTS = (".bmp", ".tif", ".tiff", ".jpg", ".png")
EXCLUDE_MINUTES = {"0"}
MAG_TO_UM_PER_PIXEL = {40: 0.088725, 20: 0.17353, 10: 0.34392}
BASE_UM_PER_PIXEL = MAG_TO_UM_PER_PIXEL[40]


def normalize_folder_name(name):
    return re.sub(r"[（）()・\s]", "", name)


def get_minute_mag_key(filename):
    m = re.search(r"(\d+)分.*?(\d+)倍", filename)
    return m.groups() if m else None


def find_image_csv_pairs(image_root, csv_root):
    """crystal_dataset_pipeline.ipynb と同じロジックで画像・CSVを対応付ける。"""
    img_dirs = {normalize_folder_name(d): d
                for d in os.listdir(image_root)
                if os.path.isdir(os.path.join(image_root, d))}
    csv_dirs = {normalize_folder_name(d): d
                for d in os.listdir(csv_root)
                if os.path.isdir(os.path.join(csv_root, d))}
    common = set(img_dirs) & set(csv_dirs)
    print(f"画像フォルダ: {len(img_dirs)}件  CSVフォルダ: {len(csv_dirs)}件  対応: {len(common)}件")

    pairs = []
    for key in sorted(common):
        image_dir = os.path.join(image_root, img_dirs[key])
        csv_dir = os.path.join(csv_root, csv_dirs[key])

        images = defaultdict(list)
        for f in os.listdir(image_dir):
            if f.lower().endswith(IMG_EXTS):
                k = get_minute_mag_key(f)
                if k:
                    images[k].append(f)

        csvs = defaultdict(list)
        for f in os.listdir(csv_dir):
            if f.lower().endswith(".csv"):
                k = get_minute_mag_key(f)
                if k:
                    csvs[k].append(f)

        for k in sorted(set(images) & set(csvs)):
            minute, mag = k
            if minute in EXCLUDE_MINUTES:
                continue
            mag = int(mag)
            if mag not in MAG_TO_UM_PER_PIXEL:
                continue
            for img_f, csv_f in zip(sorted(images[k]), sorted(csvs[k])):
                pairs.append((os.path.join(image_dir, img_f), os.path.join(csv_dir, csv_f), mag))

    print(f"画像・CSVペア: {len(pairs)}件")
    return pairs


def load_gt_diameters(csv_path, magnification):
    df = pd.read_csv(csv_path, encoding="cp932")
    diameter_col = next((c for c in df.columns if COL_DIAMETER in str(c)), None)
    if diameter_col is None:
        return np.array([])
    px = df[diameter_col].dropna().astype(float)
    return (px * MAG_TO_UM_PER_PIXEL[magnification]).to_numpy()


def predict_diameters(image_path, magnification, detection_model, resnet, transform, device,
                       patch_size, overlap_ratio):
    sliced_result = get_sliced_prediction(
        str(image_path), detection_model,
        slice_height=patch_size, slice_width=patch_size,
        overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio,
        verbose=0,
    )
    orig_img = Image.open(image_path).convert("RGB")
    W, H = orig_img.size
    crop_scale = MAG_TO_UM_PER_PIXEL[magnification] / BASE_UM_PER_PIXEL

    preds = []
    with torch.no_grad():
        for pred in sliced_result.object_prediction_list:
            x1 = max(0, int(pred.bbox.minx))
            y1 = max(0, int(pred.bbox.miny))
            x2 = min(W, int(pred.bbox.maxx))
            y2 = min(H, int(pred.bbox.maxy))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = orig_img.crop((x1, y1, x2, y2))
            if crop_scale != 1.0:
                new_w = max(1, round(crop.width * crop_scale))
                new_h = max(1, round(crop.height * crop_scale))
                crop = crop.resize((new_w, new_h))
            x = transform(crop).unsqueeze(0).to(device)
            preds.append(resnet(x).item())
    return np.array(preds)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", type=str, required=True)
    p.add_argument("--csv-root", type=str, required=True)
    p.add_argument("--yolo-weights", type=str,
                   default="../yolo_project/runs/detect/crystal_yolo/weights/best.pt")
    p.add_argument("--resnet-weights", type=str,
                   default="../vscode_project/outputs/best_model.pth")
    p.add_argument("--patch-size", type=int, default=640)
    p.add_argument("--overlap-ratio", type=float, default=0.15)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output-csv", type=str, default="results/validation_results.csv")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    pairs = find_image_csv_pairs(args.image_root, args.csv_root)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.yolo_weights,
        confidence_threshold=args.conf, device=str(device),
    )
    resnet = build_model().to(device)
    resnet.load_state_dict(torch.load(args.resnet_weights, map_location=device))
    resnet.eval()
    transform = make_resnet_transform()

    rows = []
    for i, (image_path, csv_path, mag) in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {os.path.basename(image_path)} ({mag}倍)")
        try:
            gt = load_gt_diameters(csv_path, mag)
            pred = predict_diameters(image_path, mag, detection_model, resnet, transform, device,
                                      args.patch_size, args.overlap_ratio)
        except Exception as e:
            print(f"  [スキップ] {e}")
            continue

        if len(gt) == 0 or len(pred) == 0:
            print(f"  [スキップ] 手作業{len(gt)}件 / 予測{len(pred)}件")
            continue

        diff_pct = (pred.mean() / gt.mean() - 1) * 100
        count_diff_pct = (len(pred) / len(gt) - 1) * 100
        rows.append({
            "image": os.path.basename(image_path),
            "magnification": mag,
            "n_manual": len(gt),
            "n_pred": len(pred),
            "mean_manual_um": gt.mean(),
            "mean_pred_um": pred.mean(),
            "mean_diff_pct": diff_pct,
            "median_manual_um": np.median(gt),
            "median_pred_um": np.median(pred),
            "count_diff_pct": count_diff_pct,
        })
        print(f"  手作業: {len(gt)}件 平均{gt.mean():.2f}µm  /  "
              f"予測: {len(pred)}件 平均{pred.mean():.2f}µm  差{diff_pct:+.1f}%")

    result_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n保存: {out_path}")

    if len(result_df):
        print("\n=== 全体サマリー（画像単位の平均） ===")
        print(f"平均粒径の差: 平均{result_df['mean_diff_pct'].mean():+.2f}%  "
              f"標準偏差{result_df['mean_diff_pct'].std():.2f}%")
        print(f"件数の差: 平均{result_df['count_diff_pct'].mean():+.2f}%")


if __name__ == "__main__":
    main()
