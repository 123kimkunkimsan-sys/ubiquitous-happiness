"""結晶検出(YOLO)+粒径回帰(ResNet)を組み合わせた推論パイプライン。

元の顕微鏡画像（1枚）から、
  1. SAHIでパッチ分割してYOLO検出（重複はSAHIが自動でNMS除去）
  2. 検出した各結晶を元画像からcrop
  3. ResNetで円相当径(µm)を推定
を一気に行い、CSVと可視化画像を出力する。

使い方:
    python predict.py --image path/to/raw_image.png
    python predict.py --image path/to/raw_image.png \
        --yolo-weights ../yolo_project/runs/detect/crystal_yolo/weights/best.pt \
        --resnet-weights ../vscode_project/outputs/best_model.pth
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from resnet_utils import build_model, make_resnet_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--yolo-weights", type=str,
                   default="../yolo_project/runs/detect/crystal_yolo/weights/best.pt")
    p.add_argument("--resnet-weights", type=str,
                   default="../vscode_project/outputs/best_model.pth")
    p.add_argument("--patch-size", type=int, default=640)
    p.add_argument("--overlap-ratio", type=float, default=0.15)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--output-dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    image_path = Path(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 1: YOLO (SAHI経由でパッチ分割+推論+重複除去) =====
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.yolo_weights,
        confidence_threshold=args.conf,
        device=str(device),
    )

    sliced_result = get_sliced_prediction(
        str(image_path),
        detection_model,
        slice_height=args.patch_size,
        slice_width=args.patch_size,
        overlap_height_ratio=args.overlap_ratio,
        overlap_width_ratio=args.overlap_ratio,
    )

    orig_img = Image.open(image_path).convert("RGB")
    W, H = orig_img.size

    boxes = []
    for pred in sliced_result.object_prediction_list:
        x1 = max(0, int(pred.bbox.minx))
        y1 = max(0, int(pred.bbox.miny))
        x2 = min(W, int(pred.bbox.maxx))
        y2 = min(H, int(pred.bbox.maxy))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2, float(pred.score.value)))

    print(f"検出数: {len(boxes)}件")

    # ===== Step 2+3: 各結晶をcropしてResNetで円相当径を推定 =====
    resnet = build_model().to(device)
    resnet.load_state_dict(torch.load(args.resnet_weights, map_location=device))
    resnet.eval()
    transform = make_resnet_transform()

    records = []
    with torch.no_grad():
        for (x1, y1, x2, y2, conf) in boxes:
            crop = orig_img.crop((x1, y1, x2, y2))
            x = transform(crop).unsqueeze(0).to(device)
            pred_um = resnet(x).item()
            records.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "yolo_conf": conf, "diameter_um": pred_um,
            })

    df = pd.DataFrame(records)
    csv_path = out_dir / f"{image_path.stem}_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"保存: {csv_path}")

    if len(df):
        print(f"検出数: {len(df)}件  "
              f"平均粒径: {df['diameter_um'].mean():.2f}µm  "
              f"中央値: {df['diameter_um'].median():.2f}µm")
    else:
        print("結晶が検出されませんでした。")

    # ===== 可視化 =====
    vis = orig_img.copy()
    draw = ImageDraw.Draw(vis)
    for _, row in df.iterrows():
        draw.rectangle((row.x1, row.y1, row.x2, row.y2), outline=(255, 0, 0), width=2)
        draw.text((row.x1, max(0, row.y1 - 12)), f"{row.diameter_um:.1f}", fill=(255, 0, 0))
    vis_path = out_dir / f"{image_path.stem}_annotated.png"
    vis.save(vis_path)
    print(f"可視化画像: {vis_path}")


if __name__ == "__main__":
    main()
