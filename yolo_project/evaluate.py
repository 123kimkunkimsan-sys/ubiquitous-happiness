"""結晶検出（YOLO）の評価スクリプト。学習済みモデルをvalデータで評価する。

使い方:
    python evaluate.py --weights runs/detect/crystal_yolo/weights/best.pt
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/detect/crystal_yolo/weights/best.pt")
    p.add_argument("--data", type=str, default="data/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)

    print(f"mAP50    = {metrics.box.map50:.4f}")
    print(f"mAP50-95 = {metrics.box.map:.4f}")
    print(f"Precision = {metrics.box.mp:.4f}")
    print(f"Recall    = {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
