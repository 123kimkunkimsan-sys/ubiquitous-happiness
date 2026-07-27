"""結晶検出（YOLO）の学習スクリプト。

colab/yolo_dataset_pipeline.ipynb でエクスポートしたzipを data/ に展開してから実行する。

使い方:
    python train.py
    python train.py --model yolov8s.pt --epochs 100 --imgsz 640
"""
import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/data.yaml")
    p.add_argument("--model", type=str, default="yolov8n.pt",
                   help="事前学習済みの重み。n(nano)/s(small)/m(medium)... 軽い方から試す")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="outputs")
    p.add_argument("--name", type=str, default="crystal_yolo")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
