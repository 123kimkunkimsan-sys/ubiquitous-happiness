# 結晶検出（YOLO） - 学習・評価（VSCode用）

`colab/yolo_dataset_pipeline.ipynb` で作成したパッチ+YOLOラベルのデータセットを使って、
`ultralytics`のYOLOで結晶検出モデルを学習・評価する。

## セットアップ

```powershell
cd yolo_project
pip install -r requirements.txt
```

GPU（CUDA）を使う場合は、`torch`がCUDA対応版になっているか確認してください
（`vscode_project`側ですでにCUDA対応のtorchを入れていれば、同じ環境で流用できます）。

## データの配置

Colabからダウンロードした `yolo_dataset.zip` を、この `yolo_project/` 直下に
`data/` として展開してください。最終的に以下の構成になります。

```
yolo_project/
  data/
    images/train/, images/val/
    labels/train/, labels/val/
    data.yaml
  train.py
  evaluate.py
```

## 学習

```powershell
python train.py
```

- デフォルトは`yolov8n.pt`（軽量・事前学習済み）からの転移学習、100エポック、imgsz=640
- 結果は `outputs/crystal_yolo/weights/best.pt` に保存される

主要な設定はコマンドライン引数で変更できます。

```powershell
python train.py --model yolov8s.pt --epochs 150 --batch 8
```

## 評価

```powershell
python evaluate.py --weights outputs/crystal_yolo/weights/best.pt
```

mAP50・mAP50-95・Precision・Recallを表示します。
