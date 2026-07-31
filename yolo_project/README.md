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

`data/`（`images/train/`・`images/val/`・`labels/train/`・`labels/val/`・`data.yaml`）を
この `yolo_project/` 直下に用意する。作り方は2通りある。

```
yolo_project/
  data/
    images/train/, images/val/
    labels/train/, labels/val/
    data.yaml
  local_dataset_pipeline.ipynb  # ローカルでdata/を直接作成（推奨）
  train.py
  evaluate.py
  train.ipynb        # notebook版（学習・評価・予測結果の目視確認）
```

### 方法A: ローカルで直接作成（推奨）

元データ（画像・CSVフォルダ）がローカルにある場合は、`local_dataset_pipeline.ipynb`を
VSCodeで開き、`IMAGE_ROOT`/`CSV_ROOT`を実際のパスに変更して上から実行するだけでよい。
Google Driveのマウントも、zip化・ダウンロード・展開も不要で、`data/`が直接この場所に作られる。

### 方法B: Colab経由

元データがGoogle Drive上にある場合は、Colabで`colab/yolo_dataset_pipeline.ipynb`を実行して
`yolo_dataset.zip`をダウンロードし、`yolo_project/`直下に`data/`として展開する。

どちらの方法も、手作業CSVに混ざる閾値処理由来の極小ノイズ矩形（`MIN_BOX_SIZE_PX`/`MIN_BOX_AREA_PX`
未満）を除外するフィルタが入っている。これより前に作った`yolo_dataset.zip`を使っている場合は、
どちらかの方法で`data/`を作り直して再学習することを推奨する（オーバーサンプリングは適用しないこと。
過去の検証で精度が悪化することが確認済みのため）。

## （任意）大きい結晶のオーバーサンプリング

検証してみて、サイズが大きい結晶ほど検出率が低い場合は、大きい結晶を含むtrainパッチを
複製して登場頻度を上げることができる（valデータには影響しない）。

```powershell
python oversample_large.py
```

- デフォルトは100px以上の矩形を含むパッチを8倍に複製
- 再実行しても安全（前回の複製を消してから作り直す）
- 閾値・倍率は引数で変更可能: `python oversample_large.py --threshold-px 120 --factor 10`

## 学習

```powershell
python train.py
```

- デフォルトは`yolov8n.pt`（軽量・事前学習済み）からの転移学習、100エポック、imgsz=640
- 結果は `runs/detect/crystal_yolo/weights/best.pt` に保存される（`ultralytics`のデフォルトの保存先。実行後に表示される「保存先: ...」の行でも確認できる）

主要な設定はコマンドライン引数で変更できます。

```powershell
python train.py --model yolov8s.pt --epochs 150 --batch 8
```

## 評価

```powershell
python evaluate.py --weights runs/detect/crystal_yolo/weights/best.pt
```

mAP50・mAP50-95・Precision・Recallを表示します。

## Jupyter（notebook）で実行する

`train.ipynb`に学習・評価・予測結果の目視確認（検出枠を描画した画像表示）までまとめてあります。
VSCodeで開いてカーネルを選択し、上から順にセルを実行してください（`vscode_project/train.ipynb`と同じ要領）。
