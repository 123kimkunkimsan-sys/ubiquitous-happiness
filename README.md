# ubiquitous-happiness

## 氷結晶粒径予測（ResNet回帰）パイプライン

顕微鏡画像から crop画像を作成し、ResNet18で円相当径（µm）を回帰予測するパイプライン。
データセット作成は Google Colab、学習・評価は VSCode（ローカル）で行う2段構成。

```
colab/
  crystal_dataset_pipeline.ipynb   # Step0-1: crop画像+CSV作成 / Step2: フィルタリング / Step3: zipエクスポート
  yolo_dataset_pipeline.ipynb      # 同じ元データからYOLO検出用データセット（パッチ分割）を作成
vscode_project/
  config.py / dataset.py / model.py / train.py / evaluate.py / pixel_baseline.py / train.ipynb
  README.md                        # セットアップ・実行手順の詳細
yolo_project/
  train.py / evaluate.py
  README.md                        # セットアップ・実行手順の詳細
```

### 使い方（ResNet回帰）

1. **Colab**: `colab/crystal_dataset_pipeline.ipynb` を開き、`IMAGE_ROOT` / `CSV_ROOT` を
   実際のGoogle Driveパスに変更して実行。crop画像とラベルCSVを作成し、`crystal_dataset.zip`
   としてダウンロードする。
2. **VSCode**: `vscode_project/` に `crystal_dataset.zip` を `data/` として展開し、
   `vscode_project/README.md` の手順に従って学習（`train.py`/`train.ipynb`）・評価（`evaluate.py`）を実行する。

## 結晶検出（YOLO）パイプライン

同じ元データ（顕微鏡画像＋矩形領域CSV）を使い、元画像を丸ごと検出するのではなく
パッチ分割してYOLOで結晶を検出するパイプライン。詳細は`yolo_project/README.md`を参照。

1. **Colab**: `colab/yolo_dataset_pipeline.ipynb` を実行し、`yolo_dataset.zip` をダウンロードする。
2. **VSCode**: `yolo_project/` に `yolo_dataset.zip` を `data/` として展開し、
   `yolo_project/README.md` の手順に従って学習（`train.py`）・評価（`evaluate.py`）を実行する。