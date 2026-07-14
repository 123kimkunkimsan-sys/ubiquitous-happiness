# 氷結晶粒径予測 - 学習・評価（VSCode用）

`colab/crystal_dataset_pipeline.ipynb` で作成した crop画像とラベルCSVを使って、
ResNet18による回帰モデルの学習・評価をローカル（VSCode）で行う。

## セットアップ

```powershell
cd vscode_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU（CUDA）を使う場合は、`requirements.txt` の torch/torchvision を入れる前に
[PyTorch公式サイト](https://pytorch.org/get-started/locally/) の案内に従って
CUDA対応版を個別にインストールしてください。

## データの配置

Colabからダウンロードした `crystal_dataset.zip` を、この `vscode_project/` 直下に
`data/` として展開してください。最終的に以下の構成になります。

```
vscode_project/
  data/
    crops/            # crop画像 (crop_000000.png, ...)
    dataset.csv        # filepath(ファイル名), label_um, ...
  config.py
  dataset.py
  model.py
  train.py
  evaluate.py
```

## 学習

```powershell
python train.py
```

- `outputs/best_model.pth` に best モデルを保存
- `outputs/train_split.csv` / `outputs/test_split.csv` に学習・評価分割を保存（評価時に再利用）

主要なハイパーパラメータはコマンドライン引数で上書きできます。

```powershell
python train.py --epochs 30 --lr 5e-5
```

## 評価

```powershell
python evaluate.py
```

`outputs/test_split.csv` と `outputs/best_model.pth` を読み込み、R²・MAE・RMSEを表示します。

## Jupyter（notebook）で実行する

`train.py` / `evaluate.py` と同じ内容を1つの notebook にまとめた `train.ipynb` を用意しています。
学習曲線や予測値の散布図もその場で確認できます。

**VSCode上で開く場合**（推奨）:

1. VSCodeに拡張機能 **Python** と **Jupyter**（Microsoft製）をインストール
2. `vscode_project/train.ipynb` を開く
3. 右上の「カーネルの選択」で、`pip install`したPython環境を選択
4. 上から順にセルを実行（▷ボタン、または「すべて実行」）

**ブラウザのJupyterで開く場合**:

```powershell
cd vscode_project
jupyter notebook train.ipynb
```

いずれの場合も、`data/crops/` + `data/dataset.csv` が `vscode_project/data/` に配置済みであることが前提です。
