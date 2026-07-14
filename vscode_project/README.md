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
