# 結晶検出＋粒径推定 統合パイプライン（YOLO + ResNet）

`yolo_project`で学習した検出モデルと、`vscode_project`で学習した回帰モデルを組み合わせて、
元の顕微鏡画像（1枚丸ごと）から、結晶ごとの円相当径（µm）を自動推定する。

```
元の顕微鏡画像
   ↓ SAHIでパッチ分割 + YOLO検出 + 重複除去(NMS)
検出された結晶の矩形座標一覧
   ↓ 各矩形を元画像からcrop
   ↓ ResNet（vscode_projectと同じ前処理）で円相当径を推定
CSV（矩形座標・確信度・推定粒径） + 可視化画像（検出枠+推定値を書き込んだ画像）
```

## セットアップ

```powershell
cd inference
pip install -r requirements.txt
```

## 使い方

```powershell
python predict.py --image path\to\raw_image.png
```

デフォルトでは、以下の場所にある学習済みモデルを使う。

- YOLO: `../yolo_project/runs/detect/crystal_yolo/weights/best.pt`
- ResNet: `../vscode_project/outputs/best_model.pth`

別のモデルを使う場合は指定できる。

```powershell
python predict.py --image path\to\raw_image.png `
    --yolo-weights ..\yolo_project\runs\detect\crystal_yolo-4\weights\best.pt `
    --resnet-weights ..\vscode_project\outputs\best_model.pth
```

## 出力

`results/`フォルダに以下が作られる（`--output-dir`で変更可）。

- `<画像名>_predictions.csv`: 検出した結晶ごとの矩形座標・YOLOの確信度・推定粒径(µm)
- `<画像名>_annotated.png`: 検出枠と推定粒径を元画像に書き込んだ可視化画像

実行時、検出数・平均粒径・中央値もコンソールに表示される。

## 主なオプション

| 引数 | デフォルト | 意味 |
|---|---|---|
| `--patch-size` | 640 | YOLO推論時のパッチサイズ |
| `--overlap-ratio` | 0.15 | パッチの重なり率 |
| `--conf` | 0.25 | YOLOの検出確信度しきい値 |
| `--output-dir` | `results` | 出力先フォルダ |

## 複数画像での一括検証（validate_batch.py）

`colab/crystal_dataset_pipeline.ipynb` と同じフォルダ構成（`IMAGE_ROOT`の各サブフォルダに画像、
`CSV_ROOT`の対応するサブフォルダに手作業計測CSV）をそのまま渡して、複数の画像に対して
YOLO+ResNetパイプラインを一括実行し、手作業CSVとの誤差をまとめてCSVに出力する。

```powershell
python validate_batch.py --image-root "path\to\画像データ" --csv-root "path\to\Excelデータ"
```

`results/validation_results.csv`（`--output-csv`で変更可）に、画像ごとの以下の情報が1行ずつ出力される。

| 列 | 意味 |
|---|---|
| `image` | 画像ファイル名 |
| `magnification` | 倍率 |
| `n_manual` / `n_pred` | 手作業/自動パイプラインの検出数 |
| `mean_manual_um` / `mean_pred_um` | 平均粒径(µm) |
| `mean_diff_pct` | 平均粒径の差（%） |
| `median_manual_um` / `median_pred_um` | 中央値(µm) |
| `count_diff_pct` | 検出数の差（%） |

実行後、全画像をまとめた平均誤差率・件数の差もコンソールに表示される。「0分」データは
crop用パイプラインと同様に自動的に除外される。
