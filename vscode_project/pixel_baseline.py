"""単純なベースライン（外接矩形サイズ / 前景ピクセル数）だけで円相当径がどこまで説明できるかを
検証するサニティチェック。ResNetの結果と比較して、「モデルは実質的に面積を見ているだけでは
ないか」を確認する。

train_split.csv / test_split.csv が必要（train.py または train.ipynb を一度実行済みであること）。

使い方:
    python pixel_baseline.py
"""
import numpy as np
import pandas as pd
from PIL import Image

import config

# Colab側 MAG_TO_UM_PER_PIXEL[40] と同じ。crop作成時に全crop がこのスケールに正規化されている。
BASE_UM_PER_PIXEL = 0.088725

# 学習フィットに使う件数の上限（全件使うと遅いので、線形回帰の当てはめには十分な件数でサブサンプル）
MAX_FIT_SAMPLES = 5000


def bbox_diameter_um(filepath):
    """crop画像の外接矩形サイズ（幾何平均）から、円相当径っぽいスケール量を作る。
    ピクセルの中身は一切見ない。"""
    with Image.open(filepath) as img:
        w, h = img.size
    return np.sqrt(w * h) * BASE_UM_PER_PIXEL


def _otsu_threshold(arr):
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_bg = 0.0
    w_bg = 0
    best_thresh, best_var = 0, -1.0
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t
    return best_thresh


def pixelcount_diameter_um(filepath):
    """crop内をOtsu法で二値化し、面積の小さい方（=結晶とみなす）の画素数から
    円相当径を逆算する（元のラベル定義 d = sqrt(4A/pi) と同じ式）。"""
    with Image.open(filepath) as img:
        arr = np.array(img.convert("L"), dtype=np.uint8)
    t = _otsu_threshold(arr)
    fg = min(int((arr > t).sum()), int((arr <= t).sum()))
    area_um2 = fg * (BASE_UM_PER_PIXEL ** 2)
    return np.sqrt(4 * area_um2 / np.pi)


def evaluate(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(true - pred))
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    return r2, mae, rmse


def run_baseline(name, feature_fn, train_df, test_df):
    print(f"\n=== {name} ===")

    fit_df = train_df if len(train_df) <= MAX_FIT_SAMPLES else train_df.sample(
        MAX_FIT_SAMPLES, random_state=config.SEED)

    train_feat = fit_df["filepath"].apply(feature_fn).to_numpy()
    train_label = fit_df["label_um"].to_numpy()
    test_feat = test_df["filepath"].apply(feature_fn).to_numpy()
    test_label = test_df["label_um"].to_numpy()

    # 単純な線形回帰で 特徴量 -> label_um の当てはめ（スケール・オフセットの補正込み）
    a, b = np.polyfit(train_feat, train_label, 1)
    pred = a * test_feat + b

    r2, mae, rmse = evaluate(pred, test_label)
    print(f"  fit (train {len(fit_df)}件): label_um = {a:.4f} * feature + {b:.4f}")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAE  = {mae:.4f} µm")
    print(f"  RMSE = {rmse:.4f} µm")


def main():
    train_df = pd.read_csv(config.TRAIN_SPLIT_CSV)
    test_df = pd.read_csv(config.TEST_SPLIT_CSV)
    print(f"train: {len(train_df)}件 / test: {len(test_df)}件")

    run_baseline("bbox（外接矩形サイズだけ、ピクセル内容は見ない）", bbox_diameter_um, train_df, test_df)
    run_baseline("pixelcount（Otsu二値化した面積から逆算）", pixelcount_diameter_um, train_df, test_df)

    print("\n--- 参考: ResNet18 (PAD_SIZE=512, batch=32, 20epoch) ---")
    print("  R²   = 0.9988")
    print("  MAE  = 0.1551 µm")
    print("  RMSE = 0.2331 µm")


if __name__ == "__main__":
    main()
