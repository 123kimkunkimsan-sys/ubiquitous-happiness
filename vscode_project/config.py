"""ResNet 回帰モデルの設定値。"""
from pathlib import Path

# ===== パス =====
DATA_DIR    = Path("data")
CROPS_DIR   = DATA_DIR / "crops"
DATASET_CSV = DATA_DIR / "dataset.csv"

OUTPUT_DIR      = Path("outputs")
MODEL_PATH      = OUTPUT_DIR / "best_model.pth"
TRAIN_SPLIT_CSV = OUTPUT_DIR / "train_split.csv"
TEST_SPLIT_CSV  = OUTPUT_DIR / "test_split.csv"

# ===== 前処理 =====
PAD_SIZE = 512

# ===== 学習 =====
SEED             = 42
TRAIN_RATIO      = 0.8
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST  = 64
EPOCHS           = 20
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
NUM_WORKERS      = 2
