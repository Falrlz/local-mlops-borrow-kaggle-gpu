"""
Fixed project paths.

Every folder used by the pipeline is defined here once, as an absolute path,
so the code works no matter which folder it is started from.
"""

from pathlib import Path

# mlops-project/ folder (this file lives in mlops-project/src/core/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Settings
CONFIGS_DIR = BASE_DIR / "configs"

# Datasets
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
MANIFEST_PATH = SPLITS_DIR / "split_manifest.csv"

# Pipeline outputs
OUTPUTS_DIR = BASE_DIR / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PRETRAINED_DIR = OUTPUTS_DIR / "pretrained"
MLFLOW_DIR = OUTPUTS_DIR / "mlflow"

# Split names used everywhere
SPLITS = ["train", "val", "test"]
