"""
Values shared by pack.py and unpack.py.

Standard library only: the bridge needs no environment of its own and runs
with any Python 3.11+ (from mlops-project/: `uv run python ../kaggle-bridge/pack.py`).
"""

import logging
import re
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parent
REPO_DIR = BRIDGE_DIR.parent

# The project that is sent to Kaggle (the only place its folder name is written)
PROJECT_NAME = "mlops-project"
PROJECT_DIR = REPO_DIR / PROJECT_NAME

UPLOAD_DIR = BRIDGE_DIR / "upload-to-kaggle"
DOWNLOAD_DIR = BRIDGE_DIR / "download-from-kaggle"

TRAIN_CONFIG_PATH = PROJECT_DIR / "configs" / "train.yaml"
SPLITS_DIR = PROJECT_DIR / "data" / "splits"
PRETRAINED_DIR = PROJECT_DIR / "outputs" / "pretrained"
RUNS_DIR = PROJECT_DIR / "outputs" / "runs"

# Same name as GIT_INFO_PATH in mlops-project/src/core/paths.py
GIT_INFO_FILE = "git_info.json"

# A run folder from Kaggle must hold at least these files
RUN_REQUIRED_FILES = [
    "run_info.json",
    "history.csv",
    "weights/best.pt",
    "eval/metrics.json",
]


def get_logger(name: str) -> logging.Logger:
    """A plain console logger (no third-party packages)."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    return logging.getLogger(name)


def read_run_name(train_config_path: Path = TRAIN_CONFIG_PATH) -> str:
    """Read `run_name:` from train.yaml without a YAML library."""

    text = train_config_path.read_text(encoding="utf-8")
    match = re.search(r"^run_name:\s*['\"]?([^'\"#\s]+)", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"No 'run_name:' line found in {train_config_path}")
    return match.group(1)
