"""
The run folder contract: the only module that knows its layout.

Every training run writes one self-contained folder. It is the single source
of truth for the run; MLflow is only an index built from these folders.

    outputs/runs/<run_name>/
    ├── run_info.json     run name, model, classes, data fingerprint, environment
    ├── config/           copy of the YAML files used
    ├── history.csv       loss and metrics per epoch
    ├── weights/          best.pt, last.pt (state_dict only)
    └── eval/             metrics.json, confusion_matrix_<split>.png, predictions_<split>.csv

Only relative information is stored, so a folder trained on Kaggle works on
the laptop unchanged.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import torch

RUN_INFO_FILE = "run_info.json"
HISTORY_FILE = "history.csv"
CONFIG_DIR = "config"
WEIGHTS_DIR = "weights"
EVAL_DIR = "eval"
METRICS_FILE = "metrics.json"


def create_run_folder(run_dir: Path) -> None:
    """Create an empty run folder; fail if the run name is already used."""

    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists. Use a new 'run_name'.")
    run_dir.mkdir(parents=True)


def copy_configs(run_dir: Path, config_paths: list[Path]) -> None:
    """Copy the YAML files used by the run into run_dir/config/."""

    config_dir = run_dir / CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    for config_path in config_paths:
        shutil.copy2(config_path, config_dir / config_path.name)


def write_json(path: Path, data: dict) -> None:
    """Write a dict as indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def read_json(path: Path) -> dict:
    """Read a JSON file into a dict."""

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_run_info(run_dir: Path, run_info: dict) -> None:
    """Write run_info.json."""

    write_json(run_dir / RUN_INFO_FILE, run_info)


def read_run_info(run_dir: Path) -> dict:
    """Read run_info.json; fail with a clear message if the run is missing."""

    path = run_dir / RUN_INFO_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Is '{run_dir.name}' a finished run?"
        )
    return read_json(path)


def write_history(run_dir: Path, history: list[dict]) -> None:
    """Write history.csv (one row per epoch)."""

    pd.DataFrame(history).to_csv(run_dir / HISTORY_FILE, index=False)


def weights_path(run_dir: Path, name: str) -> Path:
    """Path of weights/<name>.pt (name is 'best' or 'last')."""

    return run_dir / WEIGHTS_DIR / f"{name}.pt"


def save_weights(run_dir: Path, name: str, state_dict: dict) -> None:
    """Save a state_dict as weights/<name>.pt."""

    path = weights_path(run_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_weights(run_dir: Path, name: str) -> dict:
    """Load weights/<name>.pt (tensors only, safe to load)."""

    return torch.load(
        weights_path(run_dir, name), map_location="cpu", weights_only=True
    )


def eval_dir(run_dir: Path) -> Path:
    """Path of the eval/ folder."""

    return run_dir / EVAL_DIR


def write_metrics(run_dir: Path, metrics: dict) -> None:
    """Write eval/metrics.json."""

    write_json(eval_dir(run_dir) / METRICS_FILE, metrics)


def read_metrics(run_dir: Path) -> dict:
    """Read eval/metrics.json; fail with a clear message if not evaluated yet."""

    path = eval_dir(run_dir) / METRICS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run pipelines.evaluate_pipeline first."
        )
    return read_json(path)


def read_history(run_dir: Path) -> pd.DataFrame:
    """Read history.csv."""

    return pd.read_csv(run_dir / HISTORY_FILE)


def run_config_path(run_dir: Path, file_name: str) -> Path:
    """Path of the copy of a YAML file in config/ (e.g. 'train.yaml')."""

    return run_dir / CONFIG_DIR / file_name
