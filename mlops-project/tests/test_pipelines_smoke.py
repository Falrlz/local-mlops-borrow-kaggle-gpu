"""
Smoke test: data prep -> train -> evaluate -> log to MLflow on a few fake
images, on CPU.

Every project path is redirected to a temporary folder, so the real data/ and
outputs/ are never touched and nothing is downloaded (pretrained: false).
"""

from pathlib import Path

import pytest
from PIL import Image

import pipelines.data_prep_pipeline as data_prep
import pipelines.evaluate_pipeline as evaluate
import pipelines.full_pipeline as full
import pipelines.log_run_pipeline as log_run
import pipelines.train_pipeline as train
from src.tracking.run_folder import read_json

DATASET_YAML = """\
url: https://example.com/unused.tgz
seed: 42
split_ratio: [70, 15, 15]
save_size: 48
overwrite: false
"""

TRAIN_YAML = """\
run_name: smoke
backbone: efficientnet_b0
pretrained: false
image_size: 32
epochs: 2
batch_size: 8
lr: 0.001
patience: 5
device: cpu
num_workers: 0
seed: 42
deterministic: true
"""

EVALUATE_YAML = """\
weights: best
batch_size: 16
"""

TRACKING_YAML = """\
use_mlflow: true
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary project: configs, fake raw images, paths redirected."""

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    (configs_dir / "dataset.yaml").write_text(DATASET_YAML, encoding="utf-8")
    (configs_dir / "train.yaml").write_text(TRAIN_YAML, encoding="utf-8")
    (configs_dir / "evaluate.yaml").write_text(EVALUATE_YAML, encoding="utf-8")
    (configs_dir / "tracking.yaml").write_text(TRACKING_YAML, encoding="utf-8")

    raw_dir = tmp_path / "data" / "raw"
    colors = {"red": (200, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 200)}
    for class_name, color in colors.items():
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(20):
            mode = "L" if index == 0 else "RGB"  # one grayscale image per class
            image = Image.new("RGB", (60 + index, 50), color).convert(mode)
            image.save(class_dir / f"{index}.jpg")
    (raw_dir / "red" / "broken.jpg").write_bytes(b"not an image")

    splits_dir = tmp_path / "data" / "splits"
    manifest_path = splits_dir / "split_manifest.csv"
    runs_dir = tmp_path / "outputs" / "runs"
    mlflow_dir = tmp_path / "outputs" / "mlflow"

    monkeypatch.setattr(data_prep, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_prep, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(data_prep, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(train, "CONFIGS_DIR", configs_dir)
    monkeypatch.setattr(train, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(train, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(train, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(train, "PRETRAINED_DIR", tmp_path / "outputs" / "pretrained")
    monkeypatch.setattr(evaluate, "SPLITS_DIR", splits_dir)
    monkeypatch.setattr(evaluate, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(log_run, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(log_run, "MLFLOW_DIR", mlflow_dir)
    return tmp_path


def test_data_prep_then_full_pipeline(project: Path) -> None:
    configs_dir = project / "configs"
    train_yaml = configs_dir / "train.yaml"

    manifest = data_prep.run_data_prep(configs_dir / "dataset.yaml")
    assert len(manifest) == 60  # broken.jpg skipped

    # Train -> evaluate -> log, as on the laptop
    run_name = full.run_full_pipeline(configs_dir)
    run_dir = project / "outputs" / "runs" / run_name
    for name in ["run_info.json", "history.csv", "weights/best.pt", "weights/last.pt"]:
        assert (run_dir / name).exists()
    assert (run_dir / "config" / "dataset.yaml").exists()

    metrics = read_json(run_dir / "eval" / "metrics.json")
    assert metrics["run_name"] == "smoke"
    for split in ["val", "test"]:
        assert 0.0 <= metrics[split]["macro_f1"] <= 1.0
        assert (run_dir / "eval" / f"confusion_matrix_{split}.png").exists()
        assert (run_dir / "eval" / f"predictions_{split}.csv").exists()

    assert (project / "outputs" / "mlflow" / "mlflow.db").exists()

    # A run is logged only once
    with pytest.raises(ValueError, match="already in MLflow"):
        log_run.run_log_run(train_yaml)


def test_data_prep_refuses_to_mix_with_old_splits(project: Path) -> None:
    config_path = project / "configs" / "dataset.yaml"
    data_prep.run_data_prep(config_path)

    with pytest.raises(FileExistsError):
        data_prep.run_data_prep(config_path)


def test_train_refuses_existing_run_name(project: Path) -> None:
    configs_dir = project / "configs"
    data_prep.run_data_prep(configs_dir / "dataset.yaml")
    (project / "outputs" / "runs" / "smoke").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        train.run_train(configs_dir / "train.yaml")


def test_full_pipeline_on_kaggle_skips_mlflow(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs_dir = project / "configs"
    data_prep.run_data_prep(configs_dir / "dataset.yaml")
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")

    run_name = full.run_full_pipeline(configs_dir)

    run_dir = project / "outputs" / "runs" / run_name
    assert read_json(run_dir / "run_info.json")["environment"]["platform"] == "kaggle"
    assert (run_dir / "eval" / "metrics.json").exists()
    assert not (project / "outputs" / "mlflow").exists()
