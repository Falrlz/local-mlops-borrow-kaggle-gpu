"""Tests for config loading and the model factory."""

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from src.core.config import DatasetConfig, load_config
from src.core.paths import CONFIGS_DIR
from src.models.factory import build_model


def test_project_configs_load() -> None:
    config = load_config(CONFIGS_DIR / "dataset.yaml", DatasetConfig)
    assert sum(config.split_ratio) == 100


def test_misspelled_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "url: x\nseed: 1\nsplit_ratio: [70, 15, 15]\nsave_size: 224\n"
        "overwrite: false\nsave_sise: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(config_path, DatasetConfig)


@pytest.mark.parametrize("backbone", ["efficientnet_b0", "resnet18"])
def test_build_model_output_matches_class_count(backbone: str) -> None:
    model = build_model(backbone, num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 5)
