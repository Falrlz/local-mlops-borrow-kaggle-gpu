"""
Configuration schemas and loader.

Each YAML file in configs/ has one Pydantic class here. Loading a YAML file
through its class checks the types right away, so a typo in a key or a wrong
type fails before any work starts.
"""

from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class StrictModel(BaseModel):
    """Base class that rejects unknown keys (catches misspelled settings)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# configs/dataset.yaml
# ---------------------------------------------------------------------------
class DatasetConfig(StrictModel):
    """Settings for the download and data preparation pipelines."""

    url: str = Field(min_length=1, description="Download URL of the .tgz archive")
    seed: int
    split_ratio: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        description="train:val:test ratio"
    )
    save_size: int = Field(gt=0, description="Shorter side in data/splits/")
    overwrite: bool = Field(description="Rebuild data/splits/ if it has files")


# ---------------------------------------------------------------------------
# configs/train.yaml
# ---------------------------------------------------------------------------
class TrainConfig(StrictModel):
    """Settings for the training pipeline."""

    run_name: str = Field(min_length=1)
    backbone: str = Field(min_length=1, description="torchvision model name")
    pretrained: bool
    image_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    patience: int = Field(gt=0)
    device: Literal["auto", "cpu", "cuda"]
    num_workers: int = Field(ge=0)
    seed: int
    deterministic: bool


# ---------------------------------------------------------------------------
# configs/evaluate.yaml
# ---------------------------------------------------------------------------
class EvaluateConfig(StrictModel):
    """Settings for the evaluation pipeline (run_name comes from train.yaml)."""

    weights: Literal["best", "last"]
    batch_size: int = Field(gt=0)


# ---------------------------------------------------------------------------
# configs/tracking.yaml
# ---------------------------------------------------------------------------
class TrackingConfig(StrictModel):
    """Settings for logging to MLflow (run_name comes from train.yaml)."""

    use_mlflow: bool = Field(description="Log runs to MLflow")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(config_path: Path, schema: type[ConfigT]) -> ConfigT:
    """Read a YAML file and validate it against a schema class."""

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return schema.model_validate(data)
