"""
Configuration schemas and loader.

Each YAML file in configs/ has one Pydantic class here. Loading a YAML file
through its class checks the types right away, so a typo in a key or a wrong
type fails before any work starts.
"""

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base class that rejects unknown keys (catches misspelled settings)."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# configs/dataset.yaml
# ---------------------------------------------------------------------------
class DatasetConfig(StrictModel):
    """Settings for the dataset download."""

    url: str = Field(min_length=1, description="Download URL of the .tgz archive")


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
