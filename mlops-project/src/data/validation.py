"""
Checks on the finished data/splits/ folder.

Every check returns a list of problems. An empty list means the check passed.
"""

from pathlib import Path

import pandas as pd

from src.core.paths import SPLITS


def check_counts(manifest: pd.DataFrame, splits_dir: Path) -> list[str]:
    """Each split/class folder has as many images as the manifest says."""

    problems = []
    for split in SPLITS:
        for class_name in sorted(manifest["class_name"].unique()):
            rows = manifest[
                (manifest["split"] == split) & (manifest["class_name"] == class_name)
            ]
            class_dir = splits_dir / split / class_name
            n_files = 0
            if class_dir.exists():
                n_files = len(list(class_dir.glob("*.jpg")))

            if n_files != len(rows):
                problems.append(
                    f"{split}/{class_name}: {n_files} files, manifest has {len(rows)}"
                )
    return problems


def check_every_class_in_every_split(manifest: pd.DataFrame) -> list[str]:
    """Every class has at least one image in train, val and test."""

    problems = []
    for class_name in sorted(manifest["class_name"].unique()):
        for split in SPLITS:
            rows = manifest[
                (manifest["split"] == split) & (manifest["class_name"] == class_name)
            ]
            if len(rows) == 0:
                problems.append(f"{class_name}: no images in {split}")
    return problems


def check_no_leakage(manifest: pd.DataFrame) -> list[str]:
    """No source image and no output file appears twice (no image in two splits)."""

    problems = []
    for column in ["source", "file"]:
        duplicated = manifest[manifest[column].duplicated(keep=False)]
        for value in sorted(duplicated[column].unique()):
            problems.append(f"{column} listed more than once: {value}")
    return problems
