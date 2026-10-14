"""
Stratified train/val/test split and the split manifest.

The manifest (split_manifest.csv) records which image went to which split.
Its hash is the data fingerprint stored with every training run.

Used by pipelines/data_prep_pipeline.py.
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

MANIFEST_COLUMNS = ["split", "class_name", "file", "source"]


def stratified_split(
    df: pd.DataFrame, split_ratio: tuple[int, int, int], seed: int
) -> pd.DataFrame:
    """Add a 'split' column (train/val/test) with the same class mix in every split."""

    train_part, val_part, test_part = split_ratio
    total = train_part + val_part + test_part

    # First cut: train vs the rest. Second cut: the rest into val and test.
    train_df, rest_df = train_test_split(
        df,
        test_size=(val_part + test_part) / total,
        stratify=df["class_name"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        rest_df,
        test_size=test_part / (val_part + test_part),
        stratify=rest_df["class_name"],
        random_state=seed,
    )

    parts = [
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ]
    return pd.concat(parts).reset_index(drop=True)


def add_file_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'file' column: <split>/<class>/<name>.jpg, relative to data/splits/."""

    files = []
    for split, class_name, source in zip(
        df["split"], df["class_name"], df["source"], strict=True
    ):
        files.append(f"{split}/{class_name}/{Path(source).stem}.jpg")

    return df.assign(file=files)


def write_manifest(df: pd.DataFrame, manifest_path: Path) -> None:
    """Write split_manifest.csv (relative paths only, '\\n' line endings)."""

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(manifest_path, index=False, lineterminator="\n")


def data_fingerprint(manifest_path: Path) -> str:
    """Return the SHA-256 of split_manifest.csv (same splits -> same fingerprint)."""

    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
