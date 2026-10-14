"""
Data preparation pipeline: data/raw/<class>/ -> data/splits/<split>/<class>/.

Run once on the laptop, after EDA. data/splits/ is what travels to Kaggle.

Run from the mlops-project/ folder:
    uv run python -m pipelines.data_prep_pipeline
"""

from pathlib import Path

import pandas as pd

from src.core.config import DatasetConfig, load_config
from src.core.logger import get_logger
from src.core.paths import CONFIGS_DIR, MANIFEST_PATH, RAW_DIR, SPLITS, SPLITS_DIR
from src.data.cleaning import (
    clear_splits,
    drop_unreadable,
    find_images,
    has_existing_files,
    save_split_images,
)
from src.data.splitting import (
    add_file_names,
    data_fingerprint,
    stratified_split,
    write_manifest,
)
from src.data.validation import (
    check_counts,
    check_every_class_in_every_split,
    check_no_leakage,
)

logger = get_logger("mlops.pipeline.data_prep")


def run_data_prep(config_path: Path = CONFIGS_DIR / "dataset.yaml") -> pd.DataFrame:
    """Build data/splits/ from data/raw/ and return the split manifest."""

    logger.info("=== Starting Data Preparation Pipeline ===")

    # Step 1: Load settings and make sure old files are not mixed in
    logger.info("Step 1: Loading configuration...")
    config = load_config(config_path, DatasetConfig)

    if has_existing_files(SPLITS_DIR):
        if not config.overwrite:
            raise FileExistsError(
                f"{SPLITS_DIR} already has files. "
                "Set 'overwrite: true' in dataset.yaml to rebuild it."
            )
        logger.info("Removing the old files in data/splits/...")
        clear_splits(SPLITS_DIR)

    # Step 2: Find the images in data/raw/<class>/
    logger.info("Step 2: Scanning data/raw/...")
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR} not found. Run pipelines.download_pipeline "
            "or put your images in data/raw/<class>/."
        )
    df = find_images(RAW_DIR)
    n_classes = df["class_name"].nunique()
    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 class folders in {RAW_DIR}, found {n_classes}"
        )
    logger.info(f"{len(df)} images in {n_classes} classes")

    # Step 3: Drop files that cannot be opened
    logger.info("Step 3: Checking that every image can be read...")
    df, dropped = drop_unreadable(df, RAW_DIR)
    for source in dropped:
        logger.warning(f"Skipped unreadable file: {source}")
    logger.info(f"{len(df)} readable images, {len(dropped)} skipped")

    # Step 4: Stratified split
    logger.info(f"Step 4: Splitting {config.split_ratio} with seed {config.seed}...")
    df = stratified_split(df, config.split_ratio, config.seed)
    df = add_file_names(df)
    log_split_table(df)

    # Step 5: Save RGB images (shorter side shrunk to save_size) and the manifest
    logger.info(f"Step 5: Saving images (shorter side <= {config.save_size} px)...")
    save_split_images(df, RAW_DIR, SPLITS_DIR, config.save_size)
    write_manifest(df, MANIFEST_PATH)

    # Step 6: Validate the result
    logger.info("Step 6: Validating data/splits/...")
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    problems = []
    problems += check_counts(manifest, SPLITS_DIR)
    problems += check_every_class_in_every_split(manifest)
    problems += check_no_leakage(manifest)

    if len(problems) > 0:
        for problem in problems[:20]:
            logger.error(problem)
        raise ValueError(f"Validation failed with {len(problems)} problem(s)")

    logger.info("All checks passed")
    logger.info(f"Data fingerprint: {data_fingerprint(MANIFEST_PATH)[:12]}")
    logger.info("=== Data Preparation Completed ===")
    return manifest


def log_split_table(df: pd.DataFrame) -> None:
    """Log the number of images per class and split."""

    table = pd.crosstab(df["class_name"], df["split"])
    table = table.reindex(columns=SPLITS, fill_value=0)
    table.loc["Total"] = table.sum()
    for name, row in table.iterrows():
        logger.info(
            f"{name:<15} train {row['train']:>5}  val {row['val']:>4}  test {row['test']:>4}"
        )


if __name__ == "__main__":
    run_data_prep()
