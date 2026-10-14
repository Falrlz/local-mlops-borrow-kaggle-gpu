"""
Download pipeline: fetch the example dataset into data/raw/<class>/.

Example only: users with their own images skip this and put them in
data/raw/<class>/ directly.

Run from the mlops-project/ folder:
    uv run python -m pipelines.download_pipeline
"""

from pathlib import Path

from src.core.config import DatasetConfig, load_config
from src.core.logger import get_logger
from src.core.paths import CONFIGS_DIR, DATA_DIR, RAW_DIR
from src.data.download import download_file, extract_without_top_folder, is_populated

logger = get_logger("mlops.pipeline.download")


def run_download(config_path: Path = CONFIGS_DIR / "dataset.yaml") -> str:
    """
    Download the dataset in configs/dataset.yaml into data/raw/.

    Returns the status: READY (already on disk) or DOWNLOADED.
    """
    logger.info("=== Starting Download Pipeline ===")

    # Step 1: Load settings
    logger.info("Step 1: Loading configuration...")
    config = load_config(config_path, DatasetConfig)

    # Step 2: Skip if images are already there
    logger.info("Step 2: Checking data/raw/...")
    if is_populated(RAW_DIR):
        logger.info(f"{RAW_DIR} already has images, nothing to download.")
        logger.info("=== Download Pipeline Completed ===")
        return "READY"

    # Step 3: Download and extract
    logger.info("Step 3: Downloading and extracting...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / Path(config.url).name
    download_file(config.url, archive_path)
    extract_without_top_folder(archive_path, RAW_DIR)
    archive_path.unlink()

    logger.info("=== Download Pipeline Completed ===")
    return "DOWNLOADED"


if __name__ == "__main__":
    run_download()
