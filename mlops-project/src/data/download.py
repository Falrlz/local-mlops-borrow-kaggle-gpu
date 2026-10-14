"""
Dataset download functions.

Used by pipelines/download_pipeline.py.
"""

import shutil
import tarfile
import urllib.request
from pathlib import Path

from src.core.logger import get_logger
from src.data.cleaning import IMAGE_EXTS

logger = get_logger("mlops.download")


def is_populated(target_dir: Path) -> bool:
    """Check if directory already contains image files."""

    if not target_dir.exists():
        return False

    for path in target_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            return True
    return False


def download_file(url: str, dest_path: Path) -> None:
    """Download a remote file to dest_path."""

    logger.info(f"Downloading from [cyan]{url}[/cyan]...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, dest_path.open("wb") as file:
        shutil.copyfileobj(response, file)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded {size_mb:.1f} MB")


def extract_without_top_folder(archive_path: Path, dest_dir: Path) -> None:
    """
    Extract a .tgz archive into dest_dir, dropping its single top-level folder.

    Example: flower_photos/daisy/1.jpg -> dest_dir/daisy/1.jpg
    """

    logger.info(f"Extracting to [green]{dest_dir}[/green]...")
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            member.name = "/".join(parts[1:])
            archive.extract(member, dest_dir, filter="data")
