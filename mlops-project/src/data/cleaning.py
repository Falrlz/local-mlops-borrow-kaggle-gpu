"""
Image cleaning functions: find images, drop unreadable ones, save RGB copies.

Used by pipelines/data_prep_pipeline.py.
"""

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from rich.progress import track

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_class_dirs(raw_dir: Path) -> list[Path]:
    """Return the class folders in raw_dir, sorted by name."""

    class_dirs = []
    for entry in sorted(raw_dir.iterdir()):
        if entry.is_dir():
            class_dirs.append(entry)
    return class_dirs


def find_images(raw_dir: Path) -> pd.DataFrame:
    """
    List every image file under raw_dir/<class>/.

    Returns one row per image: class_name and source (path relative to raw_dir).
    Files outside class folders (e.g. LICENSE.txt) are ignored.
    """

    rows = []
    for class_dir in list_class_dirs(raw_dir):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                rows.append(
                    {
                        "class_name": class_dir.name,
                        "source": path.relative_to(raw_dir).as_posix(),
                    }
                )
    return pd.DataFrame(rows, columns=["class_name", "source"])


def is_readable(image_path: Path) -> bool:
    """Check that an image file can be fully decoded."""

    try:
        with Image.open(image_path) as image:
            image.load()
    except (UnidentifiedImageError, OSError):
        return False
    return True


def drop_unreadable(df: pd.DataFrame, raw_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return the readable rows and the list of sources that could not be read."""

    keep = []
    dropped = []
    for source in track(df["source"], description="Checking images"):
        readable = is_readable(raw_dir / source)
        keep.append(readable)
        if not readable:
            dropped.append(source)

    return df[keep].reset_index(drop=True), dropped


def shrink_to_short_side(image: Image.Image, size: int) -> Image.Image:
    """Shrink so the shorter side equals size; smaller images are left as they are."""

    width, height = image.size
    short_side = min(width, height)
    if short_side <= size:
        return image

    scale = size / short_side
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def has_existing_files(splits_dir: Path) -> bool:
    """Check if data/splits/ already contains any file."""

    if not splits_dir.exists():
        return False

    for path in splits_dir.rglob("*"):
        if path.is_file():
            return True
    return False


def clear_splits(splits_dir: Path) -> None:
    """Delete data/splits/ so it can be rebuilt from scratch."""

    if splits_dir.exists():
        shutil.rmtree(splits_dir)


def save_split_images(
    df: pd.DataFrame, raw_dir: Path, splits_dir: Path, save_size: int
) -> None:
    """Save every image as an RGB JPEG at splits_dir/<file>, shrunk to save_size."""

    pairs = zip(df["source"], df["file"], strict=True)
    for source, file in track(pairs, total=len(df), description="Saving images"):
        dest_path = splits_dir / file
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(raw_dir / source) as image:
            upright = ImageOps.exif_transpose(image)
            rgb = upright.convert("RGB")

        rgb = shrink_to_short_side(rgb, save_size)
        rgb.save(dest_path, "JPEG", quality=95)
