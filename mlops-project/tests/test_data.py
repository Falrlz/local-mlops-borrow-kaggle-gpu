"""Tests for src/data/: cleaning, splitting and validation."""

from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.cleaning import drop_unreadable, find_images, shrink_to_short_side
from src.data.splitting import add_file_names, stratified_split
from src.data.validation import check_every_class_in_every_split, check_no_leakage


def make_raw_dir(raw_dir: Path) -> None:
    """3 classes x 20 images, one broken file and a LICENSE.txt at the top."""

    for class_name in ["a", "b", "c"]:
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for index in range(20):
            Image.new("RGB", (40, 30)).save(class_dir / f"{index}.jpg")
    (raw_dir / "a" / "broken.jpg").write_bytes(b"not an image")
    (raw_dir / "LICENSE.txt").write_text("license", encoding="utf-8")


def test_find_images_ignores_files_outside_class_folders(tmp_path: Path) -> None:
    make_raw_dir(tmp_path)
    df = find_images(tmp_path)

    assert len(df) == 61
    assert sorted(df["class_name"].unique()) == ["a", "b", "c"]


def test_drop_unreadable(tmp_path: Path) -> None:
    make_raw_dir(tmp_path)
    df, dropped = drop_unreadable(find_images(tmp_path), tmp_path)

    assert len(df) == 60
    assert dropped == ["a/broken.jpg"]


def test_shrink_to_short_side_never_upscales() -> None:
    big = shrink_to_short_side(Image.new("RGB", (500, 333)), 224)
    small = shrink_to_short_side(Image.new("RGB", (180, 240)), 224)

    assert min(big.size) == 224
    assert big.size == (336, 224)
    assert small.size == (180, 240)


def test_stratified_split_keeps_classes_and_has_no_leakage() -> None:
    rows = []
    for class_name in ["a", "b"]:
        for index in range(40):
            rows.append(
                {"class_name": class_name, "source": f"{class_name}/{index}.jpg"}
            )
    df = add_file_names(stratified_split(pd.DataFrame(rows), (70, 15, 15), seed=42))

    counts = pd.crosstab(df["class_name"], df["split"])
    assert list(counts.loc["a"]) == list(counts.loc["b"])
    assert len(df[df["split"] == "train"]) == 56
    assert check_no_leakage(df) == []
    assert check_every_class_in_every_split(df) == []


def test_same_seed_gives_same_split() -> None:
    rows = []
    for index in range(30):
        rows.append({"class_name": "a" if index % 2 else "b", "source": f"{index}.jpg"})
    df = pd.DataFrame(rows)

    first = stratified_split(df, (70, 15, 15), seed=7)
    second = stratified_split(df, (70, 15, 15), seed=7)
    assert first.equals(second)
