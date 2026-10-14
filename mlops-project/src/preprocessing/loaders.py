"""
Datasets and data loaders for data/splits/<split>/<class>/.

Used by the training and evaluation pipelines.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def build_dataset(split_dir: Path, transform: transforms.Compose) -> ImageFolder:
    """One split folder as a dataset (class names = sorted folder names)."""

    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} not found. Run pipelines.data_prep_pipeline first."
        )
    return ImageFolder(str(split_dir), transform=transform)


def build_loader(
    dataset: ImageFolder,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """
    Wrap a dataset in a DataLoader with a seeded shuffle order.

    The shuffled (training) loader drops an incomplete last batch: a batch of a
    single image breaks BatchNorm layers.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
