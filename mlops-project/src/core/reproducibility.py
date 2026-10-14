"""
Seed, device and environment information.

Used by the training and evaluation pipelines.
"""

import os
import platform
import random
import shutil
import subprocess

import numpy as np
import torch
import torchvision

from src.core.paths import BASE_DIR


def set_seed(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and PyTorch; optionally make cuDNN deterministic."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(choice: str) -> torch.device:
    """Turn 'auto' / 'cpu' / 'cuda' into a torch device."""

    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no GPU is available")

    return torch.device(choice)


def is_kaggle() -> bool:
    """True inside a Kaggle notebook session."""

    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_git_commit() -> str | None:
    """Return the current git commit, or None outside a git checkout (e.g. Kaggle)."""

    if shutil.which("git") is None:
        return None

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def environment_info() -> dict:
    """Describe where a run happened: platform, versions, GPU, git commit."""

    run_platform = "kaggle" if is_kaggle() else "local"

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "platform": run_platform,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "git_commit": get_git_commit(),
    }
