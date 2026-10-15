"""
Seed, device and environment information.

Used by the training and evaluation pipelines.
"""

import json
import os
import platform
import random
import shutil
import subprocess

import numpy as np
import torch
import torchvision

from src.core.paths import BASE_DIR, GIT_INFO_PATH


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


def run_git(args: list[str]) -> str | None:
    """Run a read-only git command in BASE_DIR; None if git or the repo is missing."""

    if shutil.which("git") is None:
        return None

    result = subprocess.run(
        ["git", *args],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info() -> dict:
    """
    Return the git commit and whether the project has uncommitted changes.

    In a git checkout git is asked directly. On Kaggle there is no .git folder,
    so the values come from git_info.json, written into code.zip by
    kaggle-bridge/pack.py. Unknown values are None.
    """

    commit = run_git(["rev-parse", "HEAD"])
    if commit is not None:
        changes = run_git(["status", "--porcelain", "--", "."])
        return {"commit": commit, "dirty": changes != ""}

    if GIT_INFO_PATH.exists():
        with GIT_INFO_PATH.open(encoding="utf-8") as file:
            return json.load(file)

    return {"commit": None, "dirty": None}


def environment_info() -> dict:
    """Describe where a run happened: platform, versions, GPU, git commit."""

    run_platform = "kaggle" if is_kaggle() else "local"

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    git = git_info()
    return {
        "platform": run_platform,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }
