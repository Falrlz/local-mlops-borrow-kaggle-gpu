"""
Pack mlops-project for Kaggle into upload-to-kaggle/code.zip and data.zip.

Run from the mlops-project/ folder (standard library only, any Python 3.11+):
    uv run python ../kaggle-bridge/pack.py

    code.zip   mlops-project/  configs, src, pipelines, pyproject.toml, uv.lock,
                               requirements.txt, outputs/pretrained/, git_info.json
    data.zip   splits/         data/splits/ (never data/raw/)

The zips are written with zipfile, so paths inside always use "/" (Kaggle
rejects the "\\" that Windows Compress-Archive writes).
"""

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from common import (
    GIT_INFO_FILE,
    PRETRAINED_DIR,
    PROJECT_DIR,
    PROJECT_NAME,
    RUNS_DIR,
    SPLITS_DIR,
    TRAIN_CONFIG_PATH,
    UPLOAD_DIR,
    get_logger,
    read_run_name,
)

logger = get_logger("bridge.pack")

# Files and folders of the project that Kaggle needs
CODE_ITEMS = [
    "configs",
    "src",
    "pipelines",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    ".python-version",
]
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc"}


def find_problems(run_name: str) -> list[str]:
    """Everything that would make the Kaggle run fail, found on the laptop."""

    problems = []
    if not (SPLITS_DIR / "split_manifest.csv").exists():
        problems.append(
            f"{SPLITS_DIR} is missing. Run: uv run python -m pipelines.data_prep_pipeline"
        )
    if (RUNS_DIR / run_name).exists():
        problems.append(
            f"run_name '{run_name}' is already used in {RUNS_DIR}. "
            f"Set a new run_name in {TRAIN_CONFIG_PATH}."
        )
    if not (PROJECT_DIR / "uv.lock").exists():
        problems.append(f"{PROJECT_DIR / 'uv.lock'} is missing. Run: uv lock")
    return problems


def run_command(args: list[str], cwd: Path) -> str | None:
    """Run a command and return its output, or None if it is missing or fails."""

    if shutil.which(args[0]) is None:
        return None
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def export_requirements() -> None:
    """Refresh requirements.txt from uv.lock (used by the pip option on Kaggle)."""

    output = run_command(
        [
            "uv",
            "export",
            "--format",
            "requirements-txt",
            "--frozen",
            "--no-dev",
            "--no-hashes",
            "--no-emit-project",
            "--output-file",
            "requirements.txt",
        ],
        PROJECT_DIR,
    )
    if output is None:
        logger.warning(
            "uv export failed or uv is missing: requirements.txt not refreshed"
        )


def read_git_info() -> dict:
    """Commit and uncommitted-changes flag of the project (read-only git calls)."""

    commit = run_command(["git", "rev-parse", "HEAD"], PROJECT_DIR)
    if commit is None:
        return {"commit": None, "dirty": None}
    changes = run_command(["git", "status", "--porcelain", "--", "."], PROJECT_DIR)
    return {"commit": commit, "dirty": changes != ""}


def list_files(path: Path) -> list[Path]:
    """A file, or every file below a folder, without caches."""

    if path.is_file():
        return [path]

    files = []
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.suffix in SKIP_SUFFIXES:
            continue
        if SKIP_DIRS.intersection(file.parts):
            continue
        files.append(file)
    return files


def code_files() -> list[Path]:
    """Every file that goes into code.zip."""

    files = []
    for item in CODE_ITEMS:
        path = PROJECT_DIR / item
        if path.exists():
            files += list_files(path)
        else:
            logger.warning(f"{item} not found, not packed")

    if PRETRAINED_DIR.exists():
        files += list_files(PRETRAINED_DIR)
    else:
        logger.warning(
            f"{PRETRAINED_DIR} is empty: Kaggle will download the pretrained "
            "weights (needs Internet On)"
        )
    return files


def write_zip(
    zip_path: Path,
    files: list[Path],
    base_dir: Path,
    prefix: str,
    compression: int,
    extra: dict[str, str] | None = None,
) -> None:
    """Write files as <prefix>/<path relative to base_dir>, always with '/'."""

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=compression) as archive:
        for file in files:
            arcname = f"{prefix}/{file.relative_to(base_dir).as_posix()}"
            archive.write(file, arcname)
        for name, text in (extra or {}).items():
            archive.writestr(f"{prefix}/{name}", text)

    size_mb = zip_path.stat().st_size / 1024**2
    logger.info(f"{zip_path.name}: {len(files)} files, {size_mb:.1f} MB")


def is_up_to_date(zip_path: Path, source: Path) -> bool:
    """True if zip_path exists and is newer than source."""

    return zip_path.exists() and zip_path.stat().st_mtime >= source.stat().st_mtime


def main() -> None:
    """Check the project, then write code.zip and (if needed) data.zip."""

    logger.info("=== Packing for Kaggle ===")

    # Step 1: Check on the laptop what would fail on Kaggle
    run_name = read_run_name()
    logger.info(f"Step 1: Checking {PROJECT_NAME} (run_name: {run_name})...")
    problems = find_problems(run_name)
    if len(problems) > 0:
        for problem in problems:
            logger.error(problem)
        sys.exit(1)

    git_info = read_git_info()
    if git_info["dirty"]:
        logger.warning(
            "Uncommitted changes in the project: the run will record commit "
            f"{git_info['commit']} with git_dirty: true"
        )

    # Step 2: code.zip (small, rebuilt every time)
    logger.info("Step 2: Writing code.zip...")
    export_requirements()
    write_zip(
        UPLOAD_DIR / "code.zip",
        code_files(),
        PROJECT_DIR,
        PROJECT_NAME,
        zipfile.ZIP_DEFLATED,
        extra={GIT_INFO_FILE: json.dumps(git_info, indent=2)},
    )

    # Step 3: data.zip (large, only rebuilt when data/splits/ changed)
    logger.info("Step 3: Writing data.zip...")
    data_zip = UPLOAD_DIR / "data.zip"
    if is_up_to_date(data_zip, SPLITS_DIR / "split_manifest.csv"):
        logger.info("data.zip is up to date (delete it to rebuild)")
    else:
        # JPEGs are already compressed: store them as they are
        write_zip(
            data_zip, list_files(SPLITS_DIR), SPLITS_DIR, "splits", zipfile.ZIP_STORED
        )

    logger.info(f"=== Done: upload the zips in {UPLOAD_DIR} to Kaggle ===")


if __name__ == "__main__":
    main()
