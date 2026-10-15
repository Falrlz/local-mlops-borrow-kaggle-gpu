"""
Unpack a run downloaded from Kaggle into mlops-project/outputs/runs/<run_name>/.

Run from the mlops-project/ folder (standard library only, any Python 3.11+):
    uv run python ../kaggle-bridge/unpack.py             # newest zip in download-from-kaggle/
    uv run python ../kaggle-bridge/unpack.py path/to/run.zip

The run folder is found by its run_info.json, so a zip with an extra folder
level (run/run/...) is unpacked correctly too. Nothing is overwritten.
"""

import json
import sys
import zipfile
from pathlib import Path, PurePosixPath

from common import (
    DOWNLOAD_DIR,
    PROJECT_NAME,
    RUN_REQUIRED_FILES,
    RUNS_DIR,
    get_logger,
    read_run_name,
)

logger = get_logger("bridge.unpack")


def pick_zip(args: list[str]) -> Path:
    """The zip given on the command line, or the newest one in download-from-kaggle/."""

    if len(args) > 0:
        return Path(args[0])

    zips = sorted(DOWNLOAD_DIR.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if len(zips) == 0:
        raise FileNotFoundError(
            f"No zip in {DOWNLOAD_DIR}. Download <run_name>.zip from the Kaggle "
            "notebook's Output tab into that folder."
        )
    return zips[-1]


def find_run_root(names: list[str]) -> PurePosixPath:
    """The folder inside the zip that holds run_info.json (must be exactly one)."""

    roots = []
    for name in names:
        path = PurePosixPath(name)
        if path.name == "run_info.json":
            roots.append(path.parent)

    if len(roots) != 1:
        raise ValueError(
            f"Expected one run_info.json in the zip, found {len(roots)}. "
            "Is this a run zip from train_on_kaggle.ipynb?"
        )
    return roots[0]


def extract_run(archive: zipfile.ZipFile, root: PurePosixPath, target: Path) -> int:
    """Extract every file below root into target; return the file count."""

    count = 0
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or root not in path.parents:
            continue

        relative = path.relative_to(root)
        destination = (target / relative).resolve()
        # Never write outside the target folder
        if target.resolve() not in destination.parents:
            raise ValueError(f"Unsafe path in zip: {info.filename}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as output:
            output.write(source.read())
        count += 1
    return count


def missing_files(run_dir: Path) -> list[str]:
    """Required run files that are not in run_dir."""

    missing = []
    for name in RUN_REQUIRED_FILES:
        if not (run_dir / name).exists():
            missing.append(name)
    return missing


def main() -> None:
    """Unpack one run zip and tell the next step."""

    logger.info("=== Unpacking a Kaggle run ===")

    # Step 1: Open the zip and find the run inside it
    zip_path = pick_zip(sys.argv[1:])
    logger.info(f"Step 1: Reading {zip_path}...")
    with zipfile.ZipFile(zip_path) as archive:
        root = find_run_root(archive.namelist())
        run_info = json.loads(archive.read(str(root / "run_info.json")))
        run_name = run_info["run_name"]

        # Step 2: Extract to outputs/runs/<run_name>/, never over an existing run
        run_dir = RUNS_DIR / run_name
        logger.info(f"Step 2: Extracting to {run_dir}...")
        if run_dir.exists():
            logger.error(f"{run_dir} already exists. Nothing was changed.")
            sys.exit(1)
        count = extract_run(archive, root, run_dir)
    logger.info(f"{count} files extracted")

    # Step 3: Check that the run is complete
    logger.info("Step 3: Checking the run folder...")
    missing = missing_files(run_dir)
    if len(missing) > 0:
        logger.error(f"Missing in the run folder: {missing}")
        sys.exit(1)

    environment = run_info["environment"]
    logger.info(
        f"Run {run_name}: {run_info['epochs_trained']} epoch(s) on "
        f"{environment['gpu'] or 'CPU'}, best val macro F1 "
        f"{run_info['best_val_macro_f1']:.4f}"
    )

    # Step 4: Next step (log_run reads run_name from train.yaml)
    if read_run_name() != run_name:
        logger.warning(
            f"configs/train.yaml has a different run_name. Set "
            f"'run_name: {run_name}' before logging."
        )
    logger.info(
        f"=== Done. Next, in {PROJECT_NAME}/: "
        "uv run python -m pipelines.log_run_pipeline ==="
    )


if __name__ == "__main__":
    main()
