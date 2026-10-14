"""
Training pipeline: train one model and write its run folder.

Same command on the laptop and on Kaggle. MLflow is not used here: the run
folder is logged to MLflow afterwards, for local and Kaggle runs alike.

Run from the mlops-project/ folder:
    uv run python -m pipelines.train_pipeline
"""

import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from src.core.config import TrainConfig, load_config
from src.core.logger import get_logger
from src.core.paths import (
    CONFIGS_DIR,
    MANIFEST_PATH,
    PRETRAINED_DIR,
    RUNS_DIR,
    SPLITS_DIR,
)
from src.core.reproducibility import environment_info, get_device, set_seed
from src.data.splitting import data_fingerprint
from src.models.factory import build_model
from src.preprocessing.loaders import build_dataset, build_loader
from src.preprocessing.transforms import build_eval_transform, build_train_transform
from src.tracking.run_folder import (
    copy_configs,
    create_run_folder,
    save_weights,
    write_history,
    write_run_info,
)
from src.training.engine import fit

logger = get_logger("mlops.pipeline.train")


def run_train(config_path: Path = CONFIGS_DIR / "train.yaml") -> str:
    """Train the model in configs/train.yaml and return the run name."""

    logger.info("=== Starting Training Pipeline ===")
    started_at = datetime.now(UTC)
    start = time.perf_counter()

    # Step 1: Load settings and check the inputs
    logger.info("Step 1: Loading configuration...")
    config = load_config(config_path, TrainConfig)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run pipelines.data_prep_pipeline first."
        )

    run_dir = RUNS_DIR / config.run_name
    if run_dir.exists():
        raise FileExistsError(
            f"{run_dir} already exists. Use a new 'run_name' in train.yaml."
        )

    # Step 2: Seed and device
    logger.info("Step 2: Setting seed and device...")
    set_seed(config.seed, config.deterministic)
    device = get_device(config.device)
    environment = environment_info()
    logger.info(f"Device: {device} ({environment['gpu'] or 'no GPU'})")

    # Step 3: Data loaders (train with augmentation, val without)
    logger.info("Step 3: Building data loaders...")
    train_set = build_dataset(
        SPLITS_DIR / "train", build_train_transform(config.image_size)
    )
    val_set = build_dataset(SPLITS_DIR / "val", build_eval_transform(config.image_size))
    if train_set.classes != val_set.classes:
        raise ValueError(
            f"Class folders differ: train {train_set.classes}, val {val_set.classes}"
        )

    class_names = train_set.classes
    train_loader = build_loader(
        train_set, config.batch_size, True, config.num_workers, config.seed
    )
    val_loader = build_loader(
        val_set, config.batch_size, False, config.num_workers, config.seed
    )
    logger.info(
        f"{len(train_set)} train / {len(val_set)} val images, classes: {class_names}"
    )

    # Step 4: Model (pretrained weights cached in outputs/pretrained/)
    logger.info(f"Step 4: Building {config.backbone}...")
    torch.hub.set_dir(str(PRETRAINED_DIR))
    model = build_model(config.backbone, len(class_names), config.pretrained)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Step 5: Train with early stopping on val macro F1
    logger.info(f"Step 5: Training up to {config.epochs} epochs...")
    result = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config.epochs,
        config.patience,
    )
    logger.info(
        f"Best epoch {result['best_epoch']}: "
        f"val macro F1 {result['best_val_macro_f1']:.4f}"
    )

    # Step 6: Write the run folder
    logger.info(f"Step 6: Writing run folder {run_dir}...")
    create_run_folder(run_dir)
    copy_configs(run_dir, [config_path, CONFIGS_DIR / "dataset.yaml"])
    write_history(run_dir, result["history"])
    save_weights(run_dir, "best", result["best_state"])
    save_weights(run_dir, "last", result["last_state"])

    run_info = {
        "run_name": config.run_name,
        "backbone": config.backbone,
        "pretrained": config.pretrained,
        "image_size": config.image_size,
        "num_classes": len(class_names),
        "class_names": class_names,
        "epochs_trained": len(result["history"]),
        "best_epoch": result["best_epoch"],
        "best_val_macro_f1": result["best_val_macro_f1"],
        "n_images": {"train": len(train_set), "val": len(val_set)},
        "data_fingerprint": data_fingerprint(MANIFEST_PATH),
        "started_at": started_at.isoformat(timespec="seconds"),
        "duration_minutes": round((time.perf_counter() - start) / 60, 1),
        "environment": environment,
    }
    write_run_info(run_dir, run_info)

    logger.info("=== Training Pipeline Completed ===")
    return config.run_name


if __name__ == "__main__":
    run_train()
