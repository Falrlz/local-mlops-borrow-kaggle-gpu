"""
Log run pipeline: record one finished run folder in the local MLflow.

The same code for local and Kaggle runs. full_pipeline calls it automatically
after a local run. A Kaggle run is unzipped into outputs/runs/<run_name>/
first, then logged by hand.

Run from the mlops-project/ folder (logs the run_name in configs/train.yaml):
    uv run python -m pipelines.log_run_pipeline
"""

from pathlib import Path

from src.core.config import DatasetConfig, TrainConfig, load_config
from src.core.logger import get_logger
from src.core.paths import CONFIGS_DIR, MLFLOW_DIR, RUNS_DIR
from src.tracking.mlflow_logger import (
    epoch_metrics,
    eval_metrics,
    find_run_id,
    log_run,
    setup_mlflow,
)
from src.tracking.run_folder import (
    read_history,
    read_metrics,
    read_run_info,
    run_config_path,
)

logger = get_logger("mlops.pipeline.log_run")


def run_log_run(train_config_path: Path = CONFIGS_DIR / "train.yaml") -> str:
    """Log outputs/runs/<run_name>/ to MLflow and return the MLflow run id."""

    logger.info("=== Starting Log Run Pipeline ===")

    # Step 1: Load settings and check that the run is complete
    logger.info("Step 1: Loading configuration and run folder...")
    run_name = load_config(train_config_path, TrainConfig).run_name
    run_dir = RUNS_DIR / run_name
    run_info = read_run_info(run_dir)
    eval_results = read_metrics(run_dir)

    # Step 2: Connect to MLflow and refuse to log the same run twice
    logger.info("Step 2: Setting up MLflow...")
    experiment_id = setup_mlflow(MLFLOW_DIR)
    existing_id = find_run_id(experiment_id, run_name)
    if existing_id is not None:
        raise ValueError(
            f"Run '{run_name}' is already in MLflow (run id {existing_id})."
        )

    # Step 3: Read params, metrics and tags from the run folder
    logger.info("Step 3: Reading the run folder...")
    train_config = load_config(run_config_path(run_dir, "train.yaml"), TrainConfig)
    dataset_config = load_config(
        run_config_path(run_dir, "dataset.yaml"), DatasetConfig
    )
    params = train_config.model_dump()
    for name, value in dataset_config.model_dump().items():
        params[f"data.{name}"] = value

    epochs = epoch_metrics(read_history(run_dir))
    metrics = eval_metrics(eval_results)
    metrics["best_epoch"] = run_info["best_epoch"]
    metrics["duration_minutes"] = run_info["duration_minutes"]

    environment = run_info["environment"]
    tags = {
        "backbone": run_info["backbone"],
        "platform": environment["platform"],
        "gpu": str(environment["gpu"]),
        "git_commit": str(environment["git_commit"]),
        "git_dirty": str(environment["git_dirty"]),
        "data_fingerprint": run_info["data_fingerprint"],
        "eval_weights": eval_results["weights"],
        "started_at": run_info["started_at"],
    }
    logger.info(f"{len(params)} params, {len(epochs)} epochs, {len(metrics)} metrics")

    # Step 4: Write the MLflow run (artifacts = a copy of the run folder)
    logger.info("Step 4: Logging to MLflow...")
    run_id = log_run(run_name, run_dir, params, tags, epochs, metrics)

    logger.info(f"Logged {run_name} as MLflow run {run_id}")
    logger.info("=== Log Run Pipeline Completed ===")
    return run_id


if __name__ == "__main__":
    run_log_run()
