"""
Full pipeline: training -> evaluation -> log to MLflow.

On the laptop everything runs. On Kaggle only training and evaluation run
(there is no local MLflow there): after unpacking the run on the laptop, run
log_run_pipeline by hand.

data/splits/ must already exist (data prep runs once on the laptop). The run
name comes from configs/train.yaml.

Run from the mlops-project/ folder:
    uv run python -m pipelines.full_pipeline
"""

from pathlib import Path

from pipelines.evaluate_pipeline import run_evaluate
from pipelines.log_run_pipeline import run_log_run
from pipelines.train_pipeline import run_train
from src.core.config import TrackingConfig, load_config
from src.core.logger import get_logger
from src.core.paths import CONFIGS_DIR
from src.core.reproducibility import is_kaggle

logger = get_logger("mlops.pipeline.full")


def run_full_pipeline(configs_dir: Path = CONFIGS_DIR) -> str:
    """Train, evaluate, then (on the laptop) log to MLflow; return the run name."""

    logger.info("=== RUNNING FULL PIPELINE ===")
    train_config_path = configs_dir / "train.yaml"
    tracking = load_config(configs_dir / "tracking.yaml", TrackingConfig)

    # Step 1: Training (writes outputs/runs/<run_name>/)
    logger.info("Step 1/3: Training")
    run_name = run_train(train_config_path)

    # Step 2: Evaluation of the run just trained
    logger.info("Step 2/3: Evaluation")
    run_evaluate(configs_dir / "evaluate.yaml", train_config_path)

    # Step 3: Log the run folder to the local MLflow (laptop only)
    if is_kaggle():
        logger.info(
            "Step 3/3: On Kaggle, MLflow skipped. On the laptop, after "
            "unpacking, run log_run_pipeline."
        )
    elif tracking.use_mlflow:
        logger.info("Step 3/3: Logging to MLflow")
        run_log_run(train_config_path)
    else:
        logger.info("Step 3/3: MLflow off (use_mlflow: false), skipped")

    logger.info(f"=== FULL PIPELINE COMPLETED (run {run_name}) ===")
    return run_name


if __name__ == "__main__":
    run_full_pipeline()
