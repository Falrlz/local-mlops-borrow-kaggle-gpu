"""
Run folder -> local MLflow. The only module that imports mlflow.

Everything is kept in outputs/mlflow/: mlflow.db (runs, params, metrics, tags)
and artifacts/ (a copy of each logged run folder). Open the UI from the
mlops-project/ folder:
    uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
"""

from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "image-classification"


def setup_mlflow(mlflow_dir: Path) -> str:
    """Point MLflow to mlflow_dir, select (or create) the experiment, return its id."""

    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').as_posix()}")

    # Create the experiment ourselves so its artifacts go to mlflow_dir/artifacts/
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(mlflow_dir / "artifacts").as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def find_run_id(experiment_id: str, run_name: str) -> str | None:
    """Return the MLflow run id of run_name, or None if it was never logged."""

    runs = MlflowClient().search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if len(runs) == 0:
        return None
    return runs[0].info.run_id


def epoch_metrics(history: pd.DataFrame) -> list[tuple[int, dict[str, float]]]:
    """history.csv rows as (epoch, {metric: value}) pairs."""

    rows = []
    for record in history.to_dict(orient="records"):
        epoch = int(record.pop("epoch"))
        record["epoch_seconds"] = record.pop("seconds")
        rows.append(
            (epoch, {str(name): float(value) for name, value in record.items()})
        )
    return rows


def eval_metrics(eval_results: dict) -> dict[str, float]:
    """Summary metrics of eval/metrics.json, named eval/<split>_<metric>."""

    metrics = {}
    for split in ["val", "test"]:
        for name in ["loss", "accuracy", "macro_f1"]:
            metrics[f"eval/{split}_{name}"] = float(eval_results[split][name])
    return metrics


def log_run(
    run_name: str,
    run_dir: Path,
    params: dict,
    tags: dict[str, str],
    epochs: list[tuple[int, dict[str, float]]],
    metrics: dict[str, float],
) -> str:
    """Write one MLflow run and copy the whole run folder as artifacts."""

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        for step, values in epochs:
            mlflow.log_metrics(values, step=step)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_dir))
        return run.info.run_id
