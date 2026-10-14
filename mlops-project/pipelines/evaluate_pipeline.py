"""
Evaluation pipeline: metrics on val and test for one run folder.

Models are compared and chosen on val; test is only reported.

Run from the mlops-project/ folder (evaluates the run_name in configs/train.yaml):
    uv run python -m pipelines.evaluate_pipeline
"""

from pathlib import Path

from src.core.config import EvaluateConfig, TrainConfig, load_config
from src.core.logger import get_logger
from src.core.paths import CONFIGS_DIR, RUNS_DIR, SPLITS_DIR
from src.core.reproducibility import get_device
from src.evaluation.metrics import (
    build_confusion_matrix,
    per_class_metrics,
    summary_metrics,
)
from src.evaluation.plots import save_confusion_matrix
from src.evaluation.predict import predict, predictions_table
from src.models.factory import build_model
from src.preprocessing.loaders import build_dataset, build_loader
from src.preprocessing.transforms import build_eval_transform
from src.tracking.run_folder import eval_dir, load_weights, read_run_info, write_metrics

logger = get_logger("mlops.pipeline.evaluate")

EVAL_SPLITS = ["val", "test"]


def run_evaluate(
    config_path: Path = CONFIGS_DIR / "evaluate.yaml",
    train_config_path: Path = CONFIGS_DIR / "train.yaml",
) -> dict:
    """Evaluate the run named in train.yaml and return the metrics."""

    logger.info("=== Starting Evaluation Pipeline ===")

    # Step 1: Load settings and the run folder
    logger.info("Step 1: Loading configuration and run...")
    config = load_config(config_path, EvaluateConfig)
    run_name = load_config(train_config_path, TrainConfig).run_name

    run_dir = RUNS_DIR / run_name
    run_info = read_run_info(run_dir)
    class_names = run_info["class_names"]
    image_size = run_info["image_size"]
    logger.info(f"Run {run_name}: {run_info['backbone']}, classes {class_names}")

    # Step 2: Rebuild the model and load its weights
    logger.info(f"Step 2: Loading {config.weights}.pt...")
    device = get_device("auto")
    model = build_model(run_info["backbone"], len(class_names), pretrained=False)
    model.load_state_dict(load_weights(run_dir, config.weights))
    model = model.to(device)

    # Step 3: Predict and score every split
    output_dir = eval_dir(run_dir)
    results: dict = {"run_name": run_name, "weights": config.weights}

    for split in EVAL_SPLITS:
        logger.info(f"Step 3: Evaluating {split}...")
        dataset = build_dataset(SPLITS_DIR / split, build_eval_transform(image_size))
        if dataset.classes != class_names:
            raise ValueError(
                f"{split} classes {dataset.classes} differ from the run's {class_names}"
            )
        loader = build_loader(dataset, config.batch_size, False, 0, seed=0)
        prediction = predict(model, loader, device, f"Predicting {split}")

        matrix = build_confusion_matrix(
            prediction["y_true"], prediction["y_pred"], len(class_names)
        )
        results[split] = {
            "loss": prediction["loss"],
            **summary_metrics(prediction["y_true"], prediction["y_pred"]),
            "per_class": per_class_metrics(
                prediction["y_true"], prediction["y_pred"], class_names
            ),
            "confusion_matrix": matrix,
        }

        # Step 4: Save predictions and the confusion matrix figure
        table = predictions_table(dataset, prediction, class_names, SPLITS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / f"predictions_{split}.csv", index=False)
        save_confusion_matrix(
            matrix,
            class_names,
            f"{run_name} — {split}",
            output_dir / f"confusion_matrix_{split}.png",
        )
        logger.info(
            f"{split}: accuracy {results[split]['accuracy']:.4f}  "
            f"macro F1 {results[split]['macro_f1']:.4f}"
        )

    # Step 5: Save metrics.json
    logger.info("Step 5: Saving eval/metrics.json...")
    write_metrics(run_dir, results)

    logger.info("=== Evaluation Pipeline Completed ===")
    return results


if __name__ == "__main__":
    run_evaluate()
