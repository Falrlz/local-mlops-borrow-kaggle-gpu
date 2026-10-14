"""
Run a model over a data loader and collect its predictions.

The only inference loop in the project: used for validation during training
and for the evaluation pipeline.
"""

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.core.logger import progress_bar


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str = "Predicting",
) -> dict:
    """
    Predict every batch in loader, with a progress bar.

    Returns y_true, y_pred, confidence (softmax probability of y_pred) as lists
    and loss as the mean cross-entropy.
    """

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidence: list[float] = []
    loss_sum = 0.0

    with progress_bar() as progress:
        task = progress.add_task(description, total=len(loader), status="")

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss_sum += functional.cross_entropy(logits, labels, reduction="sum").item()

            probs = torch.softmax(logits, dim=1)
            top_prob, top_class = probs.max(dim=1)

            y_true += labels.tolist()
            y_pred += top_class.tolist()
            confidence += top_prob.tolist()

            progress.update(task, advance=1)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
        "loss": loss_sum / len(y_true),
    }


def predictions_table(
    dataset: ImageFolder, result: dict, class_names: list[str], splits_dir: Path
) -> pd.DataFrame:
    """One row per image: file, true and predicted class, confidence, correct."""

    rows = []
    for index, (image_path, _) in enumerate(dataset.samples):
        true_name = class_names[result["y_true"][index]]
        pred_name = class_names[result["y_pred"][index]]
        rows.append(
            {
                "file": Path(image_path).relative_to(splits_dir).as_posix(),
                "true": true_name,
                "pred": pred_name,
                "confidence": round(result["confidence"][index], 4),
                "correct": true_name == pred_name,
            }
        )
    return pd.DataFrame(rows)
