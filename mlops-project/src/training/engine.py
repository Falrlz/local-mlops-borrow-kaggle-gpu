"""
Training loop with early stopping on val macro F1.

Used by pipelines/train_pipeline.py.
"""

import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.core.logger import get_logger, progress_bar
from src.evaluation.metrics import summary_metrics
from src.evaluation.predict import predict

logger = get_logger("mlops.training")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    description: str,
) -> dict[str, float]:
    """
    Train for one pass over loader and return the mean training loss and
    accuracy.

    The progress bar shows the running loss and accuracy of the epoch so far.
    """

    model.train()
    loss_sum = 0.0
    n_correct = 0
    n_images = 0

    with progress_bar() as progress:
        task = progress.add_task(description, total=len(loader), status="")

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(labels)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_images += len(labels)

            status = f"loss {loss_sum / n_images:.4f}  acc {n_correct / n_images:.4f}"
            progress.update(task, advance=1, status=status)

    return {"loss": loss_sum / n_images, "accuracy": n_correct / n_images}


def validate_one_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, description: str
) -> dict[str, float]:
    """Return val loss, accuracy and macro F1."""

    result = predict(model, loader, device, description)
    metrics = summary_metrics(result["y_true"], result["y_pred"])
    return {"loss": result["loss"], **metrics}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
) -> dict:
    """
    Train up to `epochs`, stop early after `patience` epochs without a better
    val macro F1.

    Returns history (one dict per epoch), best_state / last_state (state_dicts),
    best_epoch and best_val_macro_f1.
    """

    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            f"Epoch {epoch}/{epochs} train",
        )
        val = validate_one_epoch(
            model, val_loader, device, f"Epoch {epoch}/{epochs} val"
        )
        seconds = time.perf_counter() - start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "seconds": round(seconds, 1),
            }
        )

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        marker = " *" if improved else ""
        logger.info(
            f"Epoch {epoch}/{epochs}  train_loss {train['loss']:.4f}  "
            f"train_acc {train['accuracy']:.4f}  val_loss {val['loss']:.4f}  val_acc {val['accuracy']:.4f}  "
            f"val_macro_f1 {val['macro_f1']:.4f}  ({seconds:.0f}s){marker}"
        )

        if epoch - best_epoch >= patience:
            logger.info(f"Early stopping: no val macro F1 gain for {patience} epochs")
            break

    return {
        "history": history,
        "best_state": best_state,
        "last_state": copy.deepcopy(model.state_dict()),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
    }
