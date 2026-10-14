"""
Classification metrics.

Macro F1 is the main metric: every class counts the same, so a weak small
class is not hidden by a strong large one (unlike accuracy).
"""

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def summary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Accuracy and macro F1."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_metrics(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> dict[str, dict[str, float]]:
    """Precision, recall, F1 and support for every class."""

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    result = {}
    for index, class_name in enumerate(class_names):
        result[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    return result


def build_confusion_matrix(
    y_true: list[int], y_pred: list[int], num_classes: int
) -> list[list[int]]:
    """Rows = true class, columns = predicted class."""

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return matrix.tolist()
