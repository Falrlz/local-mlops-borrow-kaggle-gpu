"""
Evaluation figures.

Used by pipelines/evaluate_pipeline.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window (works on Kaggle and in CI)

import matplotlib.pyplot as plt  # noqa: E402


def save_confusion_matrix(
    matrix: list[list[int]], class_names: list[str], title: str, output_path: Path
) -> None:
    """Save the confusion matrix as a PNG with the count in every cell."""

    size = max(4.0, 0.9 * len(class_names) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    largest = max(max(row) for row in matrix)
    for row_index, row in enumerate(matrix):
        for col_index, count in enumerate(row):
            color = "white" if count > largest / 2 else "black"
            ax.text(
                col_index, row_index, str(count), ha="center", va="center", color=color
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
