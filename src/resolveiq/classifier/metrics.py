"""Classification metrics, reports and plots."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro and weighted F1, and a per-class report.

    Macro F1 is the headline metric: it weights every queue equally, so the
    small queues count as much as the large ones.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    }
    weakest = min(per_class, key=lambda label: per_class[label]["f1"])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="weighted")), 4),
        "weakest_class": weakest,
        "weakest_class_f1": per_class[weakest]["f1"],
        "per_class": per_class,
    }


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(labels))


def save_per_class_report(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for label, row in metrics["per_class"].items():
            writer.writerow([label, row["precision"], row["recall"], row["f1"], row["support"]])


def plot_confusion_matrix(
    matrix: np.ndarray, labels: Sequence[str], path: Path, title: str
) -> None:
    """Row-normalized confusion matrix, so each row shows recall per true class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = normalized[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                color="white" if value > 0.5 else "black",
            )  # fmt: skip
    ax.set_xlabel("Predicted queue")
    ax.set_ylabel("True queue")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
