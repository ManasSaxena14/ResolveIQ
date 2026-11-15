"""Confidence calibration and the human-review threshold."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True)
class ReviewThreshold:
    threshold: float
    target_accuracy: float
    accuracy_above: float
    coverage: float


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Group predictions by confidence and compare confidence with accuracy."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lower, upper in itertools.pairwise(edges):
        last = upper == edges[-1]
        mask = (confidence >= lower) & ((confidence <= upper) if last else (confidence < upper))
        if mask.any():
            bins.append(
                ReliabilityBin(
                    lower=round(float(lower), 2),
                    upper=round(float(upper), 2),
                    count=int(mask.sum()),
                    mean_confidence=float(confidence[mask].mean()),
                    accuracy=float(correct[mask].mean()),
                )
            )
    return bins


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """Average gap between confidence and accuracy, weighted by bin size."""
    total = len(confidence)
    if total == 0:
        return 0.0
    bins = reliability_bins(confidence, correct, n_bins)
    return round(sum(b.count / total * abs(b.accuracy - b.mean_confidence) for b in bins), 4)


def choose_review_threshold(
    confidence: np.ndarray, correct: np.ndarray, target_accuracy: float = 0.9
) -> ReviewThreshold:
    """Lowest confidence above which predictions reach ``target_accuracy``.

    Predictions below the returned threshold are flagged for human review.
    ``coverage`` is the share of tickets that would be routed automatically.
    If no threshold reaches the target, the highest confidence is returned,
    which sends almost everything to review.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidence) == 0:
        raise ValueError("Cannot choose a threshold without predictions")

    order = np.argsort(-confidence)
    sorted_conf = confidence[order]
    running_accuracy = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)

    # Walk from the most to the least confident prediction and keep the last
    # position where accuracy is still at or above the target.
    meets = np.nonzero(running_accuracy >= target_accuracy)[0]
    index = int(meets[-1]) if len(meets) else 0
    threshold = float(sorted_conf[index])
    above = confidence >= threshold
    return ReviewThreshold(
        threshold=round(threshold, 4),
        target_accuracy=target_accuracy,
        accuracy_above=round(float(correct[above].mean()), 4),
        coverage=round(float(above.mean()), 4),
    )


def plot_reliability(bins: list[ReliabilityBin], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(
        [b.mean_confidence for b in bins],
        [b.accuracy for b in bins],
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
