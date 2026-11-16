"""Metrics, cost-based alert thresholds and plots for the escalation model.

Accuracy is deliberately absent: a model that never raises an alert can look
accurate on imbalanced data and still be useless. PR-AUC is the headline
metric, with recall at a fixed precision, the Brier score and ROC-AUC.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def pr_auc(labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(labels, scores))


def recall_at_precision(
    labels: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, target: float
) -> float:
    """Highest recall reachable while at least ``target`` of alerts are right."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    reachable = recall[precision >= target]
    return float(reachable.max()) if len(reachable) else 0.0


def binary_metrics(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    precision_target: float = 0.7,
) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    return {
        "pr_auc": round(pr_auc(labels, scores), 4),
        "recall_at_precision": round(recall_at_precision(labels, scores, precision_target), 4),
        "brier": round(float(brier_score_loss(labels, scores)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "positive_rate": round(float(labels.mean()), 4),
    }


@dataclass(frozen=True)
class AlertStats:
    threshold: float
    alert_rate: float
    caught: float
    false_alarms_per_100_alerts: float
    cost: float


def alert_stats(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    miss_cost: float = 1.0,
    false_alarm_cost: float = 1.0,
) -> AlertStats:
    """What happens when every score at or above ``threshold`` raises an alert."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    alerts = scores >= threshold
    positives = labels == 1
    true_alerts = int((alerts & positives).sum())
    false_alerts = int((alerts & ~positives).sum())
    misses = int((~alerts & positives).sum())
    return AlertStats(
        threshold=round(float(threshold), 2),
        alert_rate=round(float(alerts.mean()), 4),
        caught=round(true_alerts / max(1, int(positives.sum())), 4),
        false_alarms_per_100_alerts=round(100 * false_alerts / max(1, int(alerts.sum())), 1),
        cost=float(miss_cost * misses + false_alarm_cost * false_alerts),
    )


def choose_threshold(
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    miss_cost: float,
    false_alarm_cost: float = 1.0,
    thresholds: Sequence[float] = tuple(DEFAULT_THRESHOLDS),
) -> float:
    """Threshold with the lowest total cost. Use validation data only."""
    costs = [alert_stats(labels, scores, t, miss_cost, false_alarm_cost).cost for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])


@dataclass(frozen=True)
class CostRow:
    cost_ratio: float
    threshold: float
    caught: float
    false_alarms_per_100_alerts: float
    alert_rate: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def cost_table(
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    ratios: Sequence[float],
) -> list[CostRow]:
    """For each miss-to-false-alarm cost ratio: threshold from validation, results on test."""
    rows = []
    for ratio in ratios:
        threshold = choose_threshold(val_labels, val_scores, miss_cost=ratio)
        stats = alert_stats(test_labels, test_scores, threshold)
        rows.append(
            CostRow(
                cost_ratio=float(ratio),
                threshold=threshold,
                caught=stats.caught,
                false_alarms_per_100_alerts=stats.false_alarms_per_100_alerts,
                alert_rate=stats.alert_rate,
            )
        )
    return rows


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_precision_recall(
    labels: np.ndarray, scores: Mapping[str, np.ndarray], path: Path, title: str
) -> None:
    from sklearn.metrics import precision_recall_curve

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(labels, values)
        ax.plot(recall, precision, label=f"{name} (PR-AUC {pr_auc(labels, values):.3f})")
    ax.axhline(float(np.mean(labels)), linestyle="--", color="grey", label="Positive rate")
    ax.set_xlabel("Recall (escalations caught)")
    ax.set_ylabel("Precision (alerts that were right)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_probability_calibration(
    labels: np.ndarray, scores: Mapping[str, np.ndarray], path: Path, title: str, n_bins: int = 10
) -> None:
    """Mean predicted probability against the observed escalation rate."""
    from resolveiq.classifier.calibration import reliability_bins

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    for name, values in scores.items():
        bins = reliability_bins(values, labels, n_bins)
        ax.plot(
            [b.mean_confidence for b in bins], [b.accuracy for b in bins], marker="o", label=name
        )
    ax.set_xlabel("Predicted escalation probability")
    ax.set_ylabel("Observed escalation rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
