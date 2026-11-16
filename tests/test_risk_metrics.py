import numpy as np
import pytest

from resolveiq.risk.evaluate import (
    alert_stats,
    binary_metrics,
    choose_threshold,
    cost_table,
    plot_precision_recall,
    plot_probability_calibration,
    pr_auc,
    recall_at_precision,
)

LABELS = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
SCORES = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])


def test_pr_auc_is_one_for_perfect_ranking():
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_recall_at_precision():
    # Alerting on 0.9 and 0.8 is 100% precise and catches 2 of 3.
    assert recall_at_precision(LABELS, SCORES, 0.9) == pytest.approx(2 / 3)
    # At 50% precision every escalation can be caught (0.3 cut: 3 of 4 right).
    assert recall_at_precision(LABELS, SCORES, 0.5) == pytest.approx(1.0)


def test_binary_metrics_reports_majority_as_uninformative():
    constant = np.full(len(LABELS), LABELS.mean())

    metrics = binary_metrics(LABELS, constant, 0.7)

    assert metrics["pr_auc"] == pytest.approx(LABELS.mean())
    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["positive_rate"] == pytest.approx(0.3)
    assert set(metrics) == {"pr_auc", "recall_at_precision", "brier", "roc_auc", "positive_rate"}


def test_alert_stats():
    stats = alert_stats(LABELS, SCORES, 0.5, miss_cost=10, false_alarm_cost=1)

    assert stats.alert_rate == pytest.approx(0.3)
    assert stats.caught == pytest.approx(2 / 3, abs=1e-4)
    assert stats.false_alarms_per_100_alerts == pytest.approx(33.3)
    assert stats.cost == 10 * 1 + 1 * 1


def test_higher_miss_cost_lowers_the_threshold():
    cheap = choose_threshold(LABELS, SCORES, miss_cost=0.5)
    expensive = choose_threshold(LABELS, SCORES, miss_cost=20)

    assert expensive <= 0.3 < cheap


def test_cost_table_uses_validation_threshold():
    rows = cost_table(LABELS, SCORES, LABELS, SCORES, [0.5, 20])

    assert [row.cost_ratio for row in rows] == [0.5, 20.0]
    assert rows[1].caught == pytest.approx(1.0)
    assert rows[0].threshold > rows[1].threshold
    assert rows[0].as_dict()["cost_ratio"] == 0.5


def test_plots_are_written(tmp_path):
    plot_precision_recall(LABELS, {"model": SCORES}, tmp_path / "pr.png", "PR")
    plot_probability_calibration(LABELS, {"model": SCORES}, tmp_path / "cal.png", "Cal")

    assert (tmp_path / "pr.png").stat().st_size > 0
    assert (tmp_path / "cal.png").stat().st_size > 0
