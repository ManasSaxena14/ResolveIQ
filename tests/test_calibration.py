import numpy as np
import pytest

from resolveiq.classifier.calibration import (
    choose_review_threshold,
    expected_calibration_error,
    plot_reliability,
    reliability_bins,
)


def test_reliability_bins_and_ece():
    confidence = np.array([0.95, 0.95, 0.55, 0.55, 1.0])
    correct = np.array([1, 1, 1, 0, 1], dtype=bool)

    bins = reliability_bins(confidence, correct, n_bins=10)

    assert [(b.lower, b.count) for b in bins] == [(0.5, 2), (0.9, 3)]
    assert bins[0].accuracy == 0.5
    assert expected_calibration_error(confidence, correct) == pytest.approx(
        (2 * 0.05 + 3 * abs(1 - (0.95 + 0.95 + 1.0) / 3)) / 5, abs=1e-4
    )
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0


def test_review_threshold_reaches_target_accuracy():
    confidence = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 1, 0, 1, 0, 0], dtype=bool)

    result = choose_review_threshold(confidence, correct, target_accuracy=0.8)

    assert result.threshold == 0.6
    assert result.accuracy_above == pytest.approx(5 / 6, abs=1e-4)
    assert result.coverage == 0.75


def test_review_threshold_when_target_is_unreachable():
    result = choose_review_threshold(np.array([0.9, 0.8]), np.array([0, 0], dtype=bool), 0.9)

    assert result.threshold == 0.9
    assert result.coverage == 0.5


def test_review_threshold_requires_predictions():
    with pytest.raises(ValueError):
        choose_review_threshold(np.array([]), np.array([]))


def test_reliability_plot_is_written(tmp_path):
    bins = reliability_bins(np.array([0.2, 0.8]), np.array([0, 1], dtype=bool))

    plot_reliability(bins, tmp_path / "cal.png", "Test")

    assert (tmp_path / "cal.png").stat().st_size > 0
