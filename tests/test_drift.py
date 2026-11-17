import numpy as np
import pytest

from resolveiq.monitoring.drift import (
    categorical_psi,
    detect_drift,
    numeric_psi,
    render_html,
    retrain_decision,
)

RNG = np.random.default_rng(0)
REFERENCE = {
    "risk_score": RNG.uniform(0, 1, 1000).tolist(),
    "category": (["Billing"] * 500 + ["Technical"] * 500),
}


def _current(shift: bool) -> dict:
    rng = np.random.default_rng(1)
    if shift:
        return {
            "risk_score": rng.uniform(0.6, 1, 400).tolist(),
            "category": ["Technical"] * 380 + ["Billing"] * 20,
        }
    return {
        "risk_score": rng.uniform(0, 1, 400).tolist(),
        "category": ["Billing"] * 200 + ["Technical"] * 200,
    }


def test_psi_is_zero_for_identical_data_and_grows_with_shift():
    same = numeric_psi(REFERENCE["risk_score"], REFERENCE["risk_score"])
    moved = numeric_psi(REFERENCE["risk_score"], _current(True)["risk_score"])

    assert same == pytest.approx(0.0, abs=1e-9)
    assert moved > 1.0
    assert categorical_psi(["a", "b"], ["a", "b"]) == pytest.approx(0.0)
    assert categorical_psi(["a"] * 9 + ["b"], ["b"] * 9 + ["a"]) > 1.0


def test_constant_reference_is_handled():
    assert numeric_psi([1.0] * 10, [1.0] * 10) == pytest.approx(0.0, abs=1e-6)
    assert numeric_psi([1.0] * 10, [2.0] * 10) > 1.0


def test_stable_traffic_does_not_drift():
    report = detect_drift(REFERENCE, _current(False), ["risk_score"], ["category"])

    assert report.drifted == []
    assert report.share_drifted == 0
    assert report.n_reference == 1000 and report.n_current == 400


def test_shifted_traffic_drifts_on_both_features():
    report = detect_drift(REFERENCE, _current(True), ["risk_score"], ["category"])

    assert report.drifted == ["risk_score", "category"]
    assert report.features[0].test == "Kolmogorov-Smirnov"
    assert report.features[1].test == "Chi-square"
    assert report.as_dict()["share_drifted"] == 1.0


def test_significant_but_tiny_shift_is_not_drift():
    rng = np.random.default_rng(2)
    reference = {"x": rng.normal(0, 1, 50_000).tolist()}
    current = {"x": rng.normal(0.05, 1, 50_000).tolist()}

    feature = detect_drift(reference, current, ["x"], []).features[0]

    assert feature.p_value < 0.05
    assert feature.psi < 0.2
    assert feature.drifted is False


def test_single_category_has_no_test():
    report = detect_drift({"c": ["a"] * 5}, {"c": ["a"] * 5}, [], ["c"])

    assert report.features[0].p_value == 1.0


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="needs reference"):
        detect_drift({"x": []}, {"x": [1.0]}, ["x"], [])
    with pytest.raises(ValueError, match="missing"):
        detect_drift({"x": [1.0]}, {"y": [1.0]}, ["x"], [])


def test_retrain_rule():
    drifted = detect_drift(REFERENCE, _current(True), ["risk_score"], ["category"])
    stable = detect_drift(REFERENCE, _current(False), ["risk_score"], ["category"])

    by_drift = retrain_decision(drifted, 0, 0.0, 0.5, 0.3, 20)
    by_feedback = retrain_decision(stable, 40, 0.4, 0.5, 0.3, 20)
    too_few_votes = retrain_decision(stable, 5, 1.0, 0.5, 0.3, 20)

    assert by_drift.retrain and "2 of 2" in by_drift.reasons[0]
    assert by_feedback.retrain and "Thumbs-down rate 40%" in by_feedback.reasons[0]
    assert not too_few_votes.retrain


def test_html_report_is_self_contained():
    current = _current(True)
    report = detect_drift(REFERENCE, current, ["risk_score"], ["category"])
    decision = retrain_decision(report, 0, 0.0, 0.5, 0.3, 20)

    page = render_html(report, decision, REFERENCE, current, notes=["Sample <data>"])

    assert page.startswith("<!doctype html>")
    assert "Retrain recommended" in page
    assert page.count("data:image/png;base64,") == 2
    assert "Sample &lt;data&gt;" in page
    stable = detect_drift(REFERENCE, _current(False), ["risk_score"], ["category"])
    calm = render_html(
        stable, retrain_decision(stable, 0, 0, 0.5, 0.3, 20), REFERENCE, _current(False)
    )
    assert "No retraining needed" in calm
