import csv

import numpy as np
import pytest

from resolveiq.classifier.baseline import (
    load_baseline,
    predict_with_confidence,
    save_baseline,
    train_baseline,
)
from resolveiq.classifier.metrics import (
    classification_metrics,
    confusion,
    plot_confusion_matrix,
    save_per_class_report,
)

LABELS = ["billing", "outage", "returns"]


def test_classification_metrics():
    y_true = ["billing", "billing", "outage", "outage", "returns", "returns"]
    y_pred = ["billing", "billing", "outage", "returns", "returns", "returns"]

    metrics = classification_metrics(y_true, y_pred, LABELS)

    assert metrics["accuracy"] == pytest.approx(5 / 6, abs=1e-4)
    assert metrics["per_class"]["billing"]["f1"] == 1.0
    assert metrics["per_class"]["outage"]["recall"] == 0.5
    assert metrics["per_class"]["returns"]["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["weakest_class"] == "outage"
    expected_macro = (1.0 + 2 / 3 + 0.8) / 3
    assert metrics["macro_f1"] == pytest.approx(expected_macro, abs=1e-4)


def test_confusion_and_reports(tmp_path):
    y_true = ["billing", "outage", "returns", "returns"]
    y_pred = ["billing", "returns", "returns", "returns"]

    matrix = confusion(y_true, y_pred, LABELS)
    save_per_class_report(classification_metrics(y_true, y_pred, LABELS), tmp_path / "r.csv")
    plot_confusion_matrix(matrix, LABELS, tmp_path / "cm.png", "Test")

    assert matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 2]]
    rows = list(csv.DictReader((tmp_path / "r.csv").open()))
    assert [r["label"] for r in rows] == LABELS
    assert (tmp_path / "cm.png").stat().st_size > 0


def test_baseline_learns_separable_text(tmp_path):
    words = {
        "billing": "invoice charge payment",
        "outage": "server down offline",
        "returns": "return parcel exchange",
    }
    texts = [f"{words[label]} case {i}" for label in LABELS for i in range(15)]
    labels = [label for label in LABELS for _ in range(15)]

    model = train_baseline(texts, labels)
    predicted, confidence = predict_with_confidence(
        model, ["the server is down again", "refund the invoice charge"]
    )
    save_baseline(model, tmp_path / "baseline.joblib")

    assert predicted == ["outage", "billing"]
    assert np.all((confidence > 0) & (confidence <= 1))
    assert list(load_baseline(tmp_path / "baseline.joblib").predict(["return my parcel"])) == [
        "returns"
    ]
