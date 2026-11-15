"""Final test-set evaluation of the baseline and every DistilBERT seed.

Run this once, after all tuning decisions were made on the validation split.
It writes to results/classifier/:

    metrics.json            all test metrics
    comparison.md           baseline versus DistilBERT table
    per_class_report.csv    per-queue precision, recall and F1 (production model)
    confusion_matrix.png    row-normalized confusion matrix (production model)
    calibration.png         reliability curve on the test split (production model)
    errors.csv              misclassified test tickets for error analysis (not committed)

The production model is the seed with the best validation macro F1, so the
test set is never used to pick a model. It is copied to models/classifier/production.

Usage:
    python scripts/evaluate_classifier.py [--config PATH] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics

import numpy as np

from resolveiq.classifier.baseline import load_baseline
from resolveiq.classifier.calibration import (
    expected_calibration_error,
    plot_reliability,
    reliability_bins,
)
from resolveiq.classifier.data import load_labels, load_split
from resolveiq.classifier.metrics import (
    classification_metrics,
    confusion,
    plot_confusion_matrix,
    save_per_class_report,
)
from resolveiq.classifier.train import predict_proba, read_meta, select_device
from resolveiq.config import load_config
from resolveiq.tracking import tracked_run

ERROR_TEXT_CHARS = 400


def _summary(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_class"}


def _format_row(name: str, metrics: dict, train_time: str) -> str:
    return (
        f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
        f"{metrics['weakest_class']} ({metrics['weakest_class_f1']:.4f}) | {train_time} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False
    cfg = settings.classifier
    data_dir = settings.resolve(cfg.processed_dir)
    models_dir = settings.resolve(cfg.output_dir)
    out_dir = settings.resolve(settings.paths.results_dir) / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)

    test = load_split(data_dir, "test")
    labels = load_labels(data_dir)
    texts, y_true = [t.text for t in test], [t.label for t in test]

    # Baseline
    baseline_path = models_dir.parent / "baseline.joblib"
    baseline_meta = json.loads(baseline_path.with_name("baseline_meta.json").read_text())
    baseline_metrics = classification_metrics(
        y_true, load_baseline(baseline_path).predict(texts), labels
    )

    # DistilBERT seeds
    device = select_device()
    seed_dirs = sorted(
        p for p in models_dir.glob("seed-*") if p.is_dir() and not p.name.endswith("-smoke")
    )
    if not seed_dirs:
        raise SystemExit(f"No trained seeds found in {models_dir}. Run train_classifier.py first.")
    seeds: dict[str, dict] = {}
    probabilities_by_seed: dict[str, np.ndarray] = {}
    for seed_dir in seed_dirs:
        meta = read_meta(seed_dir)
        model = AutoModelForSequenceClassification.from_pretrained(seed_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(seed_dir)
        probabilities = predict_proba(model, tokenizer, texts, meta["max_length"], device)
        predictions = [labels[i] for i in probabilities.argmax(axis=1)]
        seeds[seed_dir.name] = {
            "meta": meta,
            "metrics": classification_metrics(y_true, predictions, labels),
        }
        probabilities_by_seed[seed_dir.name] = probabilities
        print(f"{seed_dir.name}: test macro F1 {seeds[seed_dir.name]['metrics']['macro_f1']:.4f}")

    # Production model: chosen on validation, never on test.
    production = max(seeds, key=lambda name: seeds[name]["meta"]["best_val_macro_f1"])
    prod_meta = seeds[production]["meta"]
    prod_metrics = seeds[production]["metrics"]
    prod_probabilities = probabilities_by_seed[production]
    confidence = prod_probabilities.max(axis=1)
    predicted = [labels[i] for i in prod_probabilities.argmax(axis=1)]
    correct = np.array([p == t for p, t in zip(predicted, y_true, strict=True)])
    threshold = float(prod_meta["review_threshold"])
    auto_routed = confidence >= threshold

    macro_f1s = [s["metrics"]["macro_f1"] for s in seeds.values()]
    accuracies = [s["metrics"]["accuracy"] for s in seeds.values()]
    spread = statistics.stdev(macro_f1s) if len(macro_f1s) > 1 else 0.0
    review = {
        "threshold": threshold,
        "target_accuracy": prod_meta.get("review_target_accuracy"),
        "test_coverage": round(float(auto_routed.mean()), 4),
        "test_accuracy_auto_routed": round(float(correct[auto_routed].mean()), 4)
        if auto_routed.any()
        else None,
        "test_accuracy_needs_review": round(float(correct[~auto_routed].mean()), 4)
        if (~auto_routed).any()
        else None,
    }
    results = {
        "test_size": len(test),
        "baseline": {"metrics": baseline_metrics, "train_seconds": baseline_meta["train_seconds"]},
        "distilbert": {
            name: {
                "metrics": s["metrics"],
                "best_val_macro_f1": s["meta"]["best_val_macro_f1"],
                "best_epoch": s["meta"]["best_epoch"],
                "train_seconds": s["meta"]["train_seconds"],
            }
            for name, s in seeds.items()
        },
        "distilbert_mean": {
            "macro_f1_mean": round(statistics.mean(macro_f1s), 4),
            "macro_f1_std": round(spread, 4),
            "accuracy_mean": round(statistics.mean(accuracies), 4),
        },
        "production_model": production,
        "production_review": review,
        "production_test_ece": expected_calibration_error(confidence, correct),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    save_per_class_report(prod_metrics, out_dir / "per_class_report.csv")
    save_per_class_report(baseline_metrics, out_dir / "per_class_report_baseline.csv")
    plot_confusion_matrix(
        confusion(y_true, predicted, labels),
        labels,
        out_dir / "confusion_matrix.png",
        f"DistilBERT ({production}) on the test split",
    )
    plot_reliability(
        reliability_bins(confidence, correct),
        out_dir / "calibration.png",
        f"Confidence calibration ({production}, test split)",
    )

    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticket_id", "true_label", "predicted_label", "confidence", "text"])
        for ticket, pred, conf in zip(test, predicted, confidence, strict=True):
            if pred != ticket.label:
                text = ticket.text.replace("\n", " ")[:ERROR_TEXT_CHARS]
                writer.writerow([ticket.ticket_id, ticket.label, pred, f"{conf:.3f}", text])

    def minutes(seconds: float) -> str:
        return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"

    seed_seconds = [s["meta"]["train_seconds"] for s in seeds.values()]
    rows = [
        "| Model | Accuracy | Macro F1 | Weakest class (F1) | Training time |",
        "|---|---|---|---|---|",
        _format_row(
            "TF-IDF + logistic regression",
            baseline_metrics,
            minutes(baseline_meta["train_seconds"]),
        ),
    ]
    for name, s in seeds.items():
        rows.append(
            _format_row(f"DistilBERT, {name}", s["metrics"], minutes(s["meta"]["train_seconds"]))
        )
    if len(seeds) > 1:
        rows.append(
            f"| DistilBERT, mean of {len(seeds)} seeds | {statistics.mean(accuracies):.4f} | "
            f"{statistics.mean(macro_f1s):.4f} +/- {spread:.4f} | | "
            f"{minutes(statistics.mean(seed_seconds))} per seed |"
        )
    report = "\n".join(
        [
            "# Ticket classifier: test results",
            "",
            f"Generated by `scripts/evaluate_classifier.py` on the frozen test split "
            f"({len(test)} tickets).",
            "",
            *rows,
            "",
            f"Production model: `{production}` (highest validation macro F1).",
            "",
            f"Review threshold {threshold:.3f} (chosen on validation for "
            f"{review['target_accuracy']:.0%} accuracy): on the test split it routes "
            f"{review['test_coverage']:.1%} of tickets automatically with accuracy "
            f"{review['test_accuracy_auto_routed']:.4f}; the rest are flagged for human review.",
            "",
        ]
    )
    (out_dir / "comparison.md").write_text(report, encoding="utf-8")

    with tracked_run(settings, "classifier", run_name="final-test") as mlflow:
        if mlflow:
            mlflow.set_tag("stage", "final-test")
            mlflow.log_metrics(
                {
                    "test_baseline_macro_f1": baseline_metrics["macro_f1"],
                    "test_baseline_accuracy": baseline_metrics["accuracy"],
                    "test_distilbert_macro_f1_mean": results["distilbert_mean"]["macro_f1_mean"],
                    "test_distilbert_macro_f1_std": results["distilbert_mean"]["macro_f1_std"],
                    "test_production_macro_f1": prod_metrics["macro_f1"],
                    "test_review_coverage": review["test_coverage"],
                }
            )
            for name in (
                "metrics.json",
                "comparison.md",
                "confusion_matrix.png",
                "calibration.png",
            ):
                mlflow.log_artifact(str(out_dir / name))

    production_dir = models_dir / "production"
    shutil.copytree(models_dir / production, production_dir, dirs_exist_ok=True)
    print()
    print(report)
    print(f"Production model copied to {production_dir}")


if __name__ == "__main__":
    main()
