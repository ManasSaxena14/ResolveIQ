"""Train the TF-IDF and logistic regression baseline and log it to MLflow.

The baseline is scored on the validation split here. Test-set scores are
produced once, together with the transformer, by scripts/evaluate_classifier.py.

Usage:
    python scripts/train_baseline.py [--config PATH] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import time

from resolveiq.classifier.baseline import save_baseline, train_baseline
from resolveiq.classifier.data import load_labels, load_split
from resolveiq.classifier.metrics import classification_metrics
from resolveiq.config import load_config
from resolveiq.tracking import tracked_run

BASELINE_FILENAME = "baseline.joblib"
BASELINE_META_FILENAME = "baseline_meta.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False
    cfg = settings.classifier
    data_dir = settings.resolve(cfg.processed_dir)
    train, validation = load_split(data_dir, "train"), load_split(data_dir, "validation")
    labels = load_labels(data_dir)

    with tracked_run(settings, "classifier", run_name="baseline-tfidf-logreg") as mlflow:
        started = time.perf_counter()
        model = train_baseline([t.text for t in train], [t.label for t in train], cfg.split.seed)
        seconds = round(time.perf_counter() - started, 1)
        predictions = model.predict([t.text for t in validation])
        metrics = classification_metrics([t.label for t in validation], predictions, labels)

        path = settings.resolve(cfg.output_dir).parent / BASELINE_FILENAME
        save_baseline(model, path)
        meta = {"train_seconds": seconds, "validation": metrics}
        path.with_name(BASELINE_META_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        if mlflow:
            mlflow.set_tag("model_family", "baseline")
            mlflow.log_params(
                {
                    "model": "tfidf-logreg",
                    "ngram_range": "1-2",
                    "C": 4.0,
                    "class_weight": "balanced",
                }
            )
            mlflow.log_metrics(
                {
                    "val_accuracy": metrics["accuracy"],
                    "val_macro_f1": metrics["macro_f1"],
                    "val_weighted_f1": metrics["weighted_f1"],
                    "train_seconds": seconds,
                }
            )

    print(f"Baseline trained in {seconds}s and saved to {path}")
    print(f"Validation accuracy  {metrics['accuracy']:.4f}")
    print(f"Validation macro F1  {metrics['macro_f1']:.4f}")
    print(f"Weakest class        {metrics['weakest_class']} (F1 {metrics['weakest_class_f1']:.4f})")


if __name__ == "__main__":
    main()
