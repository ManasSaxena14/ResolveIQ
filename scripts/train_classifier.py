"""Fine-tune DistilBERT on the ticket queues and log each run to MLflow.

Each seed is trained into models/classifier/seed-<seed>/. After training, the
review threshold is chosen on the validation set (the lowest confidence above
which accuracy reaches the target) and stored with the model.

Usage:
    python scripts/train_classifier.py                 # first seed in the config
    python scripts/train_classifier.py --seed 7
    python scripts/train_classifier.py --all-seeds
    python scripts/train_classifier.py --limit 500 --epochs 1   # quick smoke run
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from resolveiq.classifier.calibration import (
    choose_review_threshold,
    expected_calibration_error,
)
from resolveiq.classifier.data import load_labels, load_split
from resolveiq.classifier.train import (
    TrainConfig,
    predict_proba,
    select_device,
    train_classifier,
    update_meta,
)
from resolveiq.config import Settings, load_config
from resolveiq.tracking import tracked_run

TARGET_ACCURACY = 0.90


def train_seed(settings: Settings, seed: int, limit: int | None, epochs: int | None) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = settings.classifier
    data_dir = settings.resolve(cfg.processed_dir)
    train, validation = load_split(data_dir, "train"), load_split(data_dir, "validation")
    if limit:
        train, validation = train[:limit], validation[: max(limit // 4, 50)]
    labels = load_labels(data_dir)
    config = TrainConfig(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        epochs=epochs or cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        class_weights=cfg.class_weights,
        seed=seed,
    )
    suffix = "-smoke" if limit else ""
    output_dir = settings.resolve(cfg.output_dir) / f"seed-{seed}{suffix}"
    device = select_device()
    print(f"Training seed {seed} on {device} ({len(train)} train, {len(validation)} validation)")

    run_name = f"distilbert-seed-{seed}{suffix}"
    with tracked_run(settings, "classifier", run_name=run_name) as mlflow:
        if mlflow:
            mlflow.set_tag("model_family", "transformer")
            mlflow.log_params({**config.__dict__, "train_size": len(train), "device": str(device)})

        def log_epoch(record: dict[str, float]) -> None:
            print(f"  epoch {record['epoch']}: {record}")
            if mlflow:
                step = int(record["epoch"])
                mlflow.log_metrics({k: v for k, v in record.items() if k != "epoch"}, step=step)

        result = train_classifier(
            [t.text for t in train],
            [t.label for t in train],
            [t.text for t in validation],
            [t.label for t in validation],
            labels,
            config,
            output_dir,
            device=device,
            on_epoch=log_epoch,
        )

        # Choose the review threshold on validation predictions of the best checkpoint.
        model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        probabilities = predict_proba(
            model, tokenizer, [t.text for t in validation], config.max_length, device
        )
        confidence = probabilities.max(axis=1)
        correct = np.array(
            [
                labels[i] == t.label
                for i, t in zip(probabilities.argmax(axis=1), validation, strict=True)
            ]
        )
        threshold = choose_review_threshold(confidence, correct, TARGET_ACCURACY)
        ece = expected_calibration_error(confidence, correct)
        update_meta(
            output_dir,
            review_threshold=threshold.threshold,
            review_target_accuracy=threshold.target_accuracy,
            review_coverage_validation=threshold.coverage,
            val_expected_calibration_error=ece,
        )
        if mlflow:
            mlflow.log_metrics(
                {
                    "best_val_macro_f1": result.best_val_macro_f1,
                    "best_epoch": result.best_epoch,
                    "train_seconds": result.train_seconds,
                    "review_threshold": threshold.threshold,
                    "review_coverage_val": threshold.coverage,
                    "val_ece": ece,
                }
            )

    print(
        f"Seed {seed}: best validation macro F1 {result.best_val_macro_f1:.4f} at epoch "
        f"{result.best_epoch}, {result.train_seconds}s. Review threshold "
        f"{threshold.threshold:.3f} routes {threshold.coverage:.1%} of tickets automatically."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--seed", type=int, help="Seed to train (defaults to the first in config)")
    parser.add_argument("--all-seeds", action="store_true", help="Train every seed in the config")
    parser.add_argument("--limit", type=int, help="Use only the first N training tickets")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("resolveiq").setLevel(logging.INFO)
    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False
    seeds = (
        settings.classifier.seeds if args.all_seeds else [args.seed or settings.classifier.seeds[0]]
    )
    for seed in seeds:
        train_seed(settings, seed, args.limit, args.epochs)


if __name__ == "__main__":
    main()
