"""Build the escalation risk features for the train, validation and test splits.

Uses the frozen ticket splits from scripts/prepare_tickets.py. Two TF-IDF
models supply text-based features: a queue classifier (Phase 2 baseline) and,
if enabled, a model of escalation wording. For the training split their scores
are out-of-fold; validation and test are scored by models fitted on the whole
training split, which are saved for scoring new tickets.

Usage:
    python scripts/build_risk_features.py [--config PATH]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from resolveiq.classifier.baseline import save_baseline, train_baseline
from resolveiq.classifier.data import SPLIT_NAMES, load_split
from resolveiq.config import load_config
from resolveiq.risk.features import (
    FeatureSchema,
    feature_matrix,
    label_from_priority,
    out_of_fold_probabilities,
)
from resolveiq.risk.predict import QUEUE_MODEL_FILENAME, TEXT_MODEL_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a config file")
    args = parser.parse_args()

    settings = load_config(args.config)
    cfg = settings.risk
    data_dir = settings.resolve(settings.classifier.processed_dir)
    features_dir = settings.resolve(cfg.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    splits = {name: load_split(data_dir, name) for name in SPLIT_NAMES}
    train = splits["train"]

    model_dir = settings.resolve(cfg.output_dir)
    labels_by_split = {
        name: np.array(
            [label_from_priority(t.priority, cfg.label.positive_values) for t in tickets],
            dtype=np.int8,
        )
        for name, tickets in splits.items()
    }
    train_texts = [t.text for t in train]

    started = time.perf_counter()
    train_probs, queues = out_of_fold_probabilities(
        train_texts, [t.label for t in train], cfg.queue_folds, cfg.seed
    )
    queue_model = train_baseline(train_texts, [t.label for t in train], cfg.seed)
    save_baseline(queue_model, model_dir / QUEUE_MODEL_FILENAME)

    text_model, train_text_scores = None, None
    if cfg.text_score_feature:
        train_labels = labels_by_split["train"].astype(int).tolist()
        oof, _ = out_of_fold_probabilities(train_texts, train_labels, cfg.queue_folds, cfg.seed)
        train_text_scores = oof[:, 1]
        text_model = train_baseline(train_texts, train_labels, cfg.seed)
        save_baseline(text_model, model_dir / TEXT_MODEL_FILENAME)
    print(f"Text model scores ready in {time.perf_counter() - started:.0f}s")

    schema = FeatureSchema(queues=tuple(queues), text_score=cfg.text_score_feature)
    profile: dict[str, object] = {
        "label": {
            "source_column": cfg.label.source_column,
            "positive_values": cfg.label.positive_values,
            "is_proxy": True,
        },
        "feature_count": len(schema.names),
        "feature_names": schema.names,
        "queues": list(queues),
        "splits": {},
    }
    for name, tickets in splits.items():
        probabilities = (
            train_probs
            if name == "train"
            else queue_model.predict_proba([t.text for t in tickets]).astype(np.float32)
        )
        texts = [t.text for t in tickets]
        text_scores = None
        if text_model is not None:
            text_scores = (
                train_text_scores if name == "train" else text_model.predict_proba(texts)[:, 1]
            )
        features = feature_matrix(
            texts, [t.ticket_type for t in tickets], probabilities, schema, text_scores
        )
        labels = labels_by_split[name]
        np.savez_compressed(
            features_dir / f"{name}.npz",
            features=features,
            labels=labels,
            ticket_ids=np.array([t.ticket_id for t in tickets]),
        )
        profile["splits"][name] = {
            "tickets": len(tickets),
            "escalations": int(labels.sum()),
            "escalation_rate": round(float(labels.mean()), 4),
        }
        print(f"{name:<10} {len(tickets):>6} tickets, escalation rate {labels.mean():.3f}")

    (features_dir / "schema.json").write_text(
        json.dumps(schema.as_dict(), indent=2),
        encoding="utf-8",
    )
    results_dir = settings.resolve(settings.paths.results_dir) / "risk"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "data_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Features written to {features_dir}")


if __name__ == "__main__":
    main()
