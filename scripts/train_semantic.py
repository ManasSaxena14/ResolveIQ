"""Train the semantic model that decides queues and risk from a ticket's meaning.

Past tickets are the training split. The retail models are logistic regressions on the
embeddings of data/seed/retail_tickets.csv. Calibration, review thresholds and the alert
threshold use out-of-fold retail predictions and the validation split; the test split and
the holdout half of eval/ticket_testset.csv are only read by scripts/evaluate_semantic.py.

Usage:
    python scripts/train_semantic.py [--config PATH] [--no-mlflow]
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # see docs/risk_model.md

import argparse
import csv
import time
from dataclasses import replace

import numpy as np

from resolveiq.classifier.calibration import choose_review_threshold
from resolveiq.classifier.data import clean_text, load_labels, load_split
from resolveiq.config import load_config
from resolveiq.rag.embeddings import SentenceTransformerEmbedder
from resolveiq.registry import resolve_model_dir
from resolveiq.risk.evaluate import binary_metrics, choose_threshold
from resolveiq.risk.features import label_from_priority
from resolveiq.risk.predict import RiskModel
from resolveiq.semantic import (
    SemanticModel,
    SemanticParams,
    logit,
    sigmoid,
    soft_share,
    softmax,
)
from resolveiq.tracking import git_commit, tracked_run


def best_scale(scores: np.ndarray, labels: np.ndarray) -> float:
    """Temperature (as a multiplier) with the lowest log loss."""
    grid = np.arange(0.5, 4.01, 0.05)
    rows = np.arange(len(labels))
    losses = [-np.log(softmax(scores * g)[rows, labels] + 1e-9).mean() for g in grid]
    return round(float(grid[int(np.argmin(losses))]), 2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False
    cfg = settings.semantic
    started = time.perf_counter()

    data_dir = settings.resolve(settings.classifier.processed_dir)
    train, validation = load_split(data_dir, "train"), load_split(data_dir, "validation")
    labels = load_labels(data_dir)
    index = {label: i for i, label in enumerate(labels)}
    positives = settings.risk.label.positive_values
    with settings.resolve(cfg.seed_path).open(newline="", encoding="utf-8") as handle:
        seed = list(csv.DictReader(handle))
    seed_texts = [clean_text(row["body"]) for row in seed]
    seed_queue = np.array([index[row["queue"]] for row in seed])
    seed_urgent = np.array([int(row["urgent"]) for row in seed])

    embedder = SentenceTransformerEmbedder(settings.rag.embedding_model, settings.rag.query_prefix)
    print(f"Embedding {len(train)} past, {len(validation)} validation, {len(seed)} retail tickets")
    past = embedder.embed_documents([t.text for t in train])
    val = embedder.embed_documents([t.text for t in validation])
    examples = embedder.embed_documents(seed_texts)

    folds = StratifiedKFold(cfg.folds, shuffle=True, random_state=settings.risk.seed)
    queue_lr = LogisticRegression(C=cfg.queue_c, max_iter=3000)
    urgency_lr = LogisticRegression(C=cfg.urgency_c, max_iter=3000, class_weight="balanced")
    queue_oof = cross_val_predict(
        queue_lr, examples, seed_queue, cv=folds, method="decision_function"
    )
    urgency_oof = cross_val_predict(
        urgency_lr, examples, seed_urgent, cv=folds, method="decision_function"
    )
    queue_lr.fit(examples, seed_queue)
    urgency_lr.fit(examples, seed_urgent)
    platt = LogisticRegression(C=1e6).fit(urgency_oof[:, None], seed_urgent)
    scale = best_scale(queue_oof, seed_queue)
    oof_probabilities = softmax(queue_oof * scale)
    urgency_calibrated = platt.predict_proba(urgency_oof[:, None])[:, 1]

    def review(probabilities: np.ndarray, truth: np.ndarray) -> float:
        correct = probabilities.argmax(axis=1) == truth
        chosen = choose_review_threshold(probabilities.max(axis=1), correct, cfg.review_target)
        return float(chosen.threshold)

    params = SemanticParams(
        labels=labels,
        neighbours=cfg.neighbours,
        temperature=cfg.temperature,
        familiar_from=cfg.familiar_from,
        familiar_full=cfg.familiar_full,
        queue_scale=scale,
        urgency_calibration=(float(platt.coef_[0, 0]), float(platt.intercept_[0])),
        review_unfamiliar=review(oof_probabilities, seed_queue),
        examples=[
            {"text": text, "queue": row["queue"], "urgent": bool(int(row["urgent"]))}
            for text, row in zip(seed_texts, seed, strict=True)
        ],
        version=f"semantic-{git_commit()}",
        embedding_model=settings.rag.embedding_model,
    )
    model = SemanticModel(
        params,
        past=past,
        past_queue=np.array([index[t.label] for t in train]),
        past_high=np.array([label_from_priority(t.priority, positives) for t in train]),
        examples=examples,
        queue_weights=np.c_[queue_lr.coef_, queue_lr.intercept_],
        urgency_weights=np.r_[urgency_lr.coef_[0], urgency_lr.intercept_],
    )

    # Familiar tickets: stack the risk model with the similar past tickets on validation.
    val_queue = np.array([index[t.label] for t in validation])
    val_high = np.array([label_from_priority(t.priority, positives) for t in validation])
    evidence = model.evidence(val)
    risk_model = RiskModel.load(resolve_model_dir(settings, "risk").path)
    base = np.array(
        [
            s.escalation_probability
            for s in risk_model.score_texts(
                [t.text for t in validation], [t.ticket_type for t in validation]
            )
        ]
    )
    stack = LogisticRegression(C=1e6).fit(
        np.c_[logit(base), soft_share(evidence.similar_high)], val_high
    )
    a, b = (float(x) for x in stack.coef_[0])
    c = float(stack.intercept_[0])
    familiar = sigmoid(a * logit(base) + b * soft_share(evidence.similar_high) + c)
    blended = evidence.gate * familiar + (1 - evidence.gate) * evidence.urgency

    # One alert threshold for both kinds of ticket, each kind counting equally.
    ratio = settings.risk.alert_cost_ratio
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    def cost(truth: np.ndarray, scores: np.ndarray, t: float) -> float:
        alert = scores >= t
        return float(
            (ratio * ((truth == 1) & ~alert).sum() + ((truth == 0) & alert).sum()) / len(truth)
        )

    total = [cost(val_high, blended, t) + cost(seed_urgent, urgency_calibrated, t) for t in grid]
    threshold = float(grid[int(np.argmin(total))])
    model.params = replace(
        params,
        risk_stack=(a, b, c),
        review_familiar=review(evidence.queue, val_queue),
        risk_threshold=threshold,
    )

    queue_hits = evidence.queue.argmax(axis=1) == val_queue
    summary = {
        "validation": {
            "queue_accuracy": round(float(queue_hits.mean()), 4),
            "risk": binary_metrics(val_high, blended, settings.risk.precision_target),
            "risk_model_alone": binary_metrics(val_high, base, settings.risk.precision_target),
            "threshold_on_validation_only": choose_threshold(val_high, blended, miss_cost=ratio),
        },
        "retail_out_of_fold": {
            "tickets": len(seed),
            "queue_accuracy": round(float((oof_probabilities.argmax(1) == seed_queue).mean()), 4),
            "risk": binary_metrics(seed_urgent, urgency_calibrated, settings.risk.precision_target),
        },
        "past_tickets": len(train),
        "train_seconds": round(time.perf_counter() - started, 1),
    }
    out_dir = settings.resolve(cfg.output_dir)
    model.save(out_dir, summary)
    with tracked_run(settings, "semantic", run_name="semantic-model") as mlflow:
        if mlflow:
            mlflow.log_params({k: v for k, v in cfg.model_dump().items() if "path" not in k})
            mlflow.log_metrics(
                {
                    "val_queue_accuracy": summary["validation"]["queue_accuracy"],
                    "val_risk_pr_auc": summary["validation"]["risk"]["pr_auc"],
                    "retail_oof_queue_accuracy": summary["retail_out_of_fold"]["queue_accuracy"],
                    "retail_oof_risk_roc_auc": summary["retail_out_of_fold"]["risk"]["roc_auc"],
                    "alert_threshold": threshold,
                }
            )
            mlflow.log_artifacts(str(out_dir), artifact_path="semantic_model")
    p, val_summary, retail = model.params, summary["validation"], summary["retail_out_of_fold"]
    print(
        f"Validation: queue accuracy {val_summary['queue_accuracy']:.4f}, risk PR-AUC "
        f"{val_summary['risk']['pr_auc']:.4f} (risk model alone "
        f"{val_summary['risk_model_alone']['pr_auc']:.4f})"
    )
    print(
        f"Retail, out of fold: queue accuracy {retail['queue_accuracy']:.4f}, risk ROC-AUC "
        f"{retail['risk']['roc_auc']:.4f}"
    )
    print(
        f"Review below {p.review_familiar:.3f} for familiar and {p.review_unfamiliar:.3f} for "
        f"retail tickets, alert from {threshold:.2f}"
    )
    print(f"Saved to {out_dir} in {summary['train_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
