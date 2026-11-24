"""Final evaluation of the semantic model: the test split once, and the retail holdout once.

Compares it with the models it replaces, the fine-tuned transformer and the XGBoost risk
model alone. The test split is also broken down by how close each ticket is to a training
ticket, because most test tickets are near-copies of training tickets and the overall
score mainly rewards recognizing them.

Writes results/semantic/metrics.json and comparison.md.

Usage:
    python scripts/evaluate_semantic.py
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # see docs/risk_model.md

import csv
import json
from typing import Any

import numpy as np

from resolveiq.classifier.data import build_text, load_labels, load_split
from resolveiq.classifier.metrics import classification_metrics
from resolveiq.config import load_config
from resolveiq.risk.evaluate import alert_stats, binary_metrics, pr_auc
from resolveiq.risk.features import label_from_priority
from resolveiq.service import load_decisions

BANDS = ((0.0, 0.90), (0.90, 0.95), (0.95, 0.98), (0.98, 1.01))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return round(float(roc_auc_score(y, p)), 4)


def wilson(hits: int, total: int) -> list[float]:
    """95% confidence interval of a proportion."""
    z, p = 1.96, hits / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5 / (1 + z * z / total)
    return [round(center - half, 3), round(center + half, 3)]


def on_test(classifier: Any, risk: Any, tickets: list, labels: list[str], y: np.ndarray) -> dict:
    texts = [t.text for t in tickets]
    predictions = classifier.predict(texts)
    scores = risk.score_texts(texts, [t.ticket_type for t in tickets])
    p = np.array([s.escalation_probability for s in scores])
    hits = np.array([a.label == t.label for a, t in zip(predictions, tickets, strict=True)])
    sure = np.array([not a.needs_review for a in predictions])
    queue = classification_metrics(
        [t.label for t in tickets], [a.label for a in predictions], labels
    )
    stats = alert_stats(y, p, risk.threshold)
    return {
        "queue_accuracy": queue["accuracy"],
        "queue_macro_f1": queue["macro_f1"],
        "sent_on_without_review": round(float(sure.mean()), 4),
        "accuracy_without_review": round(float(hits[sure].mean()), 4) if sure.any() else None,
        "risk": binary_metrics(y, p, 0.70),
        "alert_caught": stats.caught,
        "alert_rate": stats.alert_rate,
        "_hits": hits,
        "_p": p,
    }


def on_retail(classifier: Any, risk: Any, rows: list[dict[str, str]]) -> dict:
    predictions = [classifier.predict_ticket(r["body"], r["subject"]) for r in rows]
    scores = [risk.score(r["subject"], r["body"]) for r in rows]
    hits = np.array(
        [p.label in r["accepted_queues"].split(";") for p, r in zip(predictions, rows, strict=True)]
    )
    sure = np.array([not p.needs_review for p in predictions])
    urgent = np.array([int(r["urgent"]) for r in rows])
    p = np.array([s.escalation_probability for s in scores])
    alert = np.array([s.alert for s in scores])
    return {
        "tickets": len(rows),
        "queue_right": int(hits.sum()),
        "queue_accuracy_95ci": wilson(int(hits.sum()), len(rows)),
        "sent_on_without_review": int(sure.sum()),
        "right_without_review": int(hits[sure].sum()),
        "urgent": int(urgent.sum()),
        "risk_roc_auc": roc_auc(urgent, p),
        "urgent_flagged": int(alert[urgent == 1].sum()),
        "calm_flagged": int(alert[urgent == 0].sum()),
    }


def main() -> int:
    settings = load_config()
    cfg = settings.semantic
    tickets = load_split(settings.resolve(settings.classifier.processed_dir), "test")
    labels = load_labels(settings.resolve(settings.classifier.processed_dir))
    positives = settings.risk.label.positive_values
    y = np.array([label_from_priority(t.priority, positives) for t in tickets])
    with settings.resolve(cfg.testset_path).open(newline="", encoding="utf-8") as handle:
        retail = list(csv.DictReader(handle))

    classifier, risk, versions = load_decisions(settings)
    if not hasattr(classifier, "model") or not hasattr(classifier, "encoder"):
        print("Train the semantic model first: make semantic", file=sys.stderr)
        return 1
    results: dict[str, Any] = {"versions": versions, "test": {}, "retail": {}}
    evidence = classifier.model.evidence(classifier.encoder.encode([t.text for t in tickets]))

    settings.service.decisions = "transformer"
    before, before_risk, _ = load_decisions(settings)
    results["test"]["before"] = on_test(before, before_risk, tickets, labels, y)
    results["test"]["semantic"] = on_test(classifier, risk, tickets, labels, y)
    results["bands"] = []
    for low, high in BANDS:
        mask = (evidence.familiarity >= low) & (evidence.familiarity < high)
        row: dict[str, Any] = {"from": low, "to": min(high, 1.0), "tickets": int(mask.sum())}
        for name, values in results["test"].items():
            row[f"{name}_queue_accuracy"] = round(float(values["_hits"][mask].mean()), 4)
            row[f"{name}_risk_pr_auc"] = round(pr_auc(y[mask], values["_p"][mask]), 4)
        results["bands"].append(row)
    for values in results["test"].values():
        values.pop("_hits")
        values.pop("_p")
    texts = [build_text(r["subject"], r["body"]) for r in retail]
    familiarity = classifier.model.evidence(classifier.encoder.encode(texts)).familiarity
    results["retail_max_familiarity"] = round(float(familiarity.max()), 4)
    for split in ("dev", "holdout"):
        rows = [r for r in retail if r["split"] == split]
        results["retail"][split] = {
            "before": on_retail(before, before_risk, rows),
            "semantic": on_retail(classifier, risk, rows),
        }

    out_dir = settings.resolve(settings.paths.results_dir) / "semantic"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "comparison.md").write_text(markdown(results), encoding="utf-8")
    print(markdown(results))
    return 0


def markdown(results: dict[str, Any]) -> str:
    lines = ["# Semantic model compared with the models it replaces", ""]
    lines += ["## Hand-written retail tickets", ""]
    lines += ["| Half | Model | Right queue | Sent on without review, right | Risk ROC-AUC |"
              " Urgent flagged | Calm flagged |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for split, models in results["retail"].items():
        for name, r in models.items():
            calm = r["tickets"] - r["urgent"]
            lines.append(
                f"| {split} | {name} | {r['queue_right']} of {r['tickets']} | "
                f"{r['right_without_review']} of {r['sent_on_without_review']} | "
                f"{r['risk_roc_auc']:.4f} | {r['urgent_flagged']} of {r['urgent']} | "
                f"{r['calm_flagged']} of {calm} |"
            )
    lines += ["", "## Test split of the public dataset", ""]
    lines += ["| Model | Queue accuracy | Macro F1 | Sent on without review | Accuracy of those |"
              " Risk PR-AUC | Brier |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, r in results["test"].items():
        lines.append(
            f"| {name} | {r['queue_accuracy']:.4f} | {r['queue_macro_f1']:.4f} | "
            f"{r['sent_on_without_review']:.1%} | {r['accuracy_without_review']:.1%} | "
            f"{r['risk']['pr_auc']:.4f} | {r['risk']['brier']:.4f} |"
        )
    lines += ["", "## Test split by similarity to the closest training ticket", ""]
    lines += ["| Similarity | Tickets | Queue accuracy before | Queue accuracy semantic |"
              " Risk PR-AUC before | Risk PR-AUC semantic |", "|---|---|---|---|---|---|"
              ]  # fmt: skip
    for row in results["bands"]:
        lines.append(
            f"| {row['from']:.2f} to {row['to']:.2f} | {row['tickets']} | "
            f"{row['before_queue_accuracy']:.4f} | {row['semantic_queue_accuracy']:.4f} | "
            f"{row['before_risk_pr_auc']:.4f} | {row['semantic_risk_pr_auc']:.4f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
