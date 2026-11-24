"""Compare classifier and risk results across ticket types on the test split.

The dataset has no customer attributes, so ticket type is the only group the
data allows. A large gap between groups is reported in the evaluation report.

Usage:
    python scripts/evaluate_groups.py
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # see docs/risk_model.md

import csv
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from resolveiq.classifier.data import load_split
from resolveiq.config import load_config
from resolveiq.risk.evaluate import pr_auc
from resolveiq.risk.features import label_from_priority
from resolveiq.service import load_decisions


def main() -> int:
    settings = load_config()
    tickets = load_split(settings.resolve(settings.classifier.processed_dir), "test")
    classifier, risk_model, _ = load_decisions(settings)

    texts = [t.text for t in tickets]
    predicted = [p.label for p in classifier.predict(texts)]
    scores = risk_model.score_texts(texts, [t.ticket_type for t in tickets])
    positives = settings.risk.label.positive_values
    labels = [label_from_priority(t.priority, positives) for t in tickets]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, ticket in enumerate(tickets):
        groups[ticket.ticket_type or "(none)"].append(i)
    groups["All"] = list(range(len(tickets)))

    rows = []
    for name, index in sorted(groups.items(), key=lambda item: (item[0] == "All", item[0])):
        truth = [tickets[i].label for i in index]
        guess = [predicted[i] for i in index]
        y = np.array([labels[i] for i in index])
        p = np.array([scores[i].escalation_probability for i in index])
        alerts = np.array([scores[i].alert for i in index])
        rows.append(
            {
                "group": name,
                "tickets": len(index),
                "classifier_accuracy": round(accuracy_score(truth, guess), 4),
                "classifier_macro_f1": round(
                    f1_score(truth, guess, average="macro", zero_division=0), 4
                ),
                "high_priority_rate": round(float(y.mean()), 4),
                "risk_pr_auc": round(pr_auc(y, p), 4) if 0 < y.sum() < len(y) else None,
                "alert_rate": round(float(alerts.mean()), 4),
                "caught": round(float(alerts[y == 1].mean()), 4) if y.sum() else None,
            }
        )

    out_dir = settings.resolve(settings.paths.results_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "group_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
