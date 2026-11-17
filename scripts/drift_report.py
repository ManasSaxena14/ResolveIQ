"""Compare recent requests with the training data and write a drift report.

Step 1, once after training: build the reference profile by running the
production models over the validation split (no ticket text is saved)::

    python scripts/drift_report.py --build-reference

Step 2, whenever needed: compare the newest rows of the request log with the
reference, apply the retrain rule and write an HTML and a JSON report::

    python scripts/drift_report.py [--log logs/requests.db] [--name drift_report]
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # see docs/risk_model.md

import argparse
import csv
import json
from pathlib import Path

from resolveiq.config import Settings, load_config
from resolveiq.monitoring.drift import detect_drift, render_html, retrain_decision
from resolveiq.monitoring.logging import RequestLog


def build_reference(settings: Settings, split: str) -> Path:
    from resolveiq.classifier.data import load_split
    from resolveiq.classifier.predict import TicketClassifier
    from resolveiq.registry import resolve_model_dir
    from resolveiq.risk.predict import RiskModel

    tickets = load_split(settings.resolve(settings.classifier.processed_dir), split)
    classifier = TicketClassifier.load(resolve_model_dir(settings, "classifier").path)
    risk_model = RiskModel.load(resolve_model_dir(settings, "risk").path)
    texts = [t.text for t in tickets]
    print(f"Scoring {len(texts)} {split} tickets")
    predictions = classifier.predict(texts)
    scores = risk_model.score_texts(texts, [t.ticket_type for t in tickets])

    path = settings.resolve(settings.monitoring.reference_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text_length", "category", "confidence", "risk_score"])
        for text, prediction, score in zip(texts, predictions, scores, strict=True):
            writer.writerow(
                [len(text), prediction.label, prediction.confidence, score.escalation_probability]
            )
    print(f"Wrote {path}")
    return path


def read_reference(path: Path, numeric: list[str]) -> dict[str, list]:
    if not path.is_file():
        raise SystemExit(f"{path} not found. Run with --build-reference first.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        column: [float(r[column]) if column in numeric else r[column] for r in rows]
        for column in rows[0]
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--split", default="validation", help="Split used for the reference")
    parser.add_argument("--log", help="Request log to read (defaults to service.request_log)")
    parser.add_argument("--name", default="drift_report", help="Base name of the report files")
    parser.add_argument("--title", default="ResolveIQ drift report")
    args = parser.parse_args()

    settings = load_config(args.config)
    monitoring = settings.monitoring
    if args.build_reference:
        build_reference(settings, args.split)
        return 0

    reference = read_reference(
        settings.resolve(monitoring.reference_path), monitoring.numeric_features
    )
    log = RequestLog(Path(args.log) if args.log else settings.resolve(settings.service.request_log))
    rows = log.recent(monitoring.recent_requests)
    if not rows:
        print("The request log is empty; nothing to compare.", file=sys.stderr)
        return 1
    current = {name: [row[name] for row in rows] for name in reference}

    report = detect_drift(
        reference,
        current,
        monitoring.numeric_features,
        monitoring.categorical_features,
        monitoring.psi_threshold,
        monitoring.p_value,
    )
    feedback = log.feedback_stats()
    decision = retrain_decision(
        report,
        feedback.total,
        feedback.down_rate,
        monitoring.retrain_drift_share,
        monitoring.retrain_thumbs_down_rate,
        monitoring.min_feedback,
    )
    notes = [
        f"Reference: production models scored on the {args.split} split.",
        f"Current: the newest {len(rows)} rows of the request log.",
        f"Feedback votes: {feedback.total} ({feedback.down} negative).",
    ]
    out_dir = settings.resolve(monitoring.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{args.name}.html"
    html_path.write_text(
        render_html(report, decision, reference, current, args.title, notes), encoding="utf-8"
    )
    summary = {
        **report.as_dict(),
        "feedback": feedback.as_dict(),
        "retrain": decision.retrain,
        "retrain_reasons": decision.reasons,
    }
    (out_dir / f"{args.name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for feature in report.features:
        flag = "DRIFT" if feature.drifted else "ok"
        print(f"{feature.name:<12} PSI {feature.psi:7.4f}  p {feature.p_value:.4g}  {flag}")
    print("Retrain:", "yes" if decision.retrain else "no", *decision.reasons, sep="\n  ")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
