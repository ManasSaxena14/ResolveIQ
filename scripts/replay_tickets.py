"""Send held-out tickets through the copilot to fill a request log.

Used to produce the sample drift reports and the in-process latency numbers.
The language model is not called, so the draft step is skipped and timings
cover retrieval, classification and risk scoring only.

Usage:
    python scripts/replay_tickets.py --split test --limit 400 --log logs/replay_test.db
    python scripts/replay_tickets.py --split test --queue "Technical Support" \
        --queue "IT Support" --limit 400 --log logs/replay_shifted.db
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # see docs/risk_model.md

import argparse
import json
import random
from pathlib import Path

import numpy as np

from resolveiq.classifier.data import load_split
from resolveiq.config import load_config
from resolveiq.monitoring.logging import RequestLog
from resolveiq.schemas import AnalyzeRequest
from resolveiq.service import Copilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--queue", action="append", help="Only replay tickets of these queues")
    parser.add_argument("--log", required=True, help="SQLite file for the replayed requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary", help="Write latency percentiles to this JSON file")
    args = parser.parse_args()

    settings = load_config(args.config)
    settings.service.request_log = Path(args.log).resolve()
    log_path = settings.service.request_log
    if log_path.exists():
        log_path.unlink()

    tickets = load_split(settings.resolve(settings.classifier.processed_dir), args.split)
    if args.queue:
        tickets = [t for t in tickets if t.label in set(args.queue)]
    random.Random(args.seed).shuffle(tickets)
    tickets = tickets[: args.limit]

    copilot = Copilot.from_settings(settings, with_llm=False)
    # The first call warms up the models and is not counted.
    copilot.analyze(AnalyzeRequest(text="Warm-up ticket about a late order"))
    copilot.request_log = RequestLog(log_path)

    totals = []
    for i, ticket in enumerate(tickets, start=1):
        text = ticket.text[: copilot.max_text_chars]
        result = copilot.analyze(AnalyzeRequest(text=text, ticket_type=ticket.ticket_type))
        totals.append(result.response.timings_ms["total"])
        if i % 50 == 0:
            print(f"{i}/{len(tickets)}")

    summary = {
        "split": args.split,
        "queues": args.queue or "all",
        "requests": len(totals),
        "p50_ms": round(float(np.percentile(totals, 50)), 1),
        "p95_ms": round(float(np.percentile(totals, 95)), 1),
        "steps": "retrieval, classifier, risk (no language model)",
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
