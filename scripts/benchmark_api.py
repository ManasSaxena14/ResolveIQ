"""Measure response time and memory of a running API.

Start the API first (``uvicorn resolveiq.api.main:app --port 8000``), then::

    python scripts/benchmark_api.py --url http://127.0.0.1:8000 --requests 50 --pid <PID>

Sends held-out test tickets to /analyze one at a time and reports the median
and 95th percentile response time. With ``--pid`` the resident memory of the
server process is sampled at idle and after the run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from resolveiq.classifier.data import load_split
from resolveiq.config import load_config


def rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    import psutil

    return round(psutil.Process(pid).memory_info().rss / 1e6, 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--pid", type=int, help="Server process id, for memory use")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", help="Write the results to this JSON file")
    args = parser.parse_args()

    settings = load_config()
    tickets = load_split(settings.resolve(settings.classifier.processed_dir), "test")
    random.Random(args.seed).shuffle(tickets)
    client = httpx.Client(base_url=args.url, timeout=120)
    health = client.get("/health").json()
    idle = rss_mb(args.pid)

    latencies, statuses, peak = [], {}, idle
    for ticket in tickets[: args.requests]:
        payload = {"text": ticket.text[: settings.service.max_text_chars]}
        started = time.perf_counter()
        response = client.post("/analyze", json=payload)
        latencies.append((time.perf_counter() - started) * 1000)
        statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
        current = rss_mb(args.pid)
        if current is not None:
            peak = max(peak or 0, current)

    result = {
        "url": args.url,
        "requests": len(latencies),
        "status_codes": {str(k): v for k, v in sorted(statuses.items())},
        "p50_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_ms": round(float(np.percentile(latencies, 95)), 1),
        "memory_idle_mb": idle,
        "memory_peak_mb": peak,
        "llm_configured": health["components"].get("llm_configured"),
        "versions": health["versions"],
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
