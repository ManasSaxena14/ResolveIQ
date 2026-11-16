"""Score a CSV of tickets and write them ranked by escalation risk.

The input CSV needs ``subject`` and ``body`` columns; ``type`` (Incident,
Problem, Request or Change) and ``id`` are optional. The output lists the id,
calibrated probability, alert flag and top three reasons, highest risk first.
Ticket text is not copied to the output.

Usage:
    python scripts/score_tickets.py INPUT.csv [--output results/risk/ranked.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from resolveiq.config import load_config
from resolveiq.risk.predict import RiskModel, score_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with subject and body columns")
    parser.add_argument("--output", type=Path, help="Where to write the ranked CSV")
    parser.add_argument("--config", help="Path to a config file")
    args = parser.parse_args()

    settings = load_config(args.config)
    model = RiskModel.load(settings.resolve(settings.risk.output_dir))
    output = args.output or args.input.with_name(f"{args.input.stem}_ranked.csv")
    count = score_csv(model, args.input, output)
    print(f"Scored {count} tickets; ranked list written to {output}")


if __name__ == "__main__":
    main()
