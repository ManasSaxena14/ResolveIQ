"""Clean, deduplicate and split the ticket dataset.

Reads data/raw/, writes train, validation and test CSV files plus the label
list to data/processed/, and saves a data profile (row counts, class balance,
token lengths) to results/classifier/data_profile.json.

Usage:
    python scripts/prepare_tickets.py [--config PATH]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from resolveiq.classifier.data import (
    class_counts,
    label_names,
    load_tickets,
    save_splits,
    split_tickets,
)
from resolveiq.config import load_config


def token_length_profile(texts: list[str], model_name: str) -> dict[str, float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = np.array([len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"]])
    return {
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        "share_over_128": round(float((lengths > 128).mean()), 4),
        "share_over_256": round(float((lengths > 256).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to a config file")
    args = parser.parse_args()

    settings = load_config(args.config)
    cfg = settings.classifier
    raw_path = settings.resolve(cfg.dataset.raw_path)
    if not raw_path.is_file():
        raise SystemExit(f"{raw_path} not found. Run scripts/download_data.py first.")

    tickets, report = load_tickets(raw_path, cfg.dataset.language, cfg.dataset.label_column)
    splits = split_tickets(tickets, cfg.split.validation, cfg.split.test, cfg.split.seed)
    labels = label_names(tickets)
    out_dir = settings.resolve(cfg.processed_dir)
    save_splits(splits, labels, out_dir)

    profile = {
        "dataset": {
            "repo_id": cfg.dataset.repo_id,
            "revision": cfg.dataset.revision,
            "filename": cfg.dataset.filename,
            "language": cfg.dataset.language,
            "label_column": cfg.dataset.label_column,
        },
        "cleaning": report.as_dict(),
        "split_sizes": {name: len(items) for name, items in splits.items()},
        "class_counts": {name: class_counts(items) for name, items in splits.items()},
        "token_lengths": token_length_profile([t.text for t in tickets], cfg.model_name),
    }
    profile_path = settings.resolve(settings.paths.results_dir) / "classifier" / "data_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(
        json.dumps({k: profile[k] for k in ("cleaning", "split_sizes", "token_lengths")}, indent=2)
    )
    print(f"\nLabels ({len(labels)}): {', '.join(labels)}")
    print(f"Splits written to {out_dir}")
    print(f"Profile written to {profile_path}")


if __name__ == "__main__":
    main()
