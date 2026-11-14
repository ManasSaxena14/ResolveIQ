"""Build the knowledge base index and log the build to MLflow.

Usage:
    python scripts/build_index.py [--config PATH] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import logging

from resolveiq.config import load_config
from resolveiq.rag.ingest import build_index
from resolveiq.tracking import tracked_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file (defaults to configs/config.yaml)")
    parser.add_argument("--no-mlflow", action="store_true", help="Do not log the build to MLflow")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("resolveiq").setLevel(logging.INFO)
    settings = load_config(args.config)
    if args.no_mlflow:
        settings.tracking.enabled = False

    run = tracked_run(settings, experiment="rag", run_name="build-index", tags={"stage": "index"})
    with run as mlflow:
        summary = build_index(settings)
        if mlflow:
            mlflow.log_params(
                {
                    "embedding_model": summary.embedding_model,
                    "chunk_tokens": summary.chunk_tokens,
                    "chunk_overlap": summary.chunk_overlap,
                }
            )
            mlflow.log_metrics(
                {
                    "files": summary.files,
                    "pages": summary.pages,
                    "chunks": summary.chunks,
                    "build_seconds": summary.build_seconds,
                }
            )

    print("Index built")
    for key, value in summary.as_dict().items():
        print(f"  {key:<16} {value}")


if __name__ == "__main__":
    main()
