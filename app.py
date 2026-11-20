"""Gradio web UI entry point, a lighter alternative to the React web app.

Run locally with ``python app.py`` and open http://127.0.0.1:7860.

On first start the knowledge base index is built if it does not exist yet.
Models are loaded from the MLflow registry or the local ``models/`` folders.
On a hosted demo they are first downloaded from the Hugging Face Hub, see
docs/deployment.md.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    # PyTorch and XGBoost bundle different OpenMP runtimes on macOS; see
    # docs/risk_model.md. Must run before either library is imported.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import logging

from resolveiq.config import get_settings
from resolveiq.rag.ingest import build_index, read_manifest
from resolveiq.registry import fetch_from_hub
from resolveiq.service import Copilot
from resolveiq.ui.gradio_app import build_demo, launch_options


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if read_manifest(settings.resolve(settings.paths.index_dir)) is None:
        logging.info("No knowledge base index found, building it")
        settings.tracking.enabled = False
        build_index(settings)
    fetch_from_hub(settings)
    copilot = Copilot.from_settings(settings)
    demo = build_demo(copilot, settings.service.source_base_url)
    demo.queue(default_concurrency_limit=2).launch(
        max_file_size=settings.service.max_upload_bytes, **launch_options()
    )


if __name__ == "__main__":
    main()
