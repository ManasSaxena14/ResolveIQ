"""Entry point for a Hugging Face Space (needs a paid Hugging Face plan, see below).

Hugging Face now requires a paid plan to create Spaces that run code, so the free
deployments are Oracle Cloud and Render (docs/deployment.md). This file stays for
accounts that have the plan, and it also runs the web app and API locally.

Serves the React web app and the API from one process on the port the Space
expects, without Docker:

1. downloads the published models from the Hugging Face Hub when the local
   folders are empty (RESOLVEIQ_CLASSIFIER_REPO, RESOLVEIQ_RISK_REPO);
2. builds the knowledge base index on first start;
3. trusts the Space proxy's X-Forwarded-For header for per-visitor rate limits;
4. starts uvicorn on 0.0.0.0 and $PORT (7860 by default).

The built frontend (frontend/dist) is uploaded with the code by
scripts/deploy_space.py, because a Space has no Node.js to build it. Run it
locally with ``python space.py`` to try exactly what the Space runs.
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

import uvicorn

from resolveiq.api.limits import AccessPolicy
from resolveiq.api.main import WEB_DIR, create_app
from resolveiq.config import get_settings
from resolveiq.rag.ingest import build_index, read_manifest
from resolveiq.registry import fetch_from_hub
from resolveiq.service import Copilot

logger = logging.getLogger("resolveiq.space")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    on_space = bool(os.environ.get("SPACE_ID"))
    if on_space:
        # The Space proxy sets X-Forwarded-For; without trusting it every visitor
        # would share the proxy's address and one rate limit.
        settings.service.rate_limit.trust_forwarded_for = True
    if not (WEB_DIR / "index.html").is_file():
        logger.warning("%s has no built web app; only the API will be served", WEB_DIR)

    fetch_from_hub(settings)
    if read_manifest(settings.resolve(settings.paths.index_dir)) is None:
        logger.info("No knowledge base index found, building it")
        settings.tracking.enabled = False
        build_index(settings)

    app = create_app(
        loader=lambda: Copilot.from_settings(settings),
        access=AccessPolicy.from_config(settings.service.rate_limit),
    )
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=on_space)


if __name__ == "__main__":
    main()
