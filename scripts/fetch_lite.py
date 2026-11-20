"""Download the lite bundle at build time on a host such as Render.

Runs during the build, so the models are part of the deployed files and a
sleeping service never downloads anything when it wakes up. The bundle comes
from the Hugging Face repository in RESOLVEIQ_LITE_REPO (published with
scripts/publish_models.py lite).

Usage:
    RESOLVEIQ_RUNTIME=lite RESOLVEIQ_LITE_REPO=<user>/resolveiq-lite python scripts/fetch_lite.py
"""

from __future__ import annotations

import sys

from resolveiq.config import load_config
from resolveiq.lite import lite_ready
from resolveiq.registry import fetch_from_hub


def main() -> int:
    settings = load_config()
    settings.service.runtime = "lite"
    lite_dir = settings.resolve(settings.service.lite_dir)
    fetched = fetch_from_hub(settings)
    if not lite_ready(lite_dir):
        print(
            f"No lite bundle in {lite_dir}. Set RESOLVEIQ_LITE_REPO to the Hugging Face "
            "repository published with scripts/publish_models.py lite.",
            file=sys.stderr,
        )
        return 1
    print("Downloaded the lite bundle" if fetched else f"Lite bundle already in {lite_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
