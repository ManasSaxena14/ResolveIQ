"""Publish the production classifier to the Hugging Face Hub.

Copies docs/model_card_classifier.md into the model folder as README.md and
uploads the folder. Requires a Hugging Face account and a write token, either
from `hf auth login` or the HF_TOKEN environment variable.

Usage:
    python scripts/publish_classifier.py --repo-id <username>/resolveiq-ticket-classifier
"""

from __future__ import annotations

import argparse
import shutil
import sys

from resolveiq.config import PROJECT_ROOT, load_config

MODEL_CARD = PROJECT_ROOT / "docs" / "model_card_classifier.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Target repository, e.g. user/name")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    settings = load_config()
    model_dir = settings.resolve(settings.classifier.output_dir) / "production"
    if not (model_dir / "config.json").is_file():
        print(
            f"{model_dir} has no model. Run scripts/evaluate_classifier.py first.", file=sys.stderr
        )
        return 1
    shutil.copyfile(MODEL_CARD, model_dir / "README.md")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(model_dir),
        commit_message="Upload ResolveIQ ticket classifier",
    )
    print(f"Published to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
