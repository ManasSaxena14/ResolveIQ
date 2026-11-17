"""Publish a production model to the Hugging Face Hub.

The model card from docs/ is copied in as README.md, and sync-tool duplicates
are left out. Requires a Hugging Face account and a write token, either from
`hf auth login` or the HF_TOKEN environment variable.

Usage:
    python scripts/publish_models.py classifier --repo-id <user>/resolveiq-ticket-classifier
    python scripts/publish_models.py risk --repo-id <user>/resolveiq-escalation-risk

A hosted demo downloads the published models at startup when the
RESOLVEIQ_CLASSIFIER_REPO and RESOLVEIQ_RISK_REPO variables are set
(see docs/deployment.md).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from resolveiq.config import PROJECT_ROOT, load_config
from resolveiq.registry import META_FILES, clean_copy

CARDS = {"classifier": "model_card_classifier.md", "risk": "model_card_risk.md"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", choices=sorted(CARDS))
    parser.add_argument("--repo-id", required=True, help="Target repository, e.g. user/name")
    parser.add_argument("--private", action="store_true", help="Create a private repository")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    settings = load_config()
    field = "classifier_dir" if args.model == "classifier" else "risk_dir"
    model_dir = settings.resolve(getattr(settings.service, field))
    if not (model_dir / META_FILES[args.model]).is_file():
        print(f"{model_dir} has no trained model. Train and evaluate it first.", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staged = clean_copy(model_dir, Path(tmp) / "model")
        shutil.copyfile(PROJECT_ROOT / "docs" / CARDS[args.model], staged / "README.md")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staged),
            commit_message=f"Upload ResolveIQ {args.model} model",
        )
    print(f"Published to https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
