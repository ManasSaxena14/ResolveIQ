"""Publish the full web app as a Hugging Face Space, without Docker.

Steps:

1. build the React app (npm run build in frontend/), unless --skip-build;
2. stage the files the Space needs in a temporary folder: the package, configs,
   the help center, the built app, space.py and requirements.txt; never .env,
   models, logs, local data or sync-tool duplicates;
3. write the Space README with its settings header (Gradio SDK, space.py);
4. create the Space if needed and upload the folder.

Before the first run: publish both models with scripts/publish_models.py, log in
with `hf auth login`, and set the Space variables and the language model key as
described in docs/deployment.md (or pass --set-variables to set the variables).

Usage:
    python scripts/deploy_space.py --space <user>/resolveiq
    python scripts/deploy_space.py --space <user>/resolveiq --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from resolveiq.config import PROJECT_ROOT
from resolveiq.rag.loaders import is_sync_duplicate

INCLUDE = [
    "space.py",
    "requirements.txt",
    "pyproject.toml",
    "LICENSE",
    "configs",
    "src/resolveiq",
    "data/kb",
    "frontend/dist",
]
SKIP_NAMES = {"__pycache__", ".DS_Store", ".env"}

SPACE_README = """---
title: ResolveIQ
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: {gradio_version}
app_file: space.py
python_version: "3.12"
license: mit
short_description: Draft replies, route tickets and predict escalation risk
---

# ResolveIQ

Support ticket copilot: a grounded reply draft, the routing queue and the escalation
risk for every ticket, with the evidence for each. The app below is the React web app
served by the FastAPI service.

Source code, evaluation and documentation: {repo_url}

Demo on synthetic data. Do not enter personal data.
"""


def gradio_version() -> str:
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("gradio=="):
            return line.split("==", 1)[1].strip()
    raise SystemExit("requirements.txt does not pin gradio")


def _wanted(path: Path) -> bool:
    return not (
        path.name in SKIP_NAMES
        or path.name.startswith(".")
        or is_sync_duplicate(path.name)
        or path.suffix in {".pyc", ".map"}
    )


def stage(
    destination: Path, root: Path = PROJECT_ROOT, include: Iterable[str] = INCLUDE
) -> list[str]:
    """Copy the Space's files into ``destination``; returns their relative paths."""
    copied: list[str] = []
    for entry in include:
        source = root / entry
        if not source.exists():
            raise FileNotFoundError(f"{entry} is missing. Build or create it first.")
        files = (
            [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
        )
        for path in files:
            relative = path.relative_to(root)
            if not all(_wanted(Path(part)) for part in relative.parts):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(relative.as_posix())
    return copied


def write_readme(destination: Path, repo_url: str) -> None:
    text = SPACE_README.format(gradio_version=gradio_version(), repo_url=repo_url)
    (destination / "README.md").write_text(text, encoding="utf-8")


def build_frontend() -> None:
    subprocess.run(["npm", "ci"], cwd=PROJECT_ROOT / "frontend", check=True)
    subprocess.run(["npm", "run", "build"], cwd=PROJECT_ROOT / "frontend", check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--space", required=True, help="Space id, e.g. user/resolveiq")
    parser.add_argument("--repo-url", default="https://github.com/ManasSaxena14/ResolveIQ")
    parser.add_argument("--skip-build", action="store_true", help="Use the existing build")
    parser.add_argument("--dry-run", action="store_true", help="Stage and list, do not upload")
    parser.add_argument(
        "--set-variables",
        nargs=2,
        metavar=("CLASSIFIER_REPO", "RISK_REPO"),
        help="Also set RESOLVEIQ_CLASSIFIER_REPO and RESOLVEIQ_RISK_REPO on the Space",
    )
    args = parser.parse_args()

    if not args.skip_build:
        build_frontend()
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "space"
        files = stage(staged)
        write_readme(staged, args.repo_url)
        size = sum((staged / f).stat().st_size for f in files) / 1e6
        print(f"Staged {len(files) + 1} files ({size:.1f} MB)")
        if args.dry_run:
            for name in ["README.md", *files]:
                print(" ", name)
            return 0

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)
        if args.set_variables:
            classifier_repo, risk_repo = args.set_variables
            api.add_space_variable(args.space, "RESOLVEIQ_CLASSIFIER_REPO", classifier_repo)
            api.add_space_variable(args.space, "RESOLVEIQ_RISK_REPO", risk_repo)
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=str(staged),
            commit_message="Deploy ResolveIQ web app",
        )
    print(f"Deployed to https://huggingface.co/spaces/{args.space}")
    print("Add the language model key as a Space secret if you have not yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
