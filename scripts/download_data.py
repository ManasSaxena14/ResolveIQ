"""Download the labeled ticket dataset into data/raw/.

The file is fetched from a pinned dataset revision on the Hugging Face Hub, so
every run downloads exactly the same data, and its SHA-256 checksum is printed
for the record in data/README.md.

Usage:
    python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys

import httpx

from resolveiq.config import load_config


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Download even if the file exists")
    args = parser.parse_args()

    dataset = load_config().classifier.dataset
    target = load_config().resolve(dataset.raw_path)
    if target.exists() and not args.force:
        print(f"{target} already exists (sha256 {sha256(target)}). Use --force to download again.")
        return 0

    url = (
        f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/"
        f"{dataset.revision}/{dataset.filename}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {dataset.repo_id}@{dataset.revision[:7]}/{dataset.filename}")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for block in response.iter_bytes():
                    handle.write(block)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    partial.replace(target)
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB, sha256 {sha256(target)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
