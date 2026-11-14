"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """A small knowledge base with one file of each supported text format."""
    kb = tmp_path / "kb"
    (kb / "billing").mkdir(parents=True)
    (kb / "billing" / "refunds.md").write_text(
        "# Refunds\n\nRefunds are issued within 3 business days of the return arriving. "
        "Card refunds take 5 to 10 business days to appear on your statement.\n",
        encoding="utf-8",
    )
    (kb / "shipping.txt").write_text(
        "SHIPPING\n\nStandard shipping takes 4 to 6 business days and is free on orders over $50. "
        "Express shipping takes 2 business days.\n",
        encoding="utf-8",
    )
    (kb / "account.html").write_text(
        "<html><head><title>Account</title><script>var x = 1;</script></head>"
        "<body><h1>Password reset</h1><p>The password reset link is valid for 30 minutes.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (kb / "notes.json").write_text("{}", encoding="utf-8")  # unsupported, must be ignored
    return kb
