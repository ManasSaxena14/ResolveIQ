"""Text cleaning applied to pages before chunking."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from resolveiq.rag.types import Page

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"([a-z,;])\n([a-z])")
_SPACES = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

# A line must appear on at least this share of a document's pages, at the top
# or bottom, to be treated as a running header or footer.
_REPEAT_SHARE = 0.5
_EDGE_LINES = 2
_MIN_PAGES_FOR_HEADER_DETECTION = 3


def normalize_text(text: str) -> str:
    """Fix line-break artifacts and whitespace while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SOFT_BREAK.sub(r"\1 \2", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _repeated_edge_lines(pages: list[Page]) -> set[str]:
    """Find lines repeated at the top or bottom of many pages of one document."""
    if len(pages) < _MIN_PAGES_FOR_HEADER_DETECTION:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        counts.update(set(lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]))
    threshold = max(2, int(len(pages) * _REPEAT_SHARE))
    return {line for line, count in counts.items() if count >= threshold}


def _strip_lines(text: str, remove: set[str]) -> str:
    if not remove:
        return text
    return "\n".join(line for line in text.splitlines() if line.strip() not in remove)


def clean_pages(pages: list[Page], min_chars: int = 20) -> list[Page]:
    """Remove running headers and footers, normalize text and drop empty pages.

    Page numbers are preserved so that citations point at the original page.
    """
    by_source: dict[str, list[Page]] = defaultdict(list)
    for page in pages:
        by_source[page.source].append(page)

    cleaned: list[Page] = []
    for source_pages in by_source.values():
        repeated = _repeated_edge_lines(source_pages)
        for page in source_pages:
            text = normalize_text(_strip_lines(page.text, repeated))
            if len(text) >= min_chars:
                cleaned.append(Page(source=page.source, page=page.page, text=text))
    return cleaned
