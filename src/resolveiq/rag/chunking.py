"""Split pages into overlapping, token-limited chunks."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from resolveiq.rag.types import Chunk, Page

TokenCounter = Callable[[str], int]

_WORD_WITH_SPACE = re.compile(r"\S+\s*")


def whitespace_token_count(text: str) -> int:
    """Approximate token count: one token per whitespace-separated word."""
    return len(text.split())


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Stable identifier, so rebuilding the index replaces chunks in place."""
    return f"{source}::p{page}::c{chunk_index}"


def split_text(
    text: str,
    chunk_tokens: int,
    chunk_overlap: int,
    count_tokens: TokenCounter = whitespace_token_count,
) -> list[str]:
    """Split ``text`` into windows of at most ``chunk_tokens`` tokens.

    Text is split on word boundaries, so a word is never cut in half. Each
    window after the first starts with up to ``chunk_overlap`` tokens taken from
    the end of the previous window. A single word longer than the limit becomes
    its own chunk.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if not 0 <= chunk_overlap < chunk_tokens:
        raise ValueError("chunk_overlap must be between 0 and chunk_tokens - 1")

    words = _WORD_WITH_SPACE.findall(text)
    if not words:
        return []
    costs = [max(1, count_tokens(word.strip())) for word in words]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end, total = start, 0
        while end < len(words) and (end == start or total + costs[end] <= chunk_tokens):
            total += costs[end]
            end += 1
        chunks.append("".join(words[start:end]).strip())
        if end >= len(words):
            break

        # Step back from the end of this window to build the overlap, always
        # moving forward by at least one word so the loop terminates.
        next_start, overlap = end, 0
        while next_start > start + 1 and overlap + costs[next_start - 1] <= chunk_overlap:
            next_start -= 1
            overlap += costs[next_start]
        start = next_start
    return chunks


def chunk_pages(
    pages: Iterable[Page],
    chunk_tokens: int,
    chunk_overlap: int,
    count_tokens: TokenCounter = whitespace_token_count,
) -> list[Chunk]:
    """Chunk every page, keeping the source file and page number on each chunk."""
    chunks: list[Chunk] = []
    for page in pages:
        texts = split_text(page.text, chunk_tokens, chunk_overlap, count_tokens)
        for index, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(page.source, page.page, index),
                    source=page.source,
                    page=page.page,
                    chunk_index=index,
                    text=text,
                )
            )
    return chunks
