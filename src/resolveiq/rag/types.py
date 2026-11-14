"""Data types shared across the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Page:
    """Text of one page of a source document.

    ``source`` is the path relative to the knowledge base directory and
    ``page`` is 1-based. Formats without pages (Markdown, HTML, text) produce a
    single page numbered 1.
    """

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A piece of a page that is embedded and retrieved as one unit."""

    chunk_id: str
    source: str
    page: int
    chunk_index: int
    text: str

    @property
    def metadata(self) -> dict[str, Any]:
        return {"source": self.source, "page": self.page, "chunk_index": self.chunk_index}

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text, **self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            source=str(data["source"]),
            page=int(data["page"]),
            chunk_index=int(data["chunk_index"]),
            text=str(data["text"]),
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval.

    ``score`` is the value used for the final ordering (cosine similarity,
    fused rank score or reranker score, depending on the settings).
    ``similarity`` is the cosine similarity from vector search, or ``None``
    when the chunk was found only by keyword search.
    """

    chunk: Chunk
    score: float
    similarity: float | None
