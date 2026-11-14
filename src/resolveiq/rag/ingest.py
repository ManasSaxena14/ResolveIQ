"""Build the retrieval index from the knowledge base."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from resolveiq.config import Settings
from resolveiq.rag.chunking import chunk_pages
from resolveiq.rag.cleaning import clean_pages
from resolveiq.rag.embeddings import Embedder, SentenceTransformerEmbedder
from resolveiq.rag.loaders import load_documents
from resolveiq.rag.store import BM25_FILENAME, BM25Index, ChromaVectorStore, VectorStore

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "index_manifest.json"


@dataclass(frozen=True)
class IndexSummary:
    files: int
    pages: int
    chunks: int
    embedding_model: str
    chunk_tokens: int
    chunk_overlap: int
    build_seconds: float
    built_at: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_index(
    settings: Settings,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> IndexSummary:
    """Load, clean, chunk and embed the knowledge base, then persist the index.

    The vector collection is rebuilt from scratch on every run so that deleted
    or renamed documents never leave stale chunks behind.
    """
    started = time.perf_counter()
    rag = settings.rag
    kb_dir = settings.resolve(settings.paths.kb_dir)
    index_dir = settings.resolve(settings.paths.index_dir)

    raw_pages = load_documents(kb_dir)
    pages = clean_pages(raw_pages)
    logger.info("Loaded %d pages from %s", len(pages), kb_dir)

    embedder = embedder or SentenceTransformerEmbedder(rag.embedding_model, rag.query_prefix)
    chunks = chunk_pages(pages, rag.chunk_tokens, rag.chunk_overlap, embedder.count_tokens)
    if not chunks:
        raise ValueError(f"No text could be extracted from the files in {kb_dir}")

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store = store or ChromaVectorStore(index_dir, rag.collection)
    store.reset()
    store.upsert(chunks, vectors)
    BM25Index(chunks).save(index_dir / BM25_FILENAME)

    summary = IndexSummary(
        files=len({page.source for page in raw_pages}),
        pages=len(pages),
        chunks=len(chunks),
        embedding_model=embedder.name,
        chunk_tokens=rag.chunk_tokens,
        chunk_overlap=rag.chunk_overlap,
        build_seconds=round(time.perf_counter() - started, 2),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    write_manifest(index_dir, summary)
    return summary


def write_manifest(index_dir: Path, summary: IndexSummary) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / MANIFEST_FILENAME).write_text(
        json.dumps(summary.as_dict(), indent=2), encoding="utf-8"
    )


def read_manifest(index_dir: Path) -> dict[str, object] | None:
    path = index_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
