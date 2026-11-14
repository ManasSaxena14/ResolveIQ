"""Persistent vector store and keyword index."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from resolveiq.rag.types import Chunk

BM25_FILENAME = "bm25_corpus.json"

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "has", "have", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "the",
        "this", "to", "was", "what", "when", "where", "which", "who", "will", "with", "you",
        "your",
    }
)  # fmt: skip


class VectorStore(Protocol):
    """Interface for nearest-neighbour search over chunk embeddings."""

    def reset(self) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None: ...

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Vector store persisted to a local folder with ChromaDB.

    The collection uses cosine distance, and ``query`` returns cosine
    similarity (1 - distance) so that higher is always better.
    """

    def __init__(self, persist_dir: Path, collection: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self._name = collection
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._get_or_create()

    def _get_or_create(self):
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def reset(self) -> None:
        existing = {c if isinstance(c, str) else c.name for c in self._client.list_collections()}
        if self._name in existing:
            self._client.delete_collection(self._name)
        self._collection = self._get_or_create()

    def upsert(
        self, chunks: Sequence[Chunk], embeddings: np.ndarray, batch_size: int = 1000
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
                embeddings=np.asarray(embeddings[start : start + batch_size]).tolist(),
            )

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(embedding).tolist()],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            chunk = Chunk.from_dict({"chunk_id": chunk_id, "text": text, **meta})
            hits.append((chunk, 1.0 - float(distance)))
        return hits

    def count(self) -> int:
        return self._collection.count()


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase word tokens with common stopwords removed."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


class BM25Index:
    """Keyword index over chunk text.

    Only the chunk corpus is saved to disk, as JSON. The BM25 statistics are
    rebuilt on load, which takes milliseconds for a help-center sized corpus
    and avoids pickling.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        corpus = [bm25_tokenize(chunk.text) for chunk in self.chunks]
        self._bm25 = BM25Okapi(corpus) if self.chunks else None

    def query(self, text: str, k: int) -> list[tuple[Chunk, float]]:
        tokens = bm25_tokenize(text)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [chunk.to_dict() for chunk in self.chunks]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        if not path.is_file():
            raise FileNotFoundError(
                f"BM25 index not found at {path}. Run scripts/build_index.py first."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([Chunk.from_dict(item) for item in data])
