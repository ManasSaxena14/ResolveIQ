"""Shared fixtures and lightweight test doubles.

The doubles stand in for the embedding model, the vector database and the
language model, so the suite runs in seconds and never downloads a model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml

from resolveiq.config import DEFAULT_CONFIG_PATH, Settings, load_config
from resolveiq.rag.store import bm25_tokenize
from resolveiq.rag.types import Chunk


class HashingEmbedder:
    """Deterministic bag-of-words embedder.

    Each token is hashed into one of ``dim`` buckets, so texts that share words
    get a high cosine similarity and unrelated texts get a low one.
    """

    name = "test-hashing-embedder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in bm25_tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vector[bucket] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return (
            np.stack([self._vector(text) for text in texts]) if texts else np.zeros((0, self.dim))
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class InMemoryVectorStore:
    """Exact cosine search over vectors held in memory."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self._chunks = list(chunks)
        self._vectors = np.asarray(embeddings, dtype=np.float32)

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        scores = self._vectors @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._chunks)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


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


@pytest.fixture
def settings(tmp_path: Path, kb_dir: Path) -> Settings:
    """Settings that point every path at a temporary directory."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["rag"].update({"chunk_tokens": 40, "chunk_overlap": 5, "top_k": 3, "candidate_k": 5})
    raw["paths"] = {
        "kb_dir": str(kb_dir),
        "index_dir": str(tmp_path / "index"),
        "results_dir": str(tmp_path / "results"),
    }
    raw["tracking"]["enabled"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)
