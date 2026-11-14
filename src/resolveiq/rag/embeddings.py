"""Text embedding models."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Interface used by ingestion and retrieval, so tests can use a stub."""

    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model.

    Vectors are L2-normalized, so the dot product equals cosine similarity.
    ``query_prefix`` is prepended to queries only, as recommended for
    asymmetric retrieval models such as BGE.
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self.max_seq_length: int = self._model.max_seq_length
        self._count_cached = lru_cache(maxsize=65536)(self._count_uncached)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 200,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            [self.query_prefix + text], normalize_embeddings=True, convert_to_numpy=True
        )[0]

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in ``text``, excluding special tokens."""
        return self._count_cached(text)

    def _count_uncached(self, text: str) -> int:
        return len(self._model.tokenizer.tokenize(text))
