"""Find the chunks most relevant to a question."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from resolveiq.config import Settings
from resolveiq.rag.embeddings import Embedder, SentenceTransformerEmbedder
from resolveiq.rag.ingest import read_manifest
from resolveiq.rag.store import BM25_FILENAME, BM25Index, ChromaVectorStore, VectorStore
from resolveiq.rag.types import Chunk, RetrievedChunk


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Re-scores (question, passage) pairs with a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.predict([(query, text) for text in texts])]


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_similarity: float
    timings_ms: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Combine several ranked lists of ids into one score per id.

    Each list contributes ``1 / (k + rank)`` for every id it contains, with
    ranks starting at 1. Ids ranked highly by several lists score highest.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Vector search with optional BM25 fusion and cross-encoder reranking.

    ``best_similarity`` in the result is always the highest cosine similarity
    from vector search. It is the signal used by the refusal rule, because
    fused and reranked scores are not comparable across questions.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        candidate_k: int,
        bm25: BM25Index | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        if candidate_k < top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.bm25 = bm25
        self.reranker = reranker
        self.rrf_k = rrf_k

    @classmethod
    def from_settings(cls, settings: Settings) -> Retriever:
        rag = settings.rag
        index_dir = settings.resolve(settings.paths.index_dir)
        manifest = read_manifest(index_dir)
        if manifest is None:
            raise FileNotFoundError(
                f"No index found in {index_dir}. Run scripts/build_index.py first."
            )
        if manifest.get("embedding_model") != rag.embedding_model:
            raise ValueError(
                f"The index was built with {manifest.get('embedding_model')} but the config "
                f"uses {rag.embedding_model}. Rebuild the index."
            )
        return cls(
            embedder=SentenceTransformerEmbedder(rag.embedding_model, rag.query_prefix),
            store=ChromaVectorStore(index_dir, rag.collection),
            top_k=rag.top_k,
            candidate_k=rag.candidate_k,
            bm25=BM25Index.load(index_dir / BM25_FILENAME) if rag.hybrid else None,
            reranker=CrossEncoderReranker(rag.reranker_model) if rag.rerank else None,
            rrf_k=rag.rrf_k,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.top_k
        widen = self.bm25 is not None or self.reranker is not None
        pool_size = max(self.candidate_k, top_k) if widen else top_k
        timings: dict[str, float] = {}

        started = time.perf_counter()
        vector_hits = self.store.query(self.embedder.embed_query(query), pool_size)
        timings["vector_ms"] = _elapsed_ms(started)

        similarity = {chunk.chunk_id: score for chunk, score in vector_hits}
        best_similarity = max(similarity.values(), default=0.0)
        chunks_by_id: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk, _ in vector_hits}

        if self.bm25 is not None:
            started = time.perf_counter()
            keyword_hits = self.bm25.query(query, pool_size)
            timings["keyword_ms"] = _elapsed_ms(started)
            chunks_by_id.update({chunk.chunk_id: chunk for chunk, _ in keyword_hits})
            fused = reciprocal_rank_fusion(
                [[c.chunk_id for c, _ in vector_hits], [c.chunk_id for c, _ in keyword_hits]],
                k=self.rrf_k,
            )
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:pool_size]
        else:
            ranked = [(chunk.chunk_id, score) for chunk, score in vector_hits]

        candidates = [
            RetrievedChunk(chunks_by_id[cid], score, similarity.get(cid)) for cid, score in ranked
        ]

        if self.reranker is not None and candidates:
            started = time.perf_counter()
            scores = self.reranker.score(query, [c.chunk.text for c in candidates])
            timings["rerank_ms"] = _elapsed_ms(started)
            candidates = [
                RetrievedChunk(c.chunk, score, c.similarity)
                for c, score in sorted(
                    zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
                )
            ]

        return RetrievalResult(candidates[:top_k], best_similarity, timings)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
