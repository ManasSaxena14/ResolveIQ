import pytest

from resolveiq.rag.retrieve import Retriever, reciprocal_rank_fusion
from resolveiq.rag.store import BM25Index
from resolveiq.rag.types import Chunk

TEXTS = [
    "Refunds are issued within three business days after the return arrives.",
    "Express shipping takes two business days and costs fifteen dollars.",
    "Gift cards never expire and have no fees.",
    "Error code E42 means the smart hub firmware must be updated.",
    "The password reset link is valid for thirty minutes.",
]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [Chunk(f"kb.md::p1::c{i}", "kb.md", 1, i, text) for i, text in enumerate(TEXTS)]


@pytest.fixture
def loaded_store(vector_store, embedder, chunks):
    vector_store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    return vector_store


def test_rrf_rewards_items_ranked_by_both_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)

    assert max(scores, key=scores.get) == "b"
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_vector_retrieval_ranks_by_similarity(embedder, loaded_store):
    retriever = Retriever(embedder, loaded_store, top_k=2, candidate_k=5)

    result = retriever.retrieve("how long is the password reset link valid")

    assert result.chunks[0].chunk.chunk_index == 4
    assert len(result.chunks) == 2
    assert result.best_similarity == pytest.approx(result.chunks[0].similarity)
    assert "vector_ms" in result.timings_ms


def test_hybrid_retrieval_fuses_keyword_hits(embedder, loaded_store, chunks):
    retriever = Retriever(embedder, loaded_store, top_k=3, candidate_k=5, bm25=BM25Index(chunks))

    result = retriever.retrieve("what does error E42 mean")

    assert result.chunks[0].chunk.chunk_index == 3
    assert "keyword_ms" in result.timings_ms
    # Fused scores are rank based, so they are below the cosine similarity.
    assert result.chunks[0].score < result.chunks[0].similarity


def test_keyword_only_hits_have_no_similarity(embedder, chunks):
    class EmptyVectorStore:
        def query(self, embedding, k):
            return []

    retriever = Retriever(embedder, EmptyVectorStore(), 2, 5, bm25=BM25Index(chunks))

    result = retriever.retrieve("gift cards expire")

    assert result.chunks[0].chunk.chunk_index == 2
    assert result.chunks[0].similarity is None
    assert result.best_similarity == 0.0


def test_reranker_reorders_candidates(embedder, loaded_store):
    class LengthReranker:
        def score(self, query, texts):
            return [-len(text) for text in texts]

    retriever = Retriever(embedder, loaded_store, 2, 5, reranker=LengthReranker())

    result = retriever.retrieve("business days")

    shortest = min(TEXTS, key=len)
    assert result.chunks[0].chunk.text == shortest
    assert result.chunks[0].score == -len(shortest)
    assert "rerank_ms" in result.timings_ms


def test_candidate_pool_must_cover_top_k(embedder, vector_store):
    with pytest.raises(ValueError):
        Retriever(embedder, vector_store, top_k=5, candidate_k=2)


def test_from_settings_requires_an_index(settings):
    with pytest.raises(FileNotFoundError, match="build_index"):
        Retriever.from_settings(settings)


def test_from_settings_detects_embedding_model_mismatch(settings):
    from resolveiq.rag.ingest import IndexSummary, write_manifest

    summary = IndexSummary(1, 1, 1, "another-model", 40, 5, 0.1, "2026-01-01T00:00:00+00:00")
    write_manifest(settings.resolve(settings.paths.index_dir), summary)

    with pytest.raises(ValueError, match="Rebuild the index"):
        Retriever.from_settings(settings)
