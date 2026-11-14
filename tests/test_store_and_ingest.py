import json

import numpy as np
import pytest

from resolveiq.rag.ingest import MANIFEST_FILENAME, build_index, read_manifest
from resolveiq.rag.store import BM25_FILENAME, BM25Index, bm25_tokenize
from resolveiq.rag.types import Chunk


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"doc::p1::c{i}", source="doc.md", page=1, chunk_index=i, text=text)


def test_bm25_tokenizer_lowercases_and_drops_stopwords():
    assert bm25_tokenize("How do I reset MY Password?") == ["reset", "password"]


def test_bm25_ranks_matching_chunk_first():
    index = BM25Index(
        [
            _chunk(0, "Express shipping takes two business days."),
            _chunk(1, "Gift cards never expire and have no fees."),
            _chunk(2, "Standard shipping is free on orders over fifty dollars."),
        ]
    )

    hits = index.query("do gift cards expire", k=2)

    assert hits[0][0].chunk_index == 1
    assert all(score > 0 for _, score in hits)


def test_bm25_returns_nothing_without_matches():
    index = BM25Index([_chunk(0, "Express shipping takes two business days.")])

    assert index.query("warranty", k=3) == []
    assert index.query("the and of", k=3) == []
    assert BM25Index([]).query("anything", k=3) == []


def test_bm25_round_trip(tmp_path):
    chunks = [
        _chunk(0, "Refunds take five days."),
        _chunk(1, "Shipping is free."),
        _chunk(2, "Gift cards never expire."),
    ]
    path = tmp_path / "bm25.json"

    BM25Index(chunks).save(path)
    loaded = BM25Index.load(path)

    assert loaded.chunks == chunks
    assert loaded.query("refunds", k=1)[0][0] == chunks[0]


def test_bm25_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        BM25Index.load(tmp_path / "missing.json")


def test_build_index_persists_everything(settings, embedder, vector_store):
    store = vector_store

    summary = build_index(settings, embedder=embedder, store=store)

    index_dir = settings.resolve(settings.paths.index_dir)
    assert summary.files == 3
    assert summary.pages == 3
    assert summary.chunks == store.count() >= 3
    assert summary.embedding_model == embedder.name
    assert (index_dir / BM25_FILENAME).is_file()
    manifest = json.loads((index_dir / MANIFEST_FILENAME).read_text())
    assert manifest["chunks"] == summary.chunks
    assert read_manifest(index_dir) == manifest


def test_build_index_replaces_previous_contents(settings, embedder, vector_store):
    store = vector_store
    store.upsert([_chunk(0, "stale")], np.ones((1, embedder.dim)))

    build_index(settings, embedder=embedder, store=store)

    assert all(hit.source != "doc.md" for hit, _ in store.query(embedder.embed_query("stale"), 10))


def test_build_index_fails_on_empty_knowledge_base(settings, embedder, vector_store):
    for path in settings.resolve(settings.paths.kb_dir).rglob("*.*"):
        path.unlink()

    with pytest.raises(ValueError, match="No text"):
        build_index(settings, embedder=embedder, store=vector_store)


def test_read_manifest_returns_none_when_missing(tmp_path):
    assert read_manifest(tmp_path) is None
