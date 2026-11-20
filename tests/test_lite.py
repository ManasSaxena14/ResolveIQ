"""Lite runtime pieces, tested with stand-in ONNX sessions and tokenizers."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from resolveiq.lite import (
    NumpyVectorStore,
    OnnxClassifier,
    OnnxEmbedder,
    _batch,
    _softmax,
    lite_ready,
)
from resolveiq.rag.types import Chunk


class FakeTokenizer:
    """Maps each word to a token id; no special tokens."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _ids(self, text):
        return [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.lower().split()]

    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(ids=self._ids(text))

    def encode_batch(self, texts):
        return [SimpleNamespace(ids=self._ids(t) or [0]) for t in texts]


class LogitSession:
    """Returns a high logit for label 0 when the input mentions a refund."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def run(self, names, inputs):
        refund = self.tokenizer.vocab.get("refund", -1)
        rows = [[4.0, 0.0] if refund in row else [0.0, 2.0] for row in inputs["input_ids"]]
        return [np.array(rows, dtype=np.float32)]


class HiddenSession:
    """Returns hidden states whose first token vector depends on the text length."""

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, names, inputs):
        ids = inputs["input_ids"]
        hidden = np.zeros((ids.shape[0], ids.shape[1], 384), dtype=np.float32)
        for row in range(ids.shape[0]):
            hidden[row, 0, 0] = inputs["attention_mask"][row].sum()
            hidden[row, 0, 1] = 1.0
        assert "token_type_ids" in inputs
        return [hidden]


def test_batch_pads_and_masks():
    batch = _batch([SimpleNamespace(ids=[5, 6, 7]), SimpleNamespace(ids=[8])], with_type_ids=True)

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["token_type_ids"].shape == (2, 3)
    assert np.allclose(_softmax(np.array([[1.0, 1.0]])), [[0.5, 0.5]])


def test_onnx_classifier_predicts_like_the_full_one():
    tokenizer = FakeTokenizer()
    classifier = OnnxClassifier(LogitSession(tokenizer), tokenizer, ["Billing", "Technical"], 0.9)

    billing = classifier.predict_ticket("Please refund me", "Charge")
    technical = classifier.predict_ticket("Server is down")

    assert billing.label == "Billing" and billing.needs_review is False
    assert technical.label == "Technical" and technical.needs_review is True
    assert set(billing.probabilities) == {"Billing", "Technical"}
    with pytest.raises(ValueError, match="empty"):
        classifier.predict(["   "])


def test_category_highlights_come_from_the_queue_model():
    from resolveiq.classifier.baseline import train_baseline

    texts = ["refund my invoice payment", "server outage down error"] * 6
    queue_model = train_baseline(texts, ["Billing", "Technical"] * 6)
    tokenizer = FakeTokenizer()
    classifier = OnnxClassifier(
        LogitSession(tokenizer), tokenizer, ["Billing", "Technical"], 0.5, queue_model
    )

    weights = dict(classifier.word_weights("please refund the invoice", "payment"))

    assert weights and max(weights.values()) == 1.0
    assert "refund" in weights
    assert all(" " not in word for word in weights)
    assert OnnxClassifier(LogitSession(tokenizer), tokenizer, ["A"], 0.5).word_weights("x") == []


def test_onnx_embedder_pools_cls_and_normalizes():
    embedder = OnnxEmbedder(HiddenSession(), FakeTokenizer(), "bge", query_prefix="query: ")

    query = embedder.embed_query("hello world")
    docs = embedder.embed_documents(["a", "a b c"])

    assert np.isclose(np.linalg.norm(query), 1.0)
    assert docs.shape == (2, 384)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    assert not np.allclose(docs[0], docs[1])
    assert embedder.count_tokens("one two three") == 3
    assert embedder.embed_documents([]).shape == (0, 384)


def test_numpy_store_round_trip_and_search(tmp_path):
    chunks = [Chunk(f"c{i}", f"doc{i}.md", 1, 0, f"text {i}") for i in range(3)]
    vectors = np.eye(3, dtype=np.float32)
    store = NumpyVectorStore(chunks, vectors)
    store.save(tmp_path / "index.npz")

    loaded = NumpyVectorStore.load(tmp_path / "index.npz")
    hits = loaded.query(np.array([0.1, 0.9, 0.0], dtype=np.float32), k=2)

    assert loaded.count() == 3
    assert [chunk.source for chunk, _ in hits] == ["doc1.md", "doc0.md"]
    loaded.reset()
    assert loaded.query(np.ones(3), 1) == []


def test_lite_ready_needs_every_file(tmp_path):
    assert lite_ready(tmp_path) is False
    for name in [
        "classifier.onnx",
        "classifier_meta.json",
        "embedder.onnx",
        "kb_index.npz",
        "lite_meta.json",
    ]:
        (tmp_path / name).write_text("x")
    assert lite_ready(tmp_path) is False
    (tmp_path / "risk").mkdir()
    (tmp_path / "risk" / "risk_meta.json").write_text(json.dumps({}))
    assert lite_ready(tmp_path) is True


def test_lite_runtime_never_imports_heavy_libraries():
    import subprocess
    import sys

    code = (
        "import sys, resolveiq.lite, resolveiq.service, resolveiq.api.main;"
        "heavy = [m for m in ('torch','transformers','sentence_transformers','chromadb') "
        "if m in sys.modules]; print(heavy)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.stdout.strip() == "[]", result.stderr[-500:]
