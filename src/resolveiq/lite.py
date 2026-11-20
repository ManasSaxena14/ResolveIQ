"""A lightweight runtime for small hosts such as Render's free tier.

The full service needs PyTorch, sentence-transformers and ChromaDB, about 760 MB
of memory and several seconds to start. The lite runtime serves the same models
without any of them:

* the fine-tuned DistilBERT classifier and the bge-small embedder run as ONNX
  graphs with 8-bit weights through ``onnxruntime``;
* the knowledge base embeddings are computed once by ``scripts/export_lite.py``
  and loaded from a NumPy file, searched by a dot product;
* the escalation risk model is the same XGBoost bundle as in the full service.

Nothing here imports torch, transformers, sentence_transformers or chromadb.
Category highlights come from the TF-IDF queue model inside the risk bundle,
because ONNX graphs have no gradients; the API documentation says so.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from resolveiq.classifier.data import build_text, clean_text
from resolveiq.classifier.predict import Prediction
from resolveiq.rag.types import Chunk

logger = logging.getLogger(__name__)

CLASSIFIER_FILE = "classifier.onnx"
CLASSIFIER_TOKENIZER = "classifier_tokenizer.json"
CLASSIFIER_META = "classifier_meta.json"
EMBEDDER_FILE = "embedder.onnx"
EMBEDDER_TOKENIZER = "embedder_tokenizer.json"
INDEX_FILE = "kb_index.npz"
LITE_META = "lite_meta.json"
RISK_DIR = "risk"


def _session(path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Small hosts have a fraction of a CPU; more threads only add overhead and memory.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_length: int) -> Any:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length)
    tokenizer.no_padding()
    return tokenizer


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _batch(encodings: Sequence[Any], with_type_ids: bool) -> dict[str, np.ndarray]:
    """Pad a list of tokenizer encodings into int64 arrays."""
    width = max(len(e.ids) for e in encodings)
    ids = np.zeros((len(encodings), width), dtype=np.int64)
    mask = np.zeros((len(encodings), width), dtype=np.int64)
    for row, encoding in enumerate(encodings):
        ids[row, : len(encoding.ids)] = encoding.ids
        mask[row, : len(encoding.ids)] = 1
    batch = {"input_ids": ids, "attention_mask": mask}
    if with_type_ids:
        batch["token_type_ids"] = np.zeros_like(ids)
    return batch


class OnnxClassifier:
    """The fine-tuned queue classifier as an 8-bit ONNX graph."""

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        labels: Sequence[str],
        review_threshold: float,
        queue_model: Any = None,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.review_threshold = review_threshold
        self.queue_model = queue_model

    @classmethod
    def load(cls, lite_dir: Path, queue_model: Any = None) -> OnnxClassifier:
        meta = json.loads((lite_dir / CLASSIFIER_META).read_text(encoding="utf-8"))
        return cls(
            session=_session(lite_dir / CLASSIFIER_FILE),
            tokenizer=_tokenizer(lite_dir / CLASSIFIER_TOKENIZER, int(meta["max_length"])),
            labels=meta["labels"],
            review_threshold=float(meta.get("review_threshold", 0.0)),
            queue_model=queue_model,
        )

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(list(texts))
        logits = self.session.run(["logits"], _batch(encodings, with_type_ids=False))[0]
        return _softmax(logits.astype(np.float64))

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        cleaned = [clean_text(text) for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Ticket text must not be empty")
        predictions = []
        for row in self.predict_proba(cleaned):
            best = int(row.argmax())
            confidence = float(row[best])
            predictions.append(
                Prediction(
                    label=self.labels[best],
                    confidence=round(confidence, 4),
                    needs_review=confidence < self.review_threshold,
                    probabilities={
                        label: round(float(p), 4) for label, p in zip(self.labels, row, strict=True)
                    },
                )
            )
        return predictions

    def predict_ticket(self, body: str, subject: str = "") -> Prediction:
        return self.predict([build_text(subject, body)])[0]

    def word_weights(self, body: str, subject: str = "", k: int = 8) -> list[tuple[str, float]]:
        """Words tied to the predicted queue, from the TF-IDF queue model in the risk bundle.

        ONNX graphs carry no gradients, so the lite runtime explains the category with
        the linear queue model instead: each word's TF-IDF weight times its coefficient
        for the predicted queue, scaled so the strongest is 1.
        """
        if self.queue_model is None:
            return []
        text = build_text(subject, body)
        label = self.predict_ticket(body, subject).label
        classes = [str(c) for c in self.queue_model.classes_]
        if label not in classes:
            return []
        vectorizer = self.queue_model.named_steps["tfidf"]
        coef = self.queue_model.named_steps["clf"].coef_
        index = classes.index(label)
        # With two classes scikit-learn keeps one row of coefficients, for the second class.
        coefficients = coef[index] if coef.shape[0] > 1 else (coef[0] if index == 1 else -coef[0])
        row = vectorizer.transform([text]).tocoo()
        vocabulary = vectorizer.get_feature_names_out()
        scores = [
            (str(vocabulary[col]), float(value * coefficients[col]))
            for col, value in zip(row.col, row.data, strict=True)
            if value * coefficients[col] > 0 and " " not in vocabulary[col]
        ]
        scores.sort(key=lambda item: -item[1])
        if not scores:
            return []
        top = scores[0][1]
        return [(word, round(score / top, 4)) for word, score in scores[:k]]


class OnnxEmbedder:
    """bge-small as an 8-bit ONNX graph: CLS pooling, L2-normalized, like the full model."""

    def __init__(self, session: Any, tokenizer: Any, name: str, query_prefix: str = "") -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.name = name
        self.query_prefix = query_prefix
        self._inputs = {i.name for i in session.get_inputs()}
        self._count_cached = lru_cache(maxsize=65536)(self._count_uncached)

    @classmethod
    def load(cls, lite_dir: Path, name: str, query_prefix: str = "") -> OnnxEmbedder:
        return cls(
            _session(lite_dir / EMBEDDER_FILE),
            _tokenizer(lite_dir / EMBEDDER_TOKENIZER, 512),
            name,
            query_prefix,
        )

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        batch = _batch(self.tokenizer.encode_batch(list(texts)), "token_type_ids" in self._inputs)
        batch = {k: v for k, v in batch.items() if k in self._inputs}
        hidden = self.session.run(None, batch)[0]
        cls_vectors = hidden[:, 0, :].astype(np.float32)
        norms = np.linalg.norm(cls_vectors, axis=1, keepdims=True)
        return cls_vectors / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return self._embed([])
        return np.concatenate([self._embed(texts[i : i + 16]) for i in range(0, len(texts), 16)])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]

    def _count_uncached(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def count_tokens(self, text: str) -> int:
        return self._count_cached(text)


class NumpyVectorStore:
    """Exact cosine search over a small, precomputed embedding matrix."""

    def __init__(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self._chunks = list(chunks)
        self._vectors = np.asarray(embeddings, dtype=np.float32)

    @classmethod
    def load(cls, path: Path) -> NumpyVectorStore:
        data = np.load(path, allow_pickle=False)
        chunks = [Chunk.from_dict(item) for item in json.loads(str(data["chunks"]))]
        return cls(chunks, data["embeddings"])

    def save(self, path: Path) -> None:
        records = json.dumps([chunk.to_dict() for chunk in self._chunks])
        np.savez_compressed(path, embeddings=self._vectors, chunks=np.array(records))

    def reset(self) -> None:
        self._chunks, self._vectors = [], np.zeros((0, 0), dtype=np.float32)

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        self._chunks = list(chunks)
        self._vectors = np.asarray(embeddings, dtype=np.float32)

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if not self._chunks:
            return []
        scores = self._vectors @ np.asarray(embedding, dtype=np.float32)
        order = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in order]

    def count(self) -> int:
        return len(self._chunks)


def lite_ready(lite_dir: Path) -> bool:
    required = [CLASSIFIER_FILE, CLASSIFIER_META, EMBEDDER_FILE, INDEX_FILE, LITE_META]
    return (
        all((lite_dir / name).is_file() for name in required)
        and (lite_dir / RISK_DIR / "risk_meta.json").is_file()
    )
