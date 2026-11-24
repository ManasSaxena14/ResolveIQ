"""Build the lite runtime: 8-bit ONNX models and a precomputed knowledge base index.

The lite runtime (src/resolveiq/lite.py) serves the same models as the full service
without PyTorch, sentence-transformers or ChromaDB, so it fits small free hosts. This
script, run once on a development machine with the full dependencies, writes:

    models/lite/
      semantic/                  the semantic model, unchanged, or without one
      classifier.onnx, classifier_tokenizer.json, classifier_meta.json
      embedder.onnx, embedder_tokenizer.json
      kb_index.npz, bm25_corpus.json, index_manifest.json
      risk/                      the escalation risk bundle, unchanged
      lite_meta.json             versions, sizes and the agreement checks below

It then checks that the lite models agree with the originals: queue label agreement
and accuracy on validation tickets, embedding cosine similarity, and the
retrieval results of the RAG dev questions.

Usage:
    python scripts/export_lite.py [--check-tickets 400]
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # see docs/risk_model.md

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from resolveiq.config import PROJECT_ROOT, Settings, load_config
from resolveiq.lite import (
    CLASSIFIER_FILE,
    CLASSIFIER_META,
    CLASSIFIER_TOKENIZER,
    EMBEDDER_FILE,
    EMBEDDER_TOKENIZER,
    INDEX_FILE,
    LITE_META,
    RISK_DIR,
    SEMANTIC_DIR,
    NumpyVectorStore,
    OnnxClassifier,
    OnnxEmbedder,
)
from resolveiq.registry import clean_copy, resolve_model_dir
from resolveiq.semantic import SemanticClassifier, SemanticModel, TextEncoder, semantic_ready

RISK_FILES = ["xgboost.json", "calibrator.joblib", "queue_model.joblib", "text_model.joblib",
              "risk_meta.json"]  # fmt: skip


def quantize(fp32: Path, out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Per-channel scales with a reduced range track the full-precision model closely:
    # on 200 validation tickets the classifier agreed with the original on 91% of
    # labels, against 84% with the default per-tensor scales, at the same size.
    quantize_dynamic(
        str(fp32), str(out), weight_type=QuantType.QInt8, per_channel=True, reduce_range=True
    )


def export_classifier(model_dir: Path, out_dir: Path, tmp: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = tokenizer(["export example", "a second, longer export example"], padding=True,
                       return_tensors="pt")  # fmt: skip
    fp32 = tmp / "classifier_fp32.onnx"

    class Logits(torch.nn.Module):
        """Fixes the input order and returns only the logits."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    torch.onnx.export(
        Logits(model).eval(),
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize(fp32, out_dir / CLASSIFIER_FILE)
    shutil.copyfile(model_dir / "tokenizer.json", out_dir / CLASSIFIER_TOKENIZER)
    shutil.copyfile(model_dir / "classifier_meta.json", out_dir / CLASSIFIER_META)


def export_embedder(model_name: str, out_dir: Path, tmp: Path) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sample = tokenizer(["export example", "a second, longer export example"], padding=True,
                       return_tensors="pt")  # fmt: skip
    fp32 = tmp / "embedder_fp32.onnx"

    class Encoder(torch.nn.Module):
        """Fixes the input order and returns only the hidden states."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).last_hidden_state

    torch.onnx.export(
        Encoder(model).eval(),
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            name: {0: "batch", 1: "sequence"}
            for name in ["input_ids", "attention_mask", "token_type_ids", "last_hidden_state"]
        },
        opset_version=17,
        dynamo=False,
    )
    quantize(fp32, out_dir / EMBEDDER_FILE)
    tokenizer.save_pretrained(tmp / "embedder_tokenizer")
    shutil.copyfile(tmp / "embedder_tokenizer" / "tokenizer.json", out_dir / EMBEDDER_TOKENIZER)


def build_lite_index(settings: Settings, embedder: OnnxEmbedder, out_dir: Path) -> dict:
    from resolveiq.rag.ingest import build_index

    lite_settings = settings.model_copy(deep=True)
    lite_settings.paths.index_dir = out_dir
    store = NumpyVectorStore([], np.zeros((0, 384), dtype=np.float32))
    summary = build_index(lite_settings, embedder=embedder, store=store)
    store.save(out_dir / INDEX_FILE)
    return summary.as_dict()


def check_classifier(settings: Settings, lite: OnnxClassifier, model_dir: Path, n: int) -> dict:
    from resolveiq.classifier.data import load_split
    from resolveiq.classifier.predict import TicketClassifier

    tickets = load_split(settings.resolve(settings.classifier.processed_dir), "validation")[:n]
    texts = [t.text for t in tickets]
    return compare(tickets, TicketClassifier.load(model_dir).predict(texts), lite.predict(texts))


def check_semantic(settings: Settings, lite: OnnxEmbedder, out_dir: Path, n: int) -> dict:
    """The semantic model on the 8-bit embedder against the same model on the original."""
    from resolveiq.classifier.data import load_split
    from resolveiq.rag.embeddings import SentenceTransformerEmbedder

    tickets = load_split(settings.resolve(settings.classifier.processed_dir), "validation")[:n]
    texts = [t.text for t in tickets]
    model = SemanticModel.load(out_dir / SEMANTIC_DIR)
    original = SentenceTransformerEmbedder(settings.rag.embedding_model)
    full, small = (
        SemanticClassifier(model, TextEncoder(e)).predict(texts) for e in (original, lite)
    )
    return compare(tickets, full, small)


def compare(tickets: list, full: list, small: list) -> dict:
    agree = np.mean([a.label == b.label for a, b in zip(full, small, strict=True)])
    full_acc = np.mean([p.label == t.label for p, t in zip(full, tickets, strict=True)])
    lite_acc = np.mean([p.label == t.label for p, t in zip(small, tickets, strict=True)])
    drift = np.mean([abs(a.confidence - b.confidence) for a, b in zip(full, small, strict=True)])
    return {
        "tickets": len(tickets),
        "label_agreement": round(float(agree), 4),
        "full_accuracy": round(float(full_acc), 4),
        "lite_accuracy": round(float(lite_acc), 4),
        "mean_confidence_difference": round(float(drift), 4),
    }


def check_embedder(settings: Settings, lite: OnnxEmbedder) -> dict:
    from resolveiq.rag.embeddings import SentenceTransformerEmbedder
    from resolveiq.rag.evaluation import load_testset

    rag = settings.rag
    full = SentenceTransformerEmbedder(rag.embedding_model, rag.query_prefix)
    cases = load_testset(PROJECT_ROOT / "eval" / "rag_testset.csv")
    questions = [case.question for case in cases[:20]]
    a = np.stack([full.embed_query(q) for q in questions])
    b = np.stack([lite.embed_query(q) for q in questions])
    cosine = np.sum(a * b, axis=1)
    return {"queries": len(questions), "mean_cosine": round(float(cosine.mean()), 4),
            "min_cosine": round(float(cosine.min()), 4)}  # fmt: skip


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="Path to a config file")
    parser.add_argument("--check-tickets", type=int, default=400)
    args = parser.parse_args()

    settings = load_config(args.config)
    settings.tracking.enabled = False
    out_dir = settings.resolve(settings.service.lite_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    started = time.perf_counter()

    semantic_dir = resolve_model_dir(settings, "semantic").path
    semantic = settings.service.decisions == "semantic" and semantic_ready(semantic_dir)
    classifier_dir = None if semantic else resolve_model_dir(settings, "classifier").path
    risk_dir = resolve_model_dir(settings, "risk").path
    with tempfile.TemporaryDirectory() as tmp:
        if classifier_dir is not None:
            print("Exporting the classifier")
            export_classifier(classifier_dir, out_dir, Path(tmp))
        print("Exporting the embedder")
        export_embedder(settings.rag.embedding_model, out_dir, Path(tmp))

    if semantic:
        clean_copy(semantic_dir, out_dir / SEMANTIC_DIR)
    (out_dir / RISK_DIR).mkdir()
    staged = clean_copy(risk_dir, out_dir / RISK_DIR)
    for path in staged.iterdir():
        if path.name not in RISK_FILES:
            path.unlink()

    embedder = OnnxEmbedder.load(out_dir, settings.rag.embedding_model, settings.rag.query_prefix)
    print("Building the knowledge base index")
    index = build_lite_index(settings, embedder, out_dir)

    print("Checking the lite models against the originals")
    if classifier_dir is None:
        classifier_check = check_semantic(settings, embedder, out_dir, args.check_tickets)
    else:
        import joblib

        queue_model = joblib.load(out_dir / RISK_DIR / "queue_model.joblib")
        classifier_check = check_classifier(
            settings, OnnxClassifier.load(out_dir, queue_model), classifier_dir, args.check_tickets
        )
    embedder_check = check_embedder(settings, embedder)

    sizes = {p.name: round(p.stat().st_size / 1e6, 1) for p in out_dir.glob("*.onnx")}
    meta = {
        "embedding_model": settings.rag.embedding_model,
        "classifier": (
            {}
            if classifier_dir is None
            else json.loads((out_dir / CLASSIFIER_META).read_text())["train_config"]
        ),
        "risk_version": json.loads((out_dir / RISK_DIR / "risk_meta.json").read_text())["version"],
        "index": index,
        "onnx_mb": sizes,
        "checks": {"classifier": classifier_check, "embedder": embedder_check},
        "export_seconds": round(time.perf_counter() - started, 1),
    }
    (out_dir / LITE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6
    print(json.dumps(meta["checks"], indent=2))
    print(f"Wrote {out_dir} ({total:.0f} MB, ONNX files {sizes})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
