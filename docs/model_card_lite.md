---
language: en
license: cc-by-nc-4.0
library_name: onnx
tags:
  - customer-support
  - onnx
  - int8
---

# ResolveIQ Lite Bundle

Everything the [ResolveIQ](https://github.com/ManasSaxena14/ResolveIQ) service needs to
run without PyTorch on a small free host (under 512 MB of memory): the fine-tuned ticket
classifier and the bge-small embedder as 8-bit ONNX graphs, the help center index and the
escalation risk model. Built by `scripts/export_lite.py` and loaded by
`src/resolveiq/lite.py` when `RESOLVEIQ_RUNTIME=lite`.

| File | Content |
|---|---|
| `classifier.onnx` | DistilBERT queue classifier, dynamic int8 quantization with per-channel scales (68 MB, from 268 MB) |
| `embedder.onnx` | `BAAI/bge-small-en-v1.5`, same quantization (34 MB), CLS pooling and L2 normalization at run time |
| `kb_index.npz`, `bm25_corpus.json` | Embeddings and keyword index of the 31 help center chunks |
| `risk/` | The escalation risk bundle (XGBoost, calibrator, TF-IDF models), unchanged |
| `lite_meta.json` | Versions, sizes and the agreement checks below |

## Agreement with the full-precision models

Measured by the export script; the lite runtime is used only if these hold up.

| Check | Result |
|---|---|
| Classifier: same label as the original, 400 validation tickets | 90.0% |
| Classifier accuracy on those tickets, lite and original | 49.8% and 48.8% |
| Mean difference in confidence | 0.043 |
| Embedder: cosine similarity to the original, 20 test questions | 0.997 mean, 0.997 lowest |
| Retrieval, RAG dev split: expected source found | 100%, as in the full runtime |
| Retrieval, dev: out-of-scope refused, answerable refused | 57.1% and 0% (full: 42.9% and 0%) |

The disagreements are tickets the original model was already unsure about; the lite
model is as accurate on the true labels.

## Running cost

On a laptop the lite API starts in about 2.6 seconds, uses 347 MB of memory at idle and
363 MB under load (25 analyses and a 500-row batch), and answers a warm analysis in under
100 ms without the language model step.

## Limitations

Category highlights come from the TF-IDF queue model inside the risk bundle, because ONNX
graphs carry no gradients; the full runtime uses gradient times input on DistilBERT. The
limitations of the classifier and the risk model apply unchanged; see their model cards.
