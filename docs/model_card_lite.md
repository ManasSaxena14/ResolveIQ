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
run without PyTorch on a small free host (under 512 MB of memory): the bge-small
embedder as an 8-bit ONNX graph, the semantic model, the help center index and the
escalation risk model. Built by `scripts/export_lite.py` and loaded by
`src/resolveiq/lite.py` when `RESOLVEIQ_RUNTIME=lite`.

| File | Content |
|---|---|
| `semantic/` | The semantic model: past ticket embeddings and the retail models, unchanged |
| `embedder.onnx` | `BAAI/bge-small-en-v1.5`, dynamic int8 quantization with per-channel scales (34 MB), CLS pooling and L2 normalization at run time |
| `kb_index.npz`, `bm25_corpus.json` | Embeddings and keyword index of the 31 help center chunks |
| `risk/` | The escalation risk bundle (XGBoost, calibrator, TF-IDF models), unchanged |
| `lite_meta.json` | Versions, sizes and the agreement checks below |

## Agreement with the full-precision models

Measured by the export script; the lite runtime is used only if these hold up.

| Check | Result |
|---|---|
| Queue: same label as with the original embedder, 400 validation tickets | 96.3% |
| Queue accuracy on those tickets, lite and original | 74.3% and 75.5% |
| Mean difference in confidence | 0.014 |
| Embedder: cosine similarity to the original, 20 test questions | 0.997 mean, 0.997 lowest |
| Retrieval, RAG dev split: expected source found | 100%, as in the full runtime |
| Retrieval, dev: out-of-scope refused, answerable refused | 57.1% and 0% (full: 42.9% and 0%) |

On the dev half of the hand-written shop tickets the lite runtime chooses 30 of 40 queues
rightly, the full runtime 31, with nearly the same risk ROC-AUC (0.905 and 0.902). Retrieval was not changed.

## Running cost

The bundle is 50 MB. After 40 analyses and a batch the API process has a memory footprint of
257 MB on a laptop.

## Limitations

Highlights test 12 words of a ticket where the full runtime tests 20. The limitations of the
semantic model and the risk model apply unchanged; see their model cards. A large batch
embeds every row, which can take minutes on a host with a fraction of a CPU.
