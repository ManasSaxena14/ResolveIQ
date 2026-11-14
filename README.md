# ResolveIQ: AI Support Ticket Intelligence Platform

Retrieval-augmented reply generation, transformer-based ticket classification and
explainable escalation prediction, delivered through a tested and monitored API.

[![ci](https://github.com/ManasSaxena14/ResolveIQ/actions/workflows/ci.yml/badge.svg)](https://github.com/ManasSaxena14/ResolveIQ/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Overview

Support teams repeat the same three steps for every ticket: find the right answer,
route the ticket to the right team, and judge whether it is likely to escalate.
ResolveIQ performs all three in a single call:

| Output | Approach |
|---|---|
| Draft reply with page-level citations | Retrieval-augmented generation over the help center |
| Ticket category with confidence | Fine-tuned DistilBERT classifier |
| Escalation probability with top reasons | XGBoost with SHAP explanations |

See [docs/architecture.md](docs/architecture.md) for the system design.

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository, configuration, tooling and CI | Complete |
| 1 | RAG assistant: ingestion, hybrid retrieval, grounded answers, evaluation | Complete |
| 2 | Deep-learning ticket classifier | Planned |
| 3 | Escalation risk model | Planned |
| 4 | API, UI, MLOps and deployment | Planned |

## RAG assistant (Phase 1)

- Ingests PDF, Markdown, text and HTML help articles, keeping file and page metadata
- Token-aware chunking with overlap, measured with the embedding model's tokenizer
- Local vector store (ChromaDB) combined with BM25 keyword search through reciprocal
  rank fusion, with optional cross-encoder reranking
- Refusal rule on retrieval similarity, so out-of-scope questions get
  "I could not find this in the documents." instead of an invented answer
- Grounded prompt with page-level citations and a prompt-injection guard
- One config value switches between a local Ollama model and hosted free tiers
  (Gemini, Groq, OpenRouter)
- Frozen 40-question test set with dev and holdout splits, automatic source and refusal
  scoring, hand-labeled correctness, and every run logged to MLflow

Details: [docs/rag_pipeline.md](docs/rag_pipeline.md).

## Getting started

Requirements: Python 3.12 on macOS or Linux. Docker is not required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-dev
cp .env.example .env
```

Build the index and evaluate retrieval:

```bash
make index
python eval/run_rag_eval.py run --split dev --retrieval-only
```

For full answers, install [Ollama](https://ollama.com) and pull the default model with
`ollama pull llama3.2`, or set `llm.provider` to a hosted provider and add its key to
`.env`. Then run:

```bash
python eval/run_rag_eval.py run --split dev --run-name baseline
```

Inspect experiment runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.

## Development

```bash
make lint        # ruff lint and format check
make test        # test suite
make coverage    # test suite with the 70% coverage gate used in CI
```

Tests replace the embedding model, vector database and language model with lightweight
doubles, so the suite runs in seconds and needs no downloads.

## Project structure

```
configs/config.yaml       all settings, validated at load time
data/kb/                  synthetic help center for the fictional retailer Altura
docs/                     architecture and component documentation
eval/                     RAG test set and evaluation script
scripts/                  offline jobs such as building the index
src/resolveiq/            application package
  config.py               configuration loading and validation
  tracking.py             MLflow helpers
  rag/                    ingestion, retrieval, generation and evaluation
tests/                    unit and component tests
```

## Data

The knowledge base is synthetic and written for this project, so every evaluation
question has a known answer. It does not describe a real company. See
[data/README.md](data/README.md).

## License

Released under the [MIT License](LICENSE).
