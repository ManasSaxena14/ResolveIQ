# Architecture

ResolveIQ answers one support ticket per call with three outputs: a cited draft reply, a
ticket category and an escalation risk score. Each output comes from its own model lane,
and an MLOps layer tracks, tests and monitors all of them.

```
                    +--------------------------------------+
  support ticket -->|  analyze(ticket)  (FastAPI / Gradio) |
                    +--------------------------------------+
                        |                |                |
             +----------+---+   +--------+------+   +-----+-----------+
             | RAG reply    |   | Category      |   | Escalation risk |
             | (Phase 1)    |   | (Phase 2)     |   | (Phase 3)       |
             +--------------+   +---------------+   +-----------------+
             Chroma + BM25      Fine-tuned          XGBoost + SHAP
             language model     DistilBERT          cost-based threshold

             MLOps layer: MLflow tracking, SQLite request log, drift reports, CI (Phase 4)
```

## Principles

- **One package, thin interfaces.** Core logic lives in `src/resolveiq`. The API and the
  web UI are thin layers over it, so tests cover both.
- **Offline versus online.** Indexing and training run in scripts. Serving only loads
  saved artifacts.
- **Partial failure.** If the language model is unavailable, the category and risk
  outputs are still returned.
- **Config-driven.** Model names, thresholds and the language model provider are set in
  `configs/config.yaml`.

## Repository layout

```
configs/            configuration
data/kb/            help center documents (synthetic)
docs/               architecture and component documentation
eval/               RAG test set and evaluation script
scripts/            offline jobs (index build, training)
src/resolveiq/      application package
  rag/              Phase 1: retrieval-augmented generation
tests/              unit and component tests with lightweight test doubles
```

## Phase status

| Phase | Component | Status |
|---|---|---|
| 0 | Repository, configuration, tooling, CI | Complete |
| 1 | RAG assistant ([details](rag_pipeline.md)) | Complete |
| 2 | Ticket classifier | Planned |
| 3 | Escalation risk model | Planned |
| 4 | Service, API, UI, monitoring, deployment | Planned |
