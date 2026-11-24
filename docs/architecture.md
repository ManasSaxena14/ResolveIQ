# Architecture

ResolveIQ answers one support ticket per call with three outputs: a cited draft reply, a
ticket category and an escalation risk score. Each output comes from its own model lane,
and an MLOps layer tracks, tests and monitors all of them.

```
                    +--------------------------------------+
  support ticket -->|  analyze(ticket)  (FastAPI + React) |
                    +--------------------------------------+
                        |                |                |
             +----------+---+   +--------+------+   +-----+-----------+
             | RAG reply    |   | Category      |   | Escalation risk |
             | (Phase 1)    |   | (Phase 2)     |   | (Phase 3)       |
             +--------------+   +---------------+   +-----------------+
             Chroma + BM25      Semantic model      Semantic model
             language model     review flag         + XGBoost, SHAP

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

## Request flow

1. `POST /analyze` (or the Analyze button) validates the ticket: 1 to 4,000 characters,
   known ticket type, no unknown fields.
2. `Copilot.analyze` assigns a request ID and runs the three lanes in order, timing each:
   retrieval and generation, classification, risk scoring.
3. If generation fails, the retrieved sources are kept and the draft is marked
   unavailable; the API answers 503 with the partial result.
4. One row goes to the SQLite request log: category, confidence, risk score, text length
   and hash, latency and model versions. The ticket text is never stored.
5. The response carries the request ID, which `/feedback` and the UI buttons use to
   store a vote.

## Model lifecycle

```
scripts (train, evaluate) --> MLflow runs --> register_models.py --> registry alias "production"
                                                                         |
                       service startup: resolve alias, cache version, load  (fallback: models/)
                                                                         |
            request log --> drift_report.py --> retrain rule --> retrain, evaluate, register
```

Details: [monitoring.md](monitoring.md).

## Repository layout

```
app.py              Gradio page, a lighter alternative to the React web app
configs/            configuration
data/kb/            help center documents (synthetic)
docs/               architecture, component docs, model cards, evaluation report
eval/               RAG test set and evaluation script
frontend/           React and TypeScript web app, served by the API from frontend/dist
scripts/            offline jobs: index, training, evaluation, registry, drift, benchmarks
src/resolveiq/      application package
  service.py        Phase 4: the three lanes behind one call
  schemas.py        request and response models shared by API and UI
  registry.py       MLflow model registry and Hub downloads
  rag/              Phase 1: retrieval-augmented generation
  classifier/       Phase 2: ticket queue classification
  risk/             Phase 3: escalation risk features, training and explanations
  api/              Phase 4: FastAPI application
  ui/               Phase 4: Gradio page
  monitoring/       Phase 4: request log, feedback and drift detection
results/            committed evaluation artifacts, drift reports, benchmarks
tests/              unit, component, contract and behaviour tests with test doubles
```

## Phase status

| Phase | Component | Status |
|---|---|---|
| 0 | Repository, configuration, tooling, CI | Complete |
| 1 | RAG assistant ([details](rag_pipeline.md)) | Complete |
| 2 | Ticket classifier ([details](classifier.md)) | Complete |
| 3 | Escalation risk model ([details](risk_model.md)) | Complete |
| 4 | Service, API, UI, monitoring, deployment ([API](api.md), [monitoring](monitoring.md), [deployment](deployment.md)) | Complete |
| 5 | Evaluation report, model cards, packaging ([report](evaluation_report.md)) | Complete |
| 6 | React web app ([details](frontend.md)) | Complete |
