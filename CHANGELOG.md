# Changelog

All notable changes to this project are documented in this file.

## [1.3.0] - 2026-11-20

### Added

- Streaming: `POST /analyze/stream` sends the category and risk first, then the draft as
  the language model writes it; Ollama, Groq, OpenRouter and Gemini clients stream
- Word highlights for both decisions (DistilBERT gradient times input, risk text model
  contributions) and a Why card in the web app
- Draft grounding: citation coverage and invalid citations on every draft, shown as
  numbered citation chips in the web app
- `run_rag_eval.py run --judge`: a language model judges correctness and support,
  replacing hand labels, with citation and support rates in the summary
- Rate limits per caller (30 a minute anonymous, 300 with an API key), API keys from
  `RESOLVEIQ_API_KEYS`, 429 with `Retry-After` and 401 for unknown keys
- Public demo of the full web app on Hugging Face Spaces without Docker: `space.py`
  serves the React app and the API, and `scripts/deploy_space.py` builds, stages and
  uploads it
- Free-only guarantee: `llm.free_only` rejects paid OpenRouter models, the Space is
  always created on free `cpu-basic` hardware with a warning for paid hardware, and a
  costs section lists every service and its free limits

### Removed

- The Render deployment option, whose free tier cannot hold the models

## [1.2.0] - 2026-11-20

### Changed

- Redesigned the web app as a triage instrument (see `frontend/DESIGN.md`)
  - Paper and Night themes, Instrument Serif, Geist and Geist Mono bundled locally
  - A verdict strip that answers route, risk and reply before the evidence
  - A ruled risk meter with the alert threshold, numbered sources with match bars and a
    response-time bar that accounts for every millisecond
  - Navigation rail on desktop, top bar on tablets, bottom tab bar on phones
  - Staggered entrance, meter fills and a risk count-up, all off under reduced motion
  - A new logo: an R whose leg resolves into the lime verdict dot on a ticket-stub tile,
    with a matching favicon, home screen icons, a link preview image and README logos

### Fixed

- File sync duplicates such as `App 2.tsx` are excluded from type checking, tests and lint

## [1.1.0] - 2026-11-17

### Added

- React and TypeScript web app served by the API
  - Analyze page with the draft, category and risk cards, linked sources, timings and
    feedback
  - Batch page with CSV upload, a ranked table, an alerts-only filter and CSV export
  - Monitoring dashboard over the request log
  - Live health status and light and dark themes
  - Vitest and Testing Library tests, and a CI job for lint, type check, tests and build
- `GET /requests` and `GET /stats` endpoints for the dashboard, without ticket text

## [1.0.0] - 2026-11-17

### Added

- Packaging (Phase 5)
  - Evaluation report with the RAG, classifier, risk, cost and system tables, results by
    ticket type, five failure cases and limits
  - Model card for the escalation risk model, and a script that publishes either model
    to the Hugging Face Hub
  - README with a screenshot, results summary, run instructions without Docker and
    limitations

- Service, UI and MLOps (Phase 4)
  - `Copilot.analyze`: draft reply, category and risk in one call, with a partial result
    when the language model is unavailable
  - FastAPI service with `/analyze`, `/batch`, `/feedback` and `/health`, request IDs,
    explicit 413, 422, 404 and 503 errors, and contract tests
  - Gradio web UI with source links, a confidence bar, a risk gauge with reasons,
    batch upload and feedback buttons
  - SQLite request log and feedback store that keep text length and hash, never text
  - Drift detection with KS and chi-square tests plus PSI, an HTML report, sample
    reports and a retrain rule
  - MLflow model registry with a `production` alias loaded by the service
  - Retrieval-only RAG experiments on dev and a final holdout run
  - API benchmark, in-process replay and per-ticket-type evaluation scripts
  - Hosting without Docker: provider and model repository settings from the environment

### Fixed

- Knowledge base indexing skips hidden files and sync-tool duplicates such as
  `refunds 2.md`

- Escalation risk model (Phase 3)
  - Written proxy label (high priority) and a leakage checklist for every dataset column
  - Feature builder: text statistics, VADER sentiment, keyword groups, ticket type,
    out-of-fold queue probabilities and an out-of-fold text score
  - Logistic regression baseline, XGBoost with early stopping and an 18-trial grid
    search, and sigmoid calibration, all tracked in MLflow
  - Metrics for imbalanced labels: PR-AUC, recall at fixed precision, Brier score and
    ROC-AUC, with precision-recall and calibration plots
  - Alert threshold chosen on validation from a cost ratio, with a cost table
  - Top three reasons per ticket from TreeSHAP values, and a SHAP summary plot
  - Batch scoring of a CSV of tickets into a list ranked by risk
  - Single-thread XGBoost when PyTorch is loaded, avoiding an OpenMP conflict on macOS

- Ticket classifier (Phase 2)
  - Download from a pinned dataset revision, cleaning, PII masking, deduplication and a
    stratified split, with a data profile
  - TF-IDF and logistic regression baseline
  - DistilBERT fine-tuning loop with class-weighted loss, warmup, length bucketing and
    early stopping on validation macro F1, running on CUDA, MPS or CPU
  - Review threshold chosen on validation, stored with the model
  - Final test-set evaluation with comparison table, per-class report, confusion matrix,
    calibration curve and error analysis
  - Model card and a script to publish the model to the Hugging Face Hub
  - Results for three training seeds (42, 7, 1234) with mean and spread

- RAG assistant (Phase 1)
  - Knowledge base of 31 synthetic help center articles for the fictional retailer Altura
  - Loaders for PDF, Markdown, text and HTML with page-level metadata
  - Text cleaning that removes running headers and footers and repairs line breaks
  - Token-aware chunking with configurable size and overlap
  - Persistent ChromaDB vector store and BM25 keyword index with an index manifest
  - Hybrid retrieval with reciprocal rank fusion and optional cross-encoder reranking
  - Grounded answers with citations, a similarity-based refusal rule and a
    prompt-injection guard
  - Language model clients for Ollama, Gemini, Groq and OpenRouter behind a config switch
  - 40-question evaluation set with dev and holdout splits and an evaluation script that
    logs every run to MLflow

## [0.1.0]

### Added

- Project structure, packaging, pinned dependencies and tooling (ruff, pytest, pre-commit)
- Validated YAML configuration
- GitHub Actions workflow running lint and tests with a coverage gate
