# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

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
