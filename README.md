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

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository, configuration, tooling and CI | Complete |
| 1 | RAG assistant | In progress |
| 2 | Deep-learning ticket classifier | Planned |
| 3 | Escalation risk model | Planned |
| 4 | API, UI, MLOps and deployment | Planned |

## Getting started

Requirements: Python 3.12 on macOS or Linux. Docker is not required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install-dev
cp .env.example .env
```

Run the checks used in CI:

```bash
make lint
make coverage
```

## Configuration

All settings live in [`configs/config.yaml`](configs/config.yaml) and are validated at
load time. Secrets are read from environment variables or a local `.env` file, which
is excluded from version control.

## License

Released under the [MIT License](LICENSE).
