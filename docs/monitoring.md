# MLOps and Monitoring

This document covers how runs are tracked, how models reach the service, what the
service records about each request, and when to retrain.

## Experiment tracking

Every training and evaluation script logs to MLflow, backed by `mlflow.db` in the project
root. Each run records its parameters, metrics, artifacts (plots, result tables, saved
models) and the git commit as the `git_commit` tag.

| Experiment | Runs |
|---|---|
| `rag` | Index builds and every retrieval and answer evaluation |
| `classifier` | Baseline, one run per DistilBERT seed, final evaluation |
| `escalation-risk` | Logistic regression, XGBoost, the tuning grid as nested runs, final evaluation |

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Port 5001 is used because macOS reserves 5000 for AirPlay.

## Model registry

`scripts/register_models.py` logs the production classifier, risk and semantic models as MLflow
models, registers them as `resolveiq-ticket-classifier`, `resolveiq-escalation-risk` and
`resolveiq-semantic`,
and moves the `production` alias to the new version. Version tags record the git commit
and the validation score.

With `service.model_source: registry` the service resolves the alias at startup,
downloads that version once into `models/registry/<name>/<version>` and loads it. If
the registry is unavailable, as on a hosted demo, it logs a warning and loads the
folders in `service.classifier_dir`, `service.risk_dir` and `service.semantic_dir`. `/health` shows which
source was used, for example `semantic-8711d86 (resolveiq-semantic/v1)`.

Promoting a new model is two commands: train and evaluate it, then run
`python scripts/register_models.py`. Rolling back is moving the alias to the previous
version in the MLflow UI.

## Request log

`Copilot.analyze` writes one row per ticket to SQLite (`service.request_log`, default
`logs/requests.db`):

| Column | Example |
|---|---|
| `request_id`, `created_at` | `3fc77d854fb2`, `2026-11-17T08:01:12+00:00` |
| `channel` | `email` |
| `category`, `confidence`, `needs_review` | `Billing and Payments`, 0.9468, 0 |
| `risk_score`, `alert` | 0.6652, 1 |
| `text_length`, `text_hash` | 91, first 16 hex characters of SHA-256 |
| `draft_status`, `latency_ms` | `unavailable`, 1148.6 |
| `versions` | JSON of the four model versions |

The ticket text is never stored. Feedback votes from `/feedback` and the UI buttons go
to a second table linked by `request_id`.

## Drift detection

`scripts/drift_report.py` compares the newest rows of the request log (500 by default)
with a reference profile and writes an HTML report and a JSON summary to
`results/monitoring/`.

The reference profile is built once by running the production models over the
validation split (`--build-reference`). It stores the same four numbers the request log
holds, with no ticket text: text length, predicted category, classifier confidence and
risk score.

Each feature is tested twice:

| Feature type | Statistical test | Size of the shift |
|---|---|---|
| Numeric (text length, confidence, risk score) | Two-sample Kolmogorov-Smirnov | Population stability index on 10 quantile bins of the reference |
| Categorical (predicted category) | Chi-square on the category counts | Population stability index over the categories |

A feature counts as drifted only when **both** agree: p < 0.05 and PSI >= 0.2. With a
few hundred rows the tests alone flag shifts too small to matter, and PSI alone moves on
small samples. A unit test checks that a significant but tiny shift (p < 0.05, PSI
below 0.2) is not reported.

The statistics are computed with SciPy rather than Evidently: Evidently 0.7.23 fails to
import with pandas 3.1, which the rest of the project uses.

### Sample reports

Both samples replay 400 tickets from the held-out test split through the service with
`scripts/replay_tickets.py`, then run the drift report on that log.

| Sample | text_length PSI | confidence PSI | risk_score PSI | category PSI | Drifted | Retrain |
|---|---|---|---|---|---|---|
| Random test tickets ([report](../results/monitoring/drift_report_test_split.html)) | 0.022 | 0.021 | 0.021 | 0.023 | none | no |
| Only Technical Support and IT Support tickets ([report](../results/monitoring/drift_report_shifted.html)) | 0.049 | 0.025 | 0.092 | 0.936 | category | no |

The random sample comes from the same distribution as the reference, so nothing drifts.
The shifted sample simulates a week in which only technical tickets arrive: the category
mix moves far from the reference. One drifted feature of four is below the 50% of the
retrain rule, so it does not ask for retraining; the semantic model stays as confident on
these tickets as on the rest, unlike the classifier it replaced. The risk score test is
significant (p = 0.000004) but its PSI is 0.092, so it is not counted.

Note: the reference risk scores come from the validation split, on which the probability
calibrator was fitted. Their distribution is therefore slightly better calibrated than on
new data. Building the reference from a fresh labeled sample would remove this.

## Retrain rule

Retrain and re-evaluate when either condition holds:

1. **Drift:** at least 50% of the monitored features drift
   (`monitoring.retrain_drift_share`), measured on the newest 500 requests.
2. **Feedback:** after at least 20 votes (`monitoring.min_feedback`), more than 30% are
   thumbs down (`monitoring.retrain_thumbs_down_rate`).

The drift rule fires on a change in what customers write about, which affects both
models. The feedback rule catches quality problems that do not show up as drift, for
example a policy change that makes the knowledge base wrong. In both cases retraining
means: collect and label recent tickets, rerun `make data`, `make classifier` or
`make risk`, compare with the current production model on the frozen test split, and
register the new version only if it is better.

## Logging

- Each HTTP request is logged with method, path, status, duration and request ID.
- `/analyze` writes a structured JSON line with the request ID, text length and hash,
  channel, category, risk score, draft status and step timings.
- Language model failures are logged with the error type only.
