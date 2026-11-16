# Escalation Risk Model

This document describes Phase 3 of ResolveIQ: a gradient-boosted tree model that
estimates, at the moment a ticket arrives, how likely it is to escalate, raises an
alert above a cost-based threshold, and explains each score with its top three reasons.

## Task

| Item | Value |
|---|---|
| Input | Ticket subject and body, and the ticket type chosen on the request form |
| Output | Calibrated escalation probability, a yes or no alert, and up to three reasons |
| Data | The 16,337 English tickets from Phase 2, see [data/README.md](../data/README.md) |
| Split | The frozen Phase 2 split: 11,435 train, 2,451 validation, 2,451 test |
| Headline metric | PR-AUC, with recall at 70% precision, Brier score and ROC-AUC |

## Label definition

**An escalation is a ticket whose `priority` column is `high`.**

The dataset has no escalation, reopen, satisfaction or resolution-time field, so this
label is a **proxy**. It measures how urgent the ticket was judged to be, not whether it
was later escalated. Results here describe how well the model recovers that judgment
from the ticket text; they make no claim about real escalations.

| Split | Tickets | High priority | Rate |
|---|---|---|---|
| Train | 11,435 | 4,460 | 39.0% |
| Validation | 2,451 | 935 | 38.1% |
| Test | 2,451 | 950 | 38.8% |

A 39% positive rate is far higher than real escalation rates. This matters for the
alert threshold below.

## Prediction moment and leakage checklist

The model scores a ticket when it arrives. Every input must be known at that moment.

| Column | Used | Reason |
|---|---|---|
| `subject`, `body` | Yes | Written by the customer when the ticket is created |
| `type` (Incident, Problem, Request, Change) | Yes | Assumed to be chosen on the request form; if agents set it during triage it should be removed |
| `priority` | No | Defines the label |
| `queue` | No | Assigned by an agent; the model uses the queue predicted from the text instead |
| `answer` | No | Written after the ticket is handled |
| `tag_1` to `tag_8` | No | Added during handling and partly restate priority and queue |
| `version`, `language` | No | Dataset bookkeeping; language is fixed to English |

Further checks:

- `LEAKAGE_COLUMNS` in `src/resolveiq/risk/features.py` lists the excluded columns, and
  a test asserts that none of them becomes a feature.
- Model-derived features (queue probabilities, text score) are **out-of-fold** on the
  training split: every ticket is scored by a model that never saw it. Validation and
  test tickets are scored by models trained on the training split only.
- The early-stopping round, the tuning grid winner, the calibration and every alert
  threshold are chosen on the validation split. The test split is scored once.
- **No time-based split.** The dataset has no timestamps, so the frozen stratified random
  split from Phase 2 is reused. A real deployment should train on older tickets and test
  on newer ones.
- **No customer history.** The dataset has no customer identifier, so features such as
  "tickets from this customer in the last 30 days" cannot be built.

## Features

| Group | Features |
|---|---|
| Text statistics | Character and word count, exclamation and question marks, share of capital letters |
| Sentiment | VADER compound score and share of negative wording |
| Keyword groups | Distinct phrases found for urgency, outage, security, data loss, money, cancelling, legal terms and repeated contact |
| Ticket type | One-hot Incident, Problem, Request, Change |
| Phase 2 output | Probability of each of the 10 queues from the TF-IDF queue classifier |
| Text score | Escalation probability from a TF-IDF and logistic regression model trained on the label |

30 features in total. The Phase 2 baseline is used for queue probabilities rather than
DistilBERT because it scores higher on validation and is fast enough to run five times
for out-of-fold predictions.

### Why the text score is included

Without it, the hand-built features reach a validation PR-AUC of only about 0.61. Adding
the text score raises it to about 0.75 (validation, same split):

| Model (validation PR-AUC) | Hand-built features only | With text score |
|---|---|---|
| Logistic regression | 0.5695 | 0.7432 |
| XGBoost, starting parameters | 0.6026 | 0.7526 |
| XGBoost, tuned | 0.6094 | 0.7564 |

The text model is a stacked feature, so it is built out-of-fold like the queue
probabilities. Set `risk.text_score_feature: false` in `configs/config.yaml` to
reproduce the left column.

## Models

| Model | Setup |
|---|---|
| Majority reference | Predicts the training escalation rate for every ticket |
| Logistic regression | Standard-scaled features, balanced class weights |
| XGBoost, starting parameters | Depth 5, learning rate 0.05, subsample 0.8, column sample 0.8, up to 1,000 trees with early stopping on validation PR-AUC (50 rounds), `scale_pos_weight` from the class ratio |
| XGBoost, tuned | Grid of 18 combinations over depth (4, 5, 6), minimum child weight (1, 5) and column sample (0.6, 0.8, 1.0); best by validation PR-AUC: depth 4, minimum child weight 1, column sample 1.0, 120 trees |
| XGBoost, tuned and calibrated | Platt scaling (sigmoid) fitted on validation scores |

Every model and each of the 18 tuning trials is logged to MLflow in the
`escalation-risk` experiment, with the saved model and plots as artifacts.

Sigmoid calibration was chosen over isotonic after comparing both: isotonic regression
produced flat steps that pushed some probabilities to exactly 0 or 1 and lowered test
PR-AUC to 0.731, while sigmoid keeps the ranking unchanged and gives the better Brier
score.

## Measured results

All numbers come from `results/risk/metrics.json`, produced by
`scripts/evaluate_risk.py` in a single run on the test split (2,451 tickets, 38.8%
positive).

| Model | PR-AUC | Recall at 70% precision | Brier score | ROC-AUC |
|---|---|---|---|---|
| Always predict the training escalation rate | 0.3876 | 0.0000 | 0.2374 | 0.5000 |
| Logistic regression | 0.7435 | 0.6221 | 0.1742 | 0.8175 |
| XGBoost, starting parameters | **0.7497** | **0.6453** | 0.1708 | 0.8206 |
| XGBoost, tuned | 0.7489 | 0.6316 | 0.1718 | **0.8208** |
| XGBoost, tuned and calibrated | 0.7489 | 0.6316 | **0.1664** | **0.8208** |

Findings:

- XGBoost nearly doubles the PR-AUC of the reference (0.75 against 0.39), so the text
  carries a real signal about priority.
- XGBoost is only about 0.6 PR-AUC points above logistic regression. Most of the signal
  comes from the text score, which both models share.
- Tuning helped on validation (0.7564 against 0.7526) but not on test. The grid gain is
  within noise for a split of this size, so the tuned model is kept for its simpler
  trees (depth 4, 120 trees) rather than for higher accuracy.
- Calibration improves the Brier score from 0.1718 to 0.1664 without changing the
  ranking. Plots: [precision-recall curves](../results/risk/precision_recall.png),
  [calibration curve](../results/risk/calibration.png).

## Alert threshold from a cost ratio

An alert costs an agent a minute of attention; a missed escalation costs far more. The
threshold is chosen on the validation split by computing the total cost
`misses x cost_of_miss + false_alarms x 1` for every threshold from 0.05 to 0.95 in
steps of 0.01 and taking the lowest. The cost ratios are assumptions, so the table
shows several:

| Cost of a miss : cost of a false alarm | Chosen threshold | Escalations caught | False alarms per 100 alerts | Tickets alerted |
|---|---|---|---|---|
| 1 : 1 | 0.53 | 59.0% | 27.8 | 31.7% |
| 2 : 1 | 0.30 | 79.3% | 38.7 | 50.1% |
| **3 : 1 (production)** | **0.23** | **86.6%** | **43.7** | **59.7%** |
| 5 : 1 | 0.19 | 91.7% | 46.9 | 66.9% |
| 10 : 1 | 0.11 | 97.5% | 55.2 | 84.2% |
| 20 : 1 | 0.06 | 99.6% | 59.6 | 95.5% |

Thresholds come from validation; the other columns are measured on test.

**Decision: 3 : 1.** With a 39% positive rate, ratios of 10 : 1 and above make the
cheapest policy "alert on almost every ticket", which gives a team lead nothing to act
on. At 3 : 1 the model catches 87% of high-priority tickets while leaving 40% of tickets
without an alert. On a real escalation label, which is rare, a ratio of 10 : 1 or more
would be appropriate and would give a far more selective threshold. Change
`risk.alert_cost_ratio` and retrain to apply another assumption.

## Explanations

Each score is explained with SHAP values from XGBoost's built-in TreeSHAP
(`pred_contribs=True`, the same algorithm as `shap.TreeExplainer`). For a ticket, the
features that raise its log-odds the most, by at least 0.05, become plain sentences, up
to three. For the text score, the sentence names the words that pushed the text model
up most, computed as TF-IDF weight times coefficient.

Examples from the test split ([all examples](../results/risk/example_explanations.md)):

| Probability | Reasons |
|---|---|
| 0.95 | Wording typical of high-priority tickets: "urgent", "firewall and", "systems resulting" |
| 0.95 | Wording typical of high-priority tickets: "system", "unexpected outage", "outage"; Predicted queue is Service Outages and Maintenance (77%) |
| 0.02 | No feature raises the score |

The [SHAP summary plot](../results/risk/shap_summary.png) shows the text score dominating
(mean absolute SHAP 1.08 log-odds), followed by the queue probabilities: Human
Resources, Returns and Exchanges and Customer Service tickets lower the risk, Technical
Support and Service Outages raise it. Keyword counts and sentiment contribute little once
the text score is present. Global values are in
[feature_importance.csv](../results/risk/feature_importance.csv).

## Batch scoring

```bash
python scripts/score_tickets.py tickets.csv --output ranked.csv
```

The input needs `subject` and `body` columns, with optional `id` and `type`. The output
lists each ticket's id, probability, alert flag and three reasons, highest risk first.
Ticket text is never copied to the output.

## Reproducing

```bash
make data            # if the Phase 2 splits do not exist yet
make risk-features   # about 1 minute: out-of-fold text models and features
make risk            # about 1 minute: all models, 18 tuning trials, calibration, threshold
make evaluate-risk   # one test-set run, plots and tables
```

## Runtime note: PyTorch and XGBoost in one process

On macOS, PyTorch and XGBoost link different OpenMP runtimes. When both start worker
threads in one process, the process crashes. `xgboost_threads()` in
`src/resolveiq/risk/train.py` therefore runs XGBoost on a single thread whenever PyTorch
is already loaded; scoring one ticket takes milliseconds either way. The test suite sets
`OMP_NUM_THREADS=1` for the same reason. The Phase 4 service, which loads both the
classifier and this model, relies on the same guard.

## Limitations

- The label is a proxy (high priority), so the model learns the dataset's urgency
  judgment, not real escalation.
- The data is synthetic and has no timestamps or customer identifiers, so there is no
  time-based split and no customer history features.
- Most of the signal comes from one stacked text feature; the hand-built features add
  little on this dataset.
- The text score reacts to wording rather than meaning. A polite change request such as
  "please add my colleague to the workspace when convenient" scores 0.59 because
  similar request phrasing is often labeled high priority in this synthetic data.
- Ticket type is assumed to be known at arrival; if it is set by agents it is a mild
  leak and should be dropped.
- English only.
