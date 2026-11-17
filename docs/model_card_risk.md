---
language: en
license: cc-by-nc-4.0
library_name: xgboost
tags:
  - customer-support
  - escalation-prediction
  - tabular-classification
  - explainability
---

# ResolveIQ Escalation Risk Model

Calibrated XGBoost model that estimates, when a support ticket arrives, how likely it is
to be treated as high priority, raises an alert above a cost-based threshold and explains
each score with up to three reasons. Part of
[ResolveIQ](https://github.com/ManasSaxena14/ResolveIQ).

## Intended use

Help a support team decide which new tickets to look at first. A person always makes the
decision: the score orders a queue, it is not a reason to ignore or penalize a customer.
Every score comes with its reasons so a person can disagree with it.

Out of scope: languages other than English, automatic actions without review, and any
decision about individual customers.

## Label

**Proxy label: a ticket counts as an escalation when its `priority` column is `high`.**
The training data has no escalation, reopen or satisfaction field, so the model learns
the dataset's urgency judgment, not whether tickets were later escalated.

## Training data

[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
revision `ddf1c81`, licensed CC BY-NC 4.0, so this model is released under the same
non-commercial license. Synthetic data, 16,337 English tickets after cleaning and
deduplication, split 70/15/15 stratified by queue (seed 42). 39% of tickets are high
priority. Personal data is masked before features are built.

## Inputs and features

The ticket subject, body and (optionally) its type. 30 features known at arrival: text
statistics, VADER sentiment, eight keyword groups, one-hot ticket type, the probabilities
of the 10 queues from a TF-IDF classifier, and a text score from a TF-IDF model trained on
the label. Queue and text models are applied out-of-fold on the training split. Columns
known only after handling (queue, answer, tags) are excluded.

## Files

| File | Content |
|---|---|
| `xgboost.json` | Booster cut to its best iteration (120 trees, depth 4) |
| `calibrator.joblib` | Sigmoid calibration fitted on validation scores |
| `queue_model.joblib`, `text_model.joblib` | TF-IDF models behind the stacked features |
| `risk_meta.json` | Feature schema, alert threshold, parameters and validation metrics |

Load with `resolveiq.risk.predict.RiskModel.load(folder)`. The joblib files run code when
loaded, so only load files from a source you trust.

## Evaluation

Test split of 2,451 tickets, scored once. Thresholds and calibration were chosen on
validation.

| Model | PR-AUC | Recall at 70% precision | Brier score | ROC-AUC |
|---|---|---|---|---|
| Always predict the base rate | 0.3876 | 0.0000 | 0.2374 | 0.5000 |
| Logistic regression | 0.7435 | 0.6221 | 0.1742 | 0.8175 |
| This model | 0.7489 | 0.6316 | 0.1664 | 0.8208 |

The alert threshold is 0.23, which minimizes cost on validation when a missed escalation
costs three times a false alarm. At that threshold the alert catches 86.6% of
high-priority test tickets and fires on 59.7% of all tickets.

Results by ticket type on test:

| Ticket type | Tickets | PR-AUC | Caught by the alert | Alert rate |
|---|---|---|---|---|
| Incident | 1,005 | 0.7918 | 91.7% | 64.5% |
| Problem | 486 | 0.7460 | 83.5% | 54.1% |
| Change | 274 | 0.7637 | 84.2% | 61.7% |
| Request | 686 | 0.6533 | 81.1% | 55.7% |

The model is weaker on Request tickets, where only 33% are high priority but 56% raise
an alert.

## Limitations

- The label is a proxy for urgency, not real escalation.
- The data is synthetic and has no timestamps or customer identifiers, so there is no
  time-based split and no customer history.
- Most of the signal comes from one stacked text feature, which reacts to wording rather
  than meaning; polite requests with urgent-sounding vocabulary can score high.
- English only.

Details: [docs/risk_model.md](https://github.com/ManasSaxena14/ResolveIQ/blob/main/docs/risk_model.md).
