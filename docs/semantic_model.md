# Semantic Model

The queue and the escalation risk of a ticket are decided from its meaning. This document
explains why the earlier models were replaced, how the new one works, and what it can and
cannot do. Code: [src/resolveiq/semantic.py](../src/resolveiq/semantic.py).

## What was wrong

The fine-tuned DistilBERT classifier and the XGBoost risk model were trained on a public
dataset about software and IT services, while the help center and the web app are about a
shop. Two findings led to this model.

**They did not work on shop tickets.** On 40 hand-written shop tickets they chose a right
queue for 16, and the risk score ordered urgent and calm tickets no better than chance
(ROC-AUC 0.48). It flagged 26 of 28 calm tickets.

**Their dataset scores were flattered by near-copies.** Half of the validation tickets have
a training ticket with an embedding similarity of at least 0.966, because the dataset
contains many rewordings of the same ticket. Ordered by TF-IDF similarity to the
training split, the validation accuracy of a TF-IDF model rises from 37% for the least
similar quarter to 94% for the most similar one. A model that
recognizes rewordings scores well here without understanding anything new.

## How it works

```
ticket -> PII masking -> bge-small embedding -> highest similarity to 11,435 past tickets
   similarity >= 0.85   familiar: the 20 most similar past tickets vote on the queue;
                        risk stacks XGBoost with their share of high priority
   similarity <= 0.75   unfamiliar: linear models on the embedding, trained on 288
                        hand-written shop tickets, decide the queue and the urgency
   in between           the two answers are blended
```

The gate follows from the data: 98% of validation tickets are above 0.82, and every ticket
in the dev half of the shop test set is below 0.81.

| Part | Detail |
|---|---|
| Embedder | `BAAI/bge-small-en-v1.5`, the one the help center search already uses |
| Past tickets | The training split only, as 16-bit embeddings (8 MB) |
| Shop examples | [data/seed/retail_tickets.csv](../data/seed/retail_tickets.csv), 288 tickets, 85 urgent |
| Queue, unfamiliar | Logistic regression (C = 20), temperature scaled on out-of-fold scores |
| Urgency, unfamiliar | Logistic regression (C = 5, balanced), Platt scaled on out-of-fold scores |
| Review thresholds | Confidence from which 90% are right: 0.519 familiar (validation), 0.881 unfamiliar (out of fold) |
| Alert threshold | 0.15, the lowest total cost at 3 : 1 with both kinds of ticket counting equally |

Explanations:

- **Highlighted words** are found by removal: each content word (up to 20, or 12 in the
  lite runtime) is taken out, the ticket is embedded again, and the change in the score is
  the word's weight. Familiar tickets keep the risk model's exact term contributions.
- **Reasons** of a familiar ticket are the share of similar past tickets with high priority
  and the risk model's SHAP reasons. An unfamiliar ticket quotes the closest urgent example.

## Results

From `scripts/evaluate_semantic.py`, run once after every choice was made
([all tables](../results/semantic/comparison.md)). "Before" is DistilBERT for the queue and
XGBoost for the risk.

### Shop tickets

[eval/ticket_testset.csv](../eval/ticket_testset.csv) holds 80 hand-written tickets with
every acceptable queue and an urgency label. It was written before the model was built.
The dev half was used while building it; the holdout half was read once.

| Half | Model | Right queue | Right when sent on without review | Risk ROC-AUC | Urgent flagged | Calm flagged |
|---|---|---|---|---|---|---|
| dev | before | 16 of 40 | 10 of 13 | 0.482 | 8 of 12 | 26 of 28 |
| dev | semantic | 31 of 40 | 16 of 16 | 0.902 | 9 of 12 | 3 of 28 |
| **holdout** | before | 17 of 40 | 8 of 10 | 0.527 | 8 of 8 | 28 of 32 |
| **holdout** | **semantic** | **34 of 40** | **14 of 14** | **0.922** | **5 of 8** | **3 of 32** |

The 95% interval of 34 out of 40 is 71% to 93%. The earlier model flagged all 8 urgent
holdout tickets because it flagged 36 of all 40.

### Public dataset, test split (2,451 tickets)

| Model | Queue accuracy | Macro F1 | Sent on without review | Accuracy of those | Risk PR-AUC | Brier |
|---|---|---|---|---|---|---|
| TF-IDF and logistic regression | 0.6091 | 0.6333 | | | | |
| before | 0.4725 | 0.5213 | 16.3% | 86.8% | 0.7489 | 0.1664 |
| **semantic** | **0.7483** | **0.7376** | **72.1%** | **89.6%** | **0.9052** | **0.1036** |

By similarity to the closest training ticket:

| Similarity | Tickets | Queue accuracy before | semantic | Risk PR-AUC before | semantic |
|---|---|---|---|---|---|
| under 0.90 | 376 | 0.301 | 0.319 | 0.538 | 0.497 |
| 0.90 to 0.95 | 565 | 0.366 | 0.496 | 0.599 | 0.722 |
| 0.95 to 0.98 | 651 | 0.507 | 0.891 | 0.837 | 0.971 |
| 0.98 and above | 859 | 0.591 | 0.994 | 0.896 | 0.999 |

**Read the overall numbers with this table.** The gain on the dataset comes from tickets
that nearly repeat a past one. That is useful in a support team, where questions do repeat,
but on genuinely new dataset tickets the model is right about a third of the time, as
before, and its risk ranking is slightly worse. The shop tickets are the fairer test.

## Cost

| | Before | Now |
|---|---|---|
| Lite bundle | 105 MB | 50 MB |
| Lite runtime, memory footprint after 40 analyses and a batch | 363 MB (resident) | 257 MB |
| Full runtime, median response without the language model | 44 ms | 289 ms |

Most of the added time is spent removing words for the highlights. The response times were
measured on an 8 GB laptop with other programs running, so they are upper estimates. A
batch now embeds every row, where before it did not.

## Limitations

- The shop tickets, for training and for testing, were written by one author. Real
  customers write differently, and 40 holdout tickets is a small sample.
- Urgency in calm business language is missed: an expiring return window and a quote
  needed before a board meeting were not flagged on the dev half.
- The model is unsure about most shop tickets: 26 of the 40 holdout tickets go to a person
  to confirm the queue.
- A quoted urgent example shows what the ticket is close to. It is evidence, not proof.
- English only.

## Reproducing

```bash
make semantic            # embeds the tickets, fits and calibrates (about 5 minutes)
make evaluate-semantic   # test split and holdout
make register            # registers it as resolveiq-semantic
```

Set `service.decisions: transformer` in `configs/config.yaml` to use DistilBERT and XGBoost
alone. The service does the same, with a warning, when the semantic model is not trained.
