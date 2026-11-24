---
language: en
license: cc-by-nc-4.0
tags:
  - customer-support
  - text-classification
  - sentence-embeddings
---

# ResolveIQ Semantic Model

Decides the routing queue and the escalation risk of a support ticket from its
`BAAI/bge-small-en-v1.5` embedding: similar past tickets decide familiar tickets, and linear
models trained on hand-written shop tickets decide the rest. Part of
[ResolveIQ](https://github.com/ManasSaxena14/ResolveIQ).

## Intended use

Suggest a queue and an order of work to a support team, with reasons. A person confirms the
queue when the model is unsure and decides about every ticket. Not for automatic actions,
other languages, or decisions about individual customers.

## Data

- Past tickets: 11,435 training tickets of
  [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
  revision `ddf1c81`, CC BY-NC 4.0, stored as embeddings with their queue and whether their
  priority was high. The license carries over.
- 288 shop tickets written for this project, with a queue and an urgency label.

## Results

| Test | Queue | Risk |
|---|---|---|
| 40 holdout shop tickets | 34 right (95% interval 71% to 93%), 14 of 14 when sent on without review | ROC-AUC 0.922, 5 of 8 urgent and 3 of 32 calm flagged |
| 2,451 test tickets of the dataset | accuracy 0.748, macro F1 0.738 | PR-AUC 0.905 |
| The 376 of those least like a training ticket | accuracy 0.319 | PR-AUC 0.497 |

The dataset contains many rewordings of the same ticket, and most of its test score comes
from recognizing them.

## Limitations

Small hand-written training and test sets by one author; urgency in calm business language
is missed; no knowledge of the customer or of time; English only. Details:
[semantic_model.md](https://github.com/ManasSaxena14/ResolveIQ/blob/main/docs/semantic_model.md).
