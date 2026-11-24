# Semantic model compared with the models it replaces

## Hand-written retail tickets

| Half | Model | Right queue | Sent on without review, right | Risk ROC-AUC | Urgent flagged | Calm flagged |
|---|---|---|---|---|---|---|
| dev | before | 16 of 40 | 10 of 13 | 0.4821 | 8 of 12 | 26 of 28 |
| dev | semantic | 31 of 40 | 16 of 16 | 0.9018 | 9 of 12 | 3 of 28 |
| holdout | before | 17 of 40 | 8 of 10 | 0.5273 | 8 of 8 | 28 of 32 |
| holdout | semantic | 34 of 40 | 14 of 14 | 0.9219 | 5 of 8 | 3 of 32 |

## Test split of the public dataset

| Model | Queue accuracy | Macro F1 | Sent on without review | Accuracy of those | Risk PR-AUC | Brier |
|---|---|---|---|---|---|---|
| before | 0.4725 | 0.5213 | 16.3% | 86.8% | 0.7489 | 0.1664 |
| semantic | 0.7483 | 0.7376 | 72.1% | 89.6% | 0.9052 | 0.1036 |

## Test split by similarity to the closest training ticket

| Similarity | Tickets | Queue accuracy before | Queue accuracy semantic | Risk PR-AUC before | Risk PR-AUC semantic |
|---|---|---|---|---|---|
| 0.00 to 0.90 | 376 | 0.3005 | 0.3191 | 0.5377 | 0.4973 |
| 0.90 to 0.95 | 565 | 0.3664 | 0.4956 | 0.5992 | 0.7217 |
| 0.95 to 0.98 | 651 | 0.5069 | 0.8909 | 0.8374 | 0.9712 |
| 0.98 to 1.00 | 859 | 0.5914 | 0.9942 | 0.8964 | 0.9992 |
