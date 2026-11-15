---
language: en
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert-base-uncased
tags:
  - customer-support
  - ticket-routing
  - text-classification
---

# ResolveIQ Ticket Classifier

DistilBERT fine-tuned to route English customer support tickets to one of 10 queues.
Part of [ResolveIQ](https://github.com/ManasSaxena14/ResolveIQ), an AI support ticket
intelligence platform.

## Intended use

Suggest a queue for an incoming support ticket and flag low-confidence predictions for
a person to confirm. It is a decision aid for support teams, not an automatic router:
every prediction below the review threshold must be checked by a person.

Out of scope: languages other than English, tickets from domains very different from
software and IT services, and any use that makes decisions about customers.

## Training data

[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets),
revision `ddf1c81`, licensed CC BY-NC 4.0, so these weights are released under the same
non-commercial license. The data is synthetic. 16,337 English tickets
after cleaning and deduplication, split 70/15/15 stratified by queue (seed 42).
Emails, URLs, phone numbers and long digit strings are masked before training.

Labels: Billing and Payments, Customer Service, General Inquiry, Human Resources,
IT Support, Product Support, Returns and Exchanges, Sales and Pre-Sales, Service Outages
and Maintenance, Technical Support.

## Training procedure

| Setting | Value |
|---|---|
| Base model | distilbert-base-uncased |
| Max sequence length | 256 |
| Optimizer | AdamW, learning rate 5e-5, weight decay 0.01 |
| Schedule | 10% linear warmup, then linear decay |
| Batch size | 16 |
| Epochs | 5 (best checkpoint by validation macro F1) |
| Loss | Class-weighted cross-entropy |
| Hardware | Apple M3, 8 GB, MPS backend, 73 minutes |

## Evaluation

Test split of 2,451 tickets, scored once.

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression baseline | 0.6091 | 0.6333 |
| This model (seed 42) | 0.4725 | 0.5213 |
| Mean of 3 seeds (42, 7, 1234) | 0.4747 | 0.5192 +/- 0.0025 |

This model is **below the baseline**. Validation macro F1 was still improving when
training stopped, so the model is undertrained. A hand review of errors also found that
most mistakes come from overlapping queue definitions and labels that contradict the
ticket text. Details: [docs/classifier.md](https://github.com/ManasSaxena14/ResolveIQ/blob/main/docs/classifier.md).

## Confidence and review

`classifier_meta.json` stores a review threshold of 0.878, chosen on validation so that
predictions above it are 90% accurate. On test, 16.3% of tickets clear the threshold with
86.8% accuracy; the rest should be reviewed. Expected calibration error on test is 0.098.

## Limitations

- Trained on synthetic tickets; performance on real support traffic is unknown.
- The source labels are noisy and several queues overlap in meaning.
- Undertrained relative to its learning curve.
- English only.

## How to use

```python
from resolveiq.classifier.predict import TicketClassifier

classifier = TicketClassifier.load(model_dir)
prediction = classifier.predict_ticket(subject="Charged twice", body="I was billed twice.")
print(prediction.label, prediction.confidence, prediction.needs_review)
```
