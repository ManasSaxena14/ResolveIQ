# Ticket Classifier

> The service no longer uses this classifier by default. The
> [semantic model](semantic_model.md) replaced it; this document remains the record of the
> fine-tuning experiment.

This document describes Phase 2 of ResolveIQ: a fine-tuned transformer that reads a
support ticket and predicts which queue (department) should handle it, with a
confidence score and a flag for human review when the model is unsure.

## Task

| Item | Value |
|---|---|
| Input | Ticket subject and body (English) |
| Output | One of 10 queues, a confidence between 0 and 1, and a `needs_review` flag |
| Data | 16,337 synthetic support tickets, see [data/README.md](../data/README.md) |
| Split | 11,435 train, 2,451 validation, 2,451 test (stratified, seed 42) |
| Headline metric | Macro F1, so that small queues count as much as large ones |

## Pipeline

```
data/raw (pinned revision)
  -> prepare_tickets.py   clean, mask PII, deduplicate, stratified split
  -> train_baseline.py    TF-IDF + logistic regression (number to beat)
  -> train_classifier.py  DistilBERT fine-tuning per seed, review threshold on validation
  -> evaluate_classifier.py  one final run on the test split, plots and comparison table
```

Every run is logged to MLflow in the `classifier` experiment.

## Models

**Baseline.** TF-IDF over unigrams and bigrams (`min_df=2`, sublinear term frequency)
with logistic regression and balanced class weights.

**DistilBERT.** `distilbert-base-uncased` fine-tuned with a plain PyTorch loop:

| Setting | Value | Reason |
|---|---|---|
| Max sequence length | 256 | The longest ticket is 210 tokens, so nothing is truncated |
| Learning rate | 5e-5 | Standard range for DistilBERT fine-tuning |
| Batch size | 16 | Fits an 8 GB Apple Silicon machine |
| Epochs | 5, early stopping with patience 1 | The task converges slowly; validation macro F1 decides the checkpoint |
| Warmup | 10% of steps, then linear decay | Stable start from pre-trained weights |
| Weight decay | 0.01, not on biases and layer norms | Standard AdamW practice |
| Class imbalance | Class-weighted cross-entropy | Queues differ in size by a factor of 20 |
| Seeds | 42, 7, 1234 | Mean and spread show how much the result depends on initialization |

Batches group tickets of similar length and are padded per batch, which removes most
padding and keeps training practical on a laptop GPU.

## Review threshold

A classifier that is right most of the time can still route a large share of tickets
wrongly. After training, the model's validation predictions are sorted by confidence
and the threshold is set to the lowest confidence at which the predictions above it are
still at least 90% accurate. Tickets below the threshold are returned with
`needs_review: true`, so a person confirms the queue. The threshold is stored with the
model in `classifier_meta.json` and checked again on the test split.

## Measured results

All numbers come from `results/classifier/metrics.json`, produced by
`scripts/evaluate_classifier.py` in a single run on the frozen test split (2,451
tickets). Each DistilBERT checkpoint was chosen on the validation split, and the
production checkpoint is the seed with the highest validation macro F1 (seed 42).

| Model | Accuracy | Macro F1 | Weighted F1 | Weakest class (F1) | Training time |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 0.6091 | **0.6333** | 0.6100 | IT Support (0.5142) | 30 s |
| DistilBERT, seed 42 (5 epochs) | 0.4725 | 0.5213 | 0.4722 | Customer Service (0.3774) | 72.9 min |
| DistilBERT, seed 7 (5 epochs) | 0.4782 | 0.5200 | 0.4797 | Customer Service (0.3861) | 56.4 min |
| DistilBERT, seed 1234 (5 epochs) | 0.4733 | 0.5164 | 0.4724 | Customer Service (0.3629) | 82.6 min |
| DistilBERT, mean of 3 seeds | 0.4747 | 0.5192 +/- 0.0025 | 0.4748 | | 70.6 min per seed |

The spread across seeds is small (0.25 macro F1 points), so the gap to the baseline is
not an artifact of one unlucky initialization.

Validation macro F1 of DistilBERT per epoch:

| Seed | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|---|---|---|---|---|---|
| 42 | 0.271 | 0.371 | 0.452 | 0.496 | 0.540 |
| 7 | 0.274 | 0.343 | 0.430 | 0.503 | 0.531 |
| 1234 | 0.305 | 0.374 | 0.416 | 0.472 | 0.512 |

Per-queue F1 on the test split, for the production checkpoint (seed 42):

| Queue | Test tickets | Baseline | DistilBERT |
|---|---|---|---|
| Billing and Payments | 239 | 0.817 | 0.811 |
| Service Outages and Maintenance | 100 | 0.724 | 0.714 |
| Human Resources | 52 | 0.755 | 0.597 |
| General Inquiry | 35 | 0.646 | 0.480 |
| Technical Support | 711 | 0.626 | 0.422 |
| Sales and Pre-Sales | 77 | 0.580 | 0.488 |
| Returns and Exchanges | 123 | 0.562 | 0.506 |
| Product Support | 461 | 0.561 | 0.410 |
| Customer Service | 362 | 0.547 | 0.377 |
| IT Support | 291 | 0.514 | 0.409 |

Plots: [confusion matrix](../results/classifier/confusion_matrix.png) and
[calibration curve](../results/classifier/calibration.png).

### The fine-tuned model does not beat the baseline

The success target was DistilBERT exceeding the baseline by at least 5 macro F1 points.
It did not: averaged over three seeds it is 11.4 points below. The evidence points to two causes.

1. **Undertraining.** Validation macro F1 was still rising by 4 to 5 points per epoch
   when training stopped at epoch 5 for all three seeds, and training loss was still falling (0.70). Training
   ran on an 8 GB Apple M3 under heavy memory pressure, so more epochs were not
   practical locally. A GPU run with more epochs is the first follow-up.
2. **A task that favors keyword features.** The dataset is synthetic, and the queues
   that both models handle well (Billing and Payments, Service Outages) have distinctive
   vocabulary. TF-IDF with bigrams picks up such cues from 11,435 examples immediately,
   while the transformer has to adapt its representations first.

Both models also share a ceiling caused by the labels themselves (see the error
analysis): neither reaches 0.65 macro F1.

### Review threshold

The threshold was chosen on validation as the lowest confidence at which accuracy
reaches 90%: **0.878**. On the test split, it routes 16.3% of tickets automatically
with 86.8% accuracy and sends the remaining 83.7% (39.5% accuracy) to human review.
The expected calibration error is 0.098 on test: the model is overconfident between
0.6 and 0.9 confidence, as the calibration curve shows.

### Decision for the service (Phase 4)

The service will load whichever classifier has the higher validation macro F1, so with
the current results it routes tickets with the TF-IDF baseline. The DistilBERT path stays
in place and replaces the baseline automatically once a better checkpoint is trained.

## Error analysis

The production model misclassifies 1,293 of 2,451 test tickets. The most frequent
confusions are all between support queues:

| True queue | Predicted queue | Count |
|---|---|---|
| Technical Support | IT Support | 175 |
| Technical Support | Product Support | 126 |
| Product Support | Technical Support | 98 |
| Customer Service | Product Support | 92 |
| Technical Support | Customer Service | 91 |

A random sample of 24 errors (seed 7, from `results/classifier/errors.csv`) was read
by hand and grouped by cause:

| Cause | Count | Example | What would help |
|---|---|---|---|
| Overlapping queue definitions | 11 | A dashboard that stops updating is labeled Technical Support and predicted Product Support; an integration glitch is labeled Technical Support and predicted IT Support | Merge or redefine the support queues, or route to a shared support pool and let agents pick the team |
| Label contradicts the content | 8 | Login failures under Returns and Exchanges; a system failure under heavy load under Billing and Payments; "intermittent service interruptions" labeled IT Support, predicted Service Outages with 0.99 confidence | Relabel a sample to measure label noise; treat confident disagreements as candidates for relabeling |
| Requests with no natural queue | 4 | Digital marketing strategy requests spread across Product Support, Customer Service, General Inquiry and IT Support | Add a dedicated queue or route such requests to General Inquiry by rule |
| Clear model mistake | 1 | A performance-analytics issue predicted as Returns and Exchanges | More training epochs |

Most errors in the sample (19 of 24) come from the labels rather than the model. This
is the main limit on what any classifier can reach on this dataset, and the reason
the review flag matters more than a small gain in macro F1.

## Usage

```python
from pathlib import Path
from resolveiq.classifier.predict import TicketClassifier

classifier = TicketClassifier.load(Path("models/classifier/production"))
prediction = classifier.predict_ticket(
    subject="Charged twice",
    body="I was billed twice for my subscription this month.",
)
print(prediction.label, prediction.confidence, prediction.needs_review)
```

## Reproducing

```bash
make data                  # download and prepare the dataset
make baseline              # TF-IDF baseline
make classifier            # DistilBERT, all seeds
make evaluate-classifier   # final test-set evaluation
```

On an Apple M3 with 8 GB of memory, one seed takes a little over an hour on the MPS
backend. A free Colab or Kaggle GPU is considerably faster; the same script runs there
unchanged.
