# Evaluation Report

Every number in this report comes from a logged run in this repository: MLflow
experiments `rag`, `classifier` and `escalation-risk`, and the files under `results/`.
Cells that could not be measured say so and why. Nothing is copied from elsewhere.

## 1. Goal and data

ResolveIQ takes one support ticket and returns a cited draft reply, the routing queue and
an escalation risk with reasons, for a human agent to act on.

| Data | Source | Size | Split |
|---|---|---|---|
| Knowledge base | 31 synthetic help articles for the fictional retailer Altura, written for this project | 31 files, 31 chunks | none |
| RAG test set | 40 hand-written questions in `eval/rag_testset.csv` | 30 answerable, 10 out of scope | dev 30, holdout 10, frozen |
| Tickets | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets), revision `ddf1c81`, CC BY-NC 4.0, synthetic | 16,337 English tickets after cleaning and deduplication | 70/15/15 stratified by queue, seed 42, frozen |

The ticket data has no timestamps, customer identifiers or escalation field. Section 6
lists what this rules out.

## 2. Method

**RAG (Phase 1).** Help articles are cleaned, split into token-aware chunks and embedded
with `BAAI/bge-small-en-v1.5` into ChromaDB. Retrieval fuses vector and BM25 rankings
with reciprocal rank fusion, with optional cross-encoder reranking. If the best cosine
similarity is below a threshold the question is refused without calling the language
model. Otherwise the model answers from the numbered passages only, cites them, and is
told to treat passages and the question as data. Details: [rag_pipeline.md](rag_pipeline.md).

**Classifier (Phase 2).** A TF-IDF and logistic regression baseline, and DistilBERT
fine-tuned with class-weighted loss and early stopping on validation macro F1, three
seeds. A review threshold chosen on validation flags predictions that are less than 90%
likely to be right. Details: [classifier.md](classifier.md).

**Escalation risk (Phase 3).** Proxy label: priority is high. 30 features known at
arrival, including out-of-fold queue probabilities and an out-of-fold text score.
Logistic regression and XGBoost with an 18-trial grid, sigmoid calibration on
validation, an alert threshold from a cost ratio and TreeSHAP reasons. Details:
[risk_model.md](risk_model.md).

**Service and MLOps (Phase 4).** One `Copilot.analyze` call behind FastAPI and Gradio, a
SQLite request log without ticket text, drift reports, an MLflow model registry and CI.
Details: [api.md](api.md), [monitoring.md](monitoring.md), [deployment.md](deployment.md).

## 3. Results

### Table A: RAG experiments (dev split, 23 answerable and 7 out-of-scope questions)

All runs are retrieval-only: they score whether the expected article is retrieved and
whether the refusal rule fires, without calling a language model. Answer correctness and
generation time need a language model and hand labeling of every answer; no language
model was available in the measurement environment, so those columns are not filled.
Runs are in MLflow under the names shown.

| Run | Chunk | Overlap | Top-k | Hybrid | Rerank | Refusal threshold | Source right % | Out-of-scope refused % | Answerable refused % | Median retrieval ms |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 `r1-vector-top5` | 500 | 50 | 5 | no | no | 0.62 | 100.0 | 42.9 | 0.0 | 10.3 |
| R2 `r2-hybrid-top5` | 500 | 50 | 5 | yes | no | 0.62 | 100.0 | 42.9 | 0.0 | 10.7 |
| R3 `r3-hybrid-rerank-top5` | 500 | 50 | 5 | yes | yes | 0.62 | 100.0 | 42.9 | 0.0 | 148.3 |
| R4 `r4-hybrid-top3` | 500 | 50 | 3 | yes | no | 0.62 | 100.0 | 42.9 | 0.0 | 10.5 |
| R5 `r5-hybrid-top8` | 500 | 50 | 8 | yes | no | 0.62 | 100.0 | 42.9 | 0.0 | 10.6 |
| R6 `r6-chunk300-hybrid-top5-clean` | 300 | 50 | 5 | yes | no | 0.62 | 100.0 | 42.9 | 0.0 | 10.5 |
| T1 `threshold-0.60` | 500 | 50 | 5 | yes | no | 0.60 | 100.0 | 28.6 | 0.0 | |
| T2 `threshold-0.64` | 500 | 50 | 5 | yes | no | 0.64 | 95.7 | 71.4 | 4.3 | |
| T3 `threshold-0.66` | 500 | 50 | 5 | yes | no | 0.66 | 91.3 | 71.4 | 8.7 | |
| T4 `threshold-0.68` | 500 | 50 | 5 | yes | no | 0.68 | 87.0 | 100.0 | 13.0 | |
| T5 `threshold-0.70` | 500 | 50 | 5 | yes | no | 0.70 | 78.3 | 100.0 | 21.7 | |
| **Final, holdout** `final-holdout-hybrid-top5` | 500 | 50 | 5 | yes | no | 0.62 | 100.0 (7 of 7) | 0.0 (0 of 3) | 0.0 | 11.1 |

Findings:

- Retrieval is saturated. With 31 short articles, every setting retrieves the expected
  article for all 23 answerable dev questions. Every article is shorter than 300 tokens,
  so 300 and 500 token chunks produce the same 31 chunks. Reranking adds about 140 ms
  without changing any result, so it stays off.
- The refusal threshold is the setting that matters. 0.62 is the highest value that
  refuses no answerable question; each step above it trades answerable questions for
  out-of-scope ones. 0.62 is kept, and the remaining out-of-scope questions rely on the
  prompt's instruction to refuse when the passages do not contain the answer.
- On the holdout split the similarity rule refused none of the three out-of-scope
  questions. Questions that mention Altura ("What is the Altura stock price today?",
  similarity 0.74) look in scope to the embedding model. See failure case 1.

The language model columns can be filled once Ollama or an API key is set up:
`python eval/run_rag_eval.py run --split dev --run-name <name> --judge` writes the answers,
lets the language model judge correctness and support against the help center passages,
and reports citation coverage and validity. A model judging its own answers can be
lenient, so the report should name the judge and spot-check a sample by hand.

### Table B: Ticket classifier (frozen test split, 2,451 tickets)

From `results/classifier/comparison.md`.

| Model | Accuracy | Macro F1 | Weakest class (F1) | Training time |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.6091 | 0.6333 | IT Support (0.5142) | 30 s |
| DistilBERT, seed 42 (production) | 0.4725 | 0.5213 | Customer Service (0.3774) | 72.9 min |
| DistilBERT, mean of 3 seeds | 0.4747 | 0.5192 +/- 0.0025 | | 70.6 min per seed |

The fine-tuned model is below the baseline; [classifier.md](classifier.md) explains why
(undertrained, overlapping queues, noisy labels). With the review threshold of 0.878,
16.3% of test tickets are routed automatically at 86.8% accuracy.

### Table C: Escalation risk model (test split, 2,451 tickets, 38.8% high priority)

From `results/risk/metrics.json`. The split is stratified, not time-based, because the
dataset has no timestamps.

| Model | PR-AUC | Recall at 70% precision | Brier score | ROC-AUC |
|---|---|---|---|---|
| Always predict the base rate | 0.3876 | 0.0000 | 0.2374 | 0.5000 |
| Logistic regression | 0.7435 | 0.6221 | 0.1742 | 0.8175 |
| XGBoost | 0.7497 | 0.6453 | 0.1708 | 0.8206 |
| XGBoost, tuned and calibrated | 0.7489 | 0.6316 | 0.1664 | 0.8208 |

### Table D: Alert threshold under different cost assumptions

Thresholds chosen on validation, other columns measured on test.

| Cost of a miss : cost of a false alarm | Chosen threshold | Misses caught % | False alarms per 100 alerts |
|---|---|---|---|
| 3 : 1 (production) | 0.23 | 86.6 | 43.7 |
| 5 : 1 | 0.19 | 91.7 | 46.9 |
| 10 : 1 | 0.11 | 97.5 | 55.2 |
| 20 : 1 | 0.06 | 99.6 | 59.6 |

With 39% positives, ratios of 10 : 1 and above alert on most tickets, so 3 : 1 is used
([risk_model.md](risk_model.md) has the full table and the reasoning).

### Table E: System behavior

| Measure | Value | How measured |
|---|---|---|
| Median and 95th percentile response time, local Mac | 43.9 ms and 49.8 ms | 50 `/analyze` requests over HTTP, `scripts/benchmark_api.py`, results in `results/system/api_benchmark_local.json` |
| Same, in process, 400 tickets | 40.1 ms and 51.7 ms | `scripts/replay_tickets.py`, `results/system/replay_latency.json` |
| Median and 95th percentile response time, public demo | not measured | the demo is not deployed yet; see [deployment.md](deployment.md) |
| Memory at idle and under load | 589 MB idle, 759 MB peak | psutil on the API process during the 50 requests |
| Test coverage | 88% of `resolveiq`, 224 tests | `pytest --cov=resolveiq` |

Measured on an Apple M3 laptop with 8 GB of memory, with the classifier on the MPS
backend. These times **exclude generation**: all 50 benchmark tickets come from the
public ticket dataset, which is about software and IT services, and the refusal rule
correctly found nothing in the Altura retail help center for any of them. With an
answer, the time is dominated by the language model, which is not measured here.

### Results by group

The only group the ticket data allows is ticket type
(`results/evaluation/group_metrics.csv`, test split):

| Ticket type | Tickets | Classifier macro F1 | High priority | Risk PR-AUC | Alert rate | High priority caught |
|---|---|---|---|---|---|---|
| Incident | 1,005 | 0.5164 | 41.9% | 0.7918 | 64.5% | 91.7% |
| Problem | 486 | 0.5303 | 38.7% | 0.7460 | 54.1% | 83.5% |
| Change | 274 | 0.4970 | 41.6% | 0.7637 | 61.7% | 84.2% |
| Request | 686 | 0.4781 | 33.1% | 0.6533 | 55.7% | 81.1% |
| All | 2,451 | 0.5213 | 38.8% | 0.7489 | 59.7% | 86.6% |

The classifier is similar across types. The risk model is clearly weaker on Request
tickets (PR-AUC 0.65 against 0.79 for incidents) and raises more false alarms there. See
failure case 4.

## 4. Plots

| Plot | File |
|---|---|
| Classifier confusion matrix | [results/classifier/confusion_matrix.png](../results/classifier/confusion_matrix.png) |
| Risk model calibration curve | [results/risk/calibration.png](../results/risk/calibration.png) |
| SHAP summary of the risk model | [results/risk/shap_summary.png](../results/risk/shap_summary.png) |
| Drift report with distributions | [results/monitoring/drift_report_shifted.html](../results/monitoring/drift_report_shifted.html) |

## 5. Failure cases

**1. Out-of-scope questions that mention the company are not refused.**
"What is the Altura stock price today?" and "Can I book a hotel room through Altura?"
reach similarities of 0.74 and 0.73, above the 0.62 threshold, because the company name
and retail vocabulary match the help articles. The similarity rule refused 0 of 3
holdout out-of-scope questions.
*Fix:* keep the prompt-level refusal as the second line of defence and measure it with a
language model; add a small classifier on question type (account, order, product versus
other), or require that the best passage shares a content word other than the company
name.

**2. Real ticket traffic does not match the knowledge base.**
All 50 benchmark tickets were refused by retrieval: the ticket dataset covers software
and IT services, while the help center describes a retail store. In the live service this
would show as a refusal rate near 100%.
*Fix:* monitor the share of refused drafts in the request log and alert when it rises;
for a real deployment, index the company's own help center, which the ingestion
pipeline already supports.

**3. The fine-tuned classifier is below its own baseline.**
DistilBERT reaches 0.52 macro F1 against 0.63 for TF-IDF. Validation F1 was still rising
at the last epoch, and a hand review found many errors between overlapping support
queues (a dashboard that stops updating labeled Technical Support, predicted Product
Support).
*Fix:* train for more epochs on a GPU, merge or redefine overlapping queues, and serve
the baseline until the transformer wins on validation.

**4. The risk model is weakest on Request tickets.**
PR-AUC is 0.65 on requests against 0.79 on incidents, and 56% of requests raise an
alert although only 33% are high priority. The single global threshold was chosen for
the overall 39% positive rate, so it alerts too often for a group with fewer
escalations; the errors within the group have not been reviewed by hand yet.
*Fix:* review a sample of false alarms on requests, try a per-type threshold or an
interaction with ticket type, and report results by type in every evaluation.

**5. Polite wording can still score high.**
"Please add my colleague to the workspace when convenient" scores 0.59 because the text
score keys on vocabulary that is often labeled high priority in this synthetic data
([risk_model.md](risk_model.md)).
*Fix:* add features for politeness and explicit time words ("when convenient", "no
rush"), and collect real escalation labels instead of the priority proxy.

## 6. Limits and next steps

- **Synthetic data.** The help center and the tickets are synthetic and come from
  different domains. Results describe the methods, not performance on a real company.
- **Proxy label.** The risk model predicts the dataset's priority judgment, not real
  escalations. There is no time-based split and no customer history.
- **Language model not measured.** Answer correctness and generation time (Table A
  columns) require a running language model and hand labeling.
- **No public demo yet.** Deployment needs a Hugging Face account and an API key; the
  steps are in [deployment.md](deployment.md).
- **English only.**

Next steps, in order: run the full RAG evaluation with a local and a hosted model; deploy
the Space and measure public response times; retrain the classifier longer or serve the
baseline; collect a real escalation label.
