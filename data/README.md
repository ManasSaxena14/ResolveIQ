# Data

## Knowledge base (`data/kb/`)

The knowledge base is the help center of **Altura**, a fictional online electronics
retailer. It contains 31 articles across orders, shipping, returns, billing, accounts,
membership, warranty, support, policies and products, stored as Markdown, HTML and plain
text to exercise every supported loader.

The articles are **synthetic**. They were written for this project so that every answer
in the RAG evaluation set has a known ground truth. They do not describe any real company,
and results measured on them do not prove performance on a real company's documentation.

To use a different knowledge base, replace the contents of `data/kb/` (PDF, TXT, Markdown
and HTML are supported) and rebuild the index with `make index`. Check that the license of
any public documentation allows reuse before copying it into the repository.

## Shop tickets (`data/seed/`)

`retail_tickets.csv` holds 288 tickets written for this project, about 29 per queue, 85 of
them marked urgent. They train the [semantic model](../docs/semantic_model.md). The test
tickets in `eval/ticket_testset.csv` were written separately, and a test checks that no
ticket is in both files. They describe no real customer and are MIT licensed like the code.

## Ticket dataset (`data/raw/`, `data/processed/`)

| Item | Value |
|---|---|
| Source | [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) on the Hugging Face Hub |
| File | `aa_dataset-tickets-multi-lang-5-2-50-version.csv` |
| Revision | `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb` (pinned in `configs/config.yaml`) |
| SHA-256 | `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0` |
| Downloaded | 2026-11-15 |
| License | CC BY-NC 4.0 (non-commercial use with attribution) |
| Nature | Synthetic customer support emails labeled with queue, priority, type and tags |

The dataset is not redistributed in this repository. Recreate it with:

```bash
make data    # scripts/download_data.py, then scripts/prepare_tickets.py
```

### Preparation

1. Keep English tickets only (16,338 of 28,587 rows).
2. Build the model input as subject plus body. Escaped line breaks (`\n`, `<br>`) are
   restored and whitespace is normalized.
3. Mask emails, URLs, phone numbers and long digit strings
   (`src/resolveiq/privacy.py`). The source data is already anonymized with placeholders
   such as `<tel_num>`; masking is applied as a safety net.
4. Remove duplicates after normalizing case and punctuation (1 removed). Texts that
   appear with conflicting labels would be dropped entirely (none found).
5. Split 70/15/15 into train, validation and test, stratified by queue with seed 42.
   The dataset has no timestamps, so a time-based split is not possible.

The `answer` and `tag_*` columns are never used as model inputs: answers are written
after the ticket arrives, and tags largely restate the queue.

### Label: routing queue

| Queue | Train | Test |
|---|---|---|
| Technical Support | 3,315 | 711 |
| Product Support | 2,151 | 461 |
| Customer Service | 1,686 | 362 |
| IT Support | 1,360 | 291 |
| Billing and Payments | 1,117 | 239 |
| Returns and Exchanges | 574 | 123 |
| Service Outages and Maintenance | 464 | 100 |
| Sales and Pre-Sales | 359 | 77 |
| Human Resources | 244 | 52 |
| General Inquiry | 165 | 35 |

The classes are imbalanced by a factor of 20, so macro F1 is the headline metric and
both models use class-balanced weighting. Full counts, cleaning statistics and token
lengths are in `results/classifier/data_profile.json`.

### Label: escalation (proxy)

The escalation risk model (Phase 3) uses the same splits. The dataset has no escalation
field, so a ticket counts as an escalation when its `priority` is `high`; `priority` is
therefore never a model input. This is a proxy label, documented with a leakage
checklist in [docs/risk_model.md](../docs/risk_model.md).

| Split | Tickets | High priority | Rate |
|---|---|---|---|
| Train | 11,435 | 4,460 | 39.0% |
| Validation | 2,451 | 935 | 38.1% |
| Test | 2,451 | 950 | 38.8% |

Features are written to `data/processed/risk/` by `scripts/build_risk_features.py` and
are not committed.
