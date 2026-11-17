# API Reference

The FastAPI service exposes the copilot over HTTP. Start it locally with:

```bash
uvicorn resolveiq.api.main:app --port 8000
```

Models load once at startup (about 10 seconds on a laptop). Interactive documentation
with every schema is served at `http://127.0.0.1:8000/docs`. Request and response models
are defined in [`src/resolveiq/schemas.py`](../src/resolveiq/schemas.py) and checked by
contract tests in `tests/test_api.py`.

| Method and path | Purpose | Errors |
|---|---|---|
| `POST /analyze` | Draft reply, category and risk for one ticket | 422 empty, too long or unknown fields; 503 language model down, partial result still returned |
| `POST /batch` | CSV of tickets in, list ranked by risk out | 413 file too large; 422 missing columns, empty rows, too many rows, not UTF-8 |
| `POST /feedback` | Store a thumbs up or down for a request | 404 unknown request ID; 422 invalid rating |
| `GET /health` | Status, model versions, index size | none |

Every response carries an `X-Request-ID` header. A caller can send its own ID in the same
header to correlate logs.

## POST /analyze

Request:

```json
{
  "text": "I was charged twice for my order this month and nobody has replied to my last two emails.",
  "subject": "",
  "ticket_type": "",
  "channel": "email",
  "product": ""
}
```

| Field | Required | Notes |
|---|---|---|
| `text` | yes | Ticket body, 1 to 4,000 characters (`service.max_text_chars`) |
| `subject` | no | Up to 300 characters; used by all three models |
| `ticket_type` | no | `Incident`, `Problem`, `Request` or `Change`, a risk model feature |
| `channel` | no | Stored in the request log for group comparisons; not a model input |
| `product` | no | Appended to the retrieval question |

Response, recorded from the local service with the production models. No language model
was running, so the status is 503 and the draft is unavailable while its sources are
still returned:

```json
{
  "request_id": "3fc77d854fb2",
  "draft_reply": {
    "text": null,
    "status": "unavailable",
    "refused": false,
    "sources": [
      {"file": "billing/double-charges-and-pending-authorizations.md", "page": 1, "score": 0.6777},
      {"file": "support/escalations-and-complaints.md", "page": 1, "score": 0.6134}
    ],
    "error": "The language model is unavailable; showing the retrieved sources only"
  },
  "category": {
    "label": "Billing and Payments",
    "confidence": 0.9468,
    "needs_review": false,
    "top": [
      {"label": "Billing and Payments", "probability": 0.9468},
      {"label": "Human Resources", "probability": 0.0166},
      {"label": "Returns and Exchanges", "probability": 0.0162}
    ]
  },
  "risk": {
    "escalation_probability": 0.6652,
    "alert": true,
    "threshold": 0.23,
    "reasons": ["Wording typical of high-priority tickets: \"twice\", \"last\", \"charged\""]
  },
  "timings_ms": {"retrieval": 14.5, "classifier": 488.2, "risk": 46.8, "total": 1148.6},
  "versions": {
    "embedder": "BAAI/bge-small-en-v1.5",
    "llm": "ollama/llama3.2",
    "classifier": "distilbert-base-uncased-seed-42",
    "risk": "xgboost-943a93b (local)"
  }
}
```

`draft_reply.status` is one of:

| Status | Meaning | HTTP |
|---|---|---|
| `ok` | Grounded draft with its sources | 200 |
| `refused` | The question is out of scope for the knowledge base; `text` holds the refusal sentence | 200 |
| `unavailable` | The language model failed or is not configured; category and risk are still valid | 503 |

`timings_ms` contains `retrieval`, `generation` (when the model answered), `classifier`,
`risk` and `total`.

## POST /batch

Multipart upload with one field, `file`:

```bash
curl -F "file=@tickets.csv" http://127.0.0.1:8000/batch
```

The CSV needs a `body` or `text` column; `subject`, `type` and `id` are optional. Rows
without an `id` are numbered from 1. Limits: 2 MB (`service.max_upload_bytes`) and 2,000
rows (`service.max_batch_rows`). Only the risk model runs, so a full file scores in
seconds. Ticket text is never returned.

```json
{
  "count": 2,
  "alerts": 1,
  "items": [
    {"rank": 1, "id": "B", "escalation_probability": 0.91, "alert": true, "reasons": ["..."]},
    {"rank": 2, "id": "A", "escalation_probability": 0.08, "alert": false, "reasons": []}
  ]
}
```

## POST /feedback

```json
{"request_id": "3fc77d854fb2", "rating": "down", "comment": "Wrong queue"}
```

Returns `{"request_id": "...", "rating": "down", "stored": true}` or 404 when the ID was
never issued. Comments are masked for emails, phone numbers and long numbers before they
are stored. Feedback feeds the retrain rule in [monitoring.md](monitoring.md).

## GET /health

```json
{
  "status": "ok",
  "components": {"retriever": true, "llm_configured": true, "classifier": true,
                 "risk": true, "request_log": true},
  "versions": {"embedder": "...", "llm": "...", "classifier": "...", "risk": "..."},
  "index": {"files": 31, "chunks": 31},
  "uptime_seconds": 2.3
}
```

`status` is `degraded` when a component is missing, for example when no knowledge base
index exists. `llm_configured` reports the configuration only; the health check does not
call the language model.

## Logging and privacy

- Each request is logged with its ID, path, status and duration.
- The `/analyze` log line and the SQLite request log hold the text length and a
  truncated SHA-256 hash, never the ticket text.
- Ticket text and documents are treated as untrusted: the prompt tells the language model
  to ignore instructions inside them.
