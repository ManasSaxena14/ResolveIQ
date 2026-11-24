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
| `POST /analyze` | Draft reply, category, risk and word highlights for one ticket | 422 empty, too long or unknown fields; 503 language model down, partial result still returned |
| `POST /analyze/stream` | The same, as server-sent events: analysis first, then the draft as it is written | 422 before the stream starts |
| `POST /batch` | CSV of tickets in, list ranked by risk out | 413 file too large; 422 missing columns, empty rows, too many rows, not UTF-8 |
| `POST /feedback` | Store a thumbs up or down for a request | 404 unknown request ID; 422 invalid rating |
| `GET /health` | Status, model versions, index size | none |
| `GET /requests` | Newest analyzed tickets with outputs and feedback, no text | 422 limit outside 1 to 500 |
| `GET /stats` | Totals over the request log for the dashboard | none |

Every response carries an `X-Request-ID` header. A caller can send its own ID in the same
header to correlate logs.

## Rate limits and API keys

The routes above, except `/health`, are counted per caller over a sliding minute:

| Caller | Identified by | Limit (`service.rate_limit`) |
|---|---|---|
| Anonymous, including the web app | Client address | 30 requests a minute |
| With a key in `X-API-Key` | The key (keys come from `RESOLVEIQ_API_KEYS`, comma separated) | 300 requests a minute |

Browsers on another origin (for example the static site on Render) are allowed through
`RESOLVEIQ_CORS_ORIGINS`, a comma-separated list of origins.

Responses carry `X-RateLimit-Limit` and `X-RateLimit-Remaining`. Over the limit the API
answers **429** with `Retry-After` in seconds; an unknown key answers **401**. Counts are
kept in memory, which suits one server process.

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

`category.top` lists all ten queues, most likely first; the example shows three. The example was recorded with
the earlier models.

`draft_reply.status` is one of:

| Status | Meaning | HTTP |
|---|---|---|
| `ok` | Grounded draft with its sources | 200 |
| `refused` | The question is out of scope for the knowledge base; `text` holds the refusal sentence | 200 |
| `unavailable` | The language model failed or is not configured; category and risk are still valid | 503 |

`timings_ms` contains `retrieval`, `generation` (when the model answered), `classifier`,
`risk`, `explain` (highlights) and `total`.

`draft_reply.grounding` (when a draft was written) checks the draft's own citations:

```json
{"sentences": 3, "cited_sentences": 2, "citations": 2, "coverage": 0.6667,
 "invalid_citations": []}
```

`invalid_citations` lists citations that point at no retrieved source, a sign of an
invented reference.

`highlights` lists the words behind each decision, for highlighting the ticket:

```json
{
  "category": [{"term": "second", "weight": 1.0}, {"term": "twice", "weight": 0.94}],
  "risk": [{"term": "twice", "weight": 0.40}, {"term": "charged twice", "weight": 0.21}]
}
```

Both are found by taking each word out and measuring the change: `category` is scaled so
the strongest word is 1, `risk` is in log-odds and negative words lower the risk. For
tickets close to past ones, `risk` gives each term's exact contribution to the risk model's
text score instead. See [semantic_model.md](semantic_model.md).

## POST /analyze/stream

Same request body as `/analyze`. The response is `text/event-stream`:

| Event | Data | When |
|---|---|---|
| `analysis` | `request_id`, `category`, `risk`, `highlights`, `versions` | After about 0.3 s |
| `sources` | The retrieved help center sources | When retrieval found relevant articles |
| `token` | `{"text": "..."}`, a piece of the draft | While the language model writes |
| `done` | `{"response": <the /analyze response>, "partial": false}` | At the end |
| `error` | `{"detail": "..."}` | Only on an unexpected failure |

```bash
curl -N -X POST http://127.0.0.1:8000/analyze/stream \
  -H "content-type: application/json" -d '{"text": "Where is my refund?"}'
```

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

## GET /requests and GET /stats

Both feed the Monitoring page of the web app and read only the request log, so they never
return ticket text.

`GET /requests?limit=50` returns the newest rows first:

```json
[
  {
    "request_id": "e3b9d6718911",
    "created_at": "2026-11-17T08:31:02+00:00",
    "channel": "email",
    "category": "Billing and Payments",
    "confidence": 0.9936,
    "needs_review": false,
    "risk_score": 0.8229,
    "alert": true,
    "text_length": 122,
    "draft_status": "unavailable",
    "latency_ms": 918.0,
    "rating": "down"
  }
]
```

`GET /stats` returns totals over the whole log: `requests`, `alerts`, `needs_review`,
`mean_risk`, `mean_confidence`, counts per `categories` and `draft_status`,
`latency_median_ms`, `latency_p95_ms` and `feedback` (`total`, `up`, `down`, `down_rate`).

## Web app

When `frontend/dist` exists (`make web-build`), the API also serves the React app at `/`.
API routes are registered first, so they always take precedence. See
[frontend.md](frontend.md).

## Logging and privacy

- Each request is logged with its ID, path, status and duration.
- The `/analyze` log line and the SQLite request log hold the text length and a
  truncated SHA-256 hash, never the ticket text.
- Ticket text and documents are treated as untrusted: the prompt tells the language model
  to ignore instructions inside them.
