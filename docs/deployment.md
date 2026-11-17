# Deployment

ResolveIQ runs without Docker. The same code serves three targets.

| Target | Command | Use |
|---|---|---|
| Local | `make web-build` then `uvicorn resolveiq.api.main:app` (React app and API), or `python app.py` (Gradio) | Development and live demos |
| Hugging Face Spaces | `app.py` with the Gradio SDK | Public demo |
| Render (optional) | `uvicorn resolveiq.api.main:app --host 0.0.0.0 --port $PORT` | Public API |

## Local

Requirements: Python 3.12 on macOS or Linux, about 2 GB of disk for packages and models.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install                 # runtime packages and the package itself
cp .env.example .env         # add a key only if you use a hosted language model
make index                   # build the knowledge base index (about 1 minute)
make data risk-features risk # data and the risk model (a few minutes)
make baseline classifier evaluate-classifier   # classifier (about 1 hour per seed on a laptop)
make web-install web-build   # React web app, needs Node.js 20 or later
make api                     # app and API on http://127.0.0.1:8000, API docs at /docs
make ui                      # optional: Gradio page on http://127.0.0.1:7860
```

Without a language model the service still returns the category, the risk and the
retrieved sources, and marks the draft unavailable. For drafts, either install
[Ollama](https://ollama.com) and run `ollama pull llama3.2`, or set
`RESOLVEIQ_LLM_PROVIDER=groq` (or `gemini`, `openrouter`) and the matching API key in
`.env`.

Optional: `make register` registers the trained models in the MLflow registry, and the
service then loads the `production` alias.

On macOS, `app.py`, the API and the monitoring scripts set `OMP_NUM_THREADS=1` before
importing PyTorch or XGBoost, because the two libraries bundle different OpenMP runtimes
that crash when both start worker threads (see [risk_model.md](risk_model.md)).

## Hugging Face Spaces

The Space runs `app.py`, the Gradio page; the React app needs the FastAPI server and is
deployed with it (locally or on Render). The trained models are not in git, so they are published to the
Hub first and downloaded when the Space starts.

1. Publish the models (needs a Hugging Face write token from `hf auth login`):

   ```bash
   python scripts/publish_models.py classifier --repo-id <user>/resolveiq-ticket-classifier
   python scripts/publish_models.py risk --repo-id <user>/resolveiq-escalation-risk
   ```

2. Create a Space with the Gradio SDK and push this repository to it. The Space reads its
   settings from a YAML header at the top of its `README.md`, so add this header to the
   copy of the README pushed to the Space:

   ```yaml
   ---
   title: ResolveIQ
   sdk: gradio
   app_file: app.py
   python_version: "3.12"
   license: mit
   ---
   ```

3. In the Space settings add:

   | Name | Kind | Value |
   |---|---|---|
   | `RESOLVEIQ_CLASSIFIER_REPO` | Variable | `<user>/resolveiq-ticket-classifier` |
   | `RESOLVEIQ_RISK_REPO` | Variable | `<user>/resolveiq-escalation-risk` |
   | `RESOLVEIQ_LLM_PROVIDER` | Variable | `groq` (Ollama cannot run on Spaces) |
   | `GROQ_API_KEY` | Secret | your key |

On first start the Space downloads the two models, builds the knowledge base index from
`data/kb/` and downloads the embedding model, which takes a few minutes. The MLflow
registry is not available there, so the service logs a warning and uses the downloaded
folders. The service used about 760 MB of memory at peak in local measurements, which
fits the free CPU hardware; check the current hardware limits on the Spaces pricing page.

## Render (optional)

Create a web service from the GitHub repository:

| Setting | Value |
|---|---|
| Build command | `pip install -r requirements.txt && pip install -e . --no-deps && cd frontend && npm ci && npm run build` |
| Start command | `uvicorn resolveiq.api.main:app --host 0.0.0.0 --port $PORT` |
| Environment | the same variables as the Space |

Free instances have little memory (512 MB at the time of writing), less than the service
needs with PyTorch and DistilBERT loaded, so treat this target as optional or use a paid
instance.

## Security habits

- Keys live in `.env` locally and in Space secrets when hosted. `.env` is in
  `.gitignore`; rotate any key that was ever committed.
- Ticket text is limited to 4,000 characters, batch files to 2 MB and 2,000 rows, and
  the UI queue to two concurrent jobs.
- Ticket text and documents are treated as untrusted; the prompt tells the language model
  to ignore instructions inside them, and a test checks this.
- The request log stores text length and a hash, never the text.
- The demo page tells visitors not to enter personal data, and all demo data is
  synthetic.

## Docker

Not used. Every target above runs from a virtual environment and `requirements.txt`, and
the CI pipeline tests the same way.
