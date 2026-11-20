# Deployment

ResolveIQ runs without Docker. The same code serves three targets.

| Target | Command | Use |
|---|---|---|
| Local | `make web-build` then `uvicorn resolveiq.api.main:app` (React app and API), or `python app.py` (Gradio) | Development and live demos |
| Oracle Cloud (free) | `deploy/oracle/setup.sh`: the full service on an Always Free server behind Caddy | Public demo that never sleeps |
| Render (free) | `render.yaml`: the lite API and the web app as a static site | Public demo that wakes fast |

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

## Oracle Cloud Always Free (recommended: never sleeps, full project)

Oracle's Always Free tier gives a real Linux server at no charge: on the Ampere A1 shape
2 CPU cores and 12 GB of memory (Oracle halved this from 4 and 24 in June 2026; check the
current limit in your console), and up to 200 GB of disk that keeps its data. That runs
the **full** service, PyTorch included, with room to spare, and it never sleeps, so there
is no wake-up delay and the request log and feedback persist. Caddy serves it over free
HTTPS. No Docker is used.

| File in `deploy/oracle/` | Runs on | What it does |
|---|---|---|
| `setup.sh` | Server, once | Installs Python 3.12 and Caddy, creates a `resolveiq` user, clones the repository into `/opt/resolveiq`, installs `requirements.txt`, builds the knowledge base index, opens ports 80 and 443 and installs the service |
| `resolveiq.service` | Server | systemd service: starts on boot, restarts on failure, listens only on 127.0.0.1, hardened |
| `Caddyfile` | Server | HTTPS with a free certificate, compression, security headers, streamed drafts passed through unbuffered |
| `push.sh` (`make oracle-push`) | Your computer | Copies the trained models and the built web app over SSH, then restarts the service |
| `update.sh` | Server | Pulls the latest code, updates packages and restarts |

### 1. Create the account and the server

1. Sign up at Oracle Cloud Free Tier. A card is asked for identity verification only;
   Always Free resources are not charged. **Do not upgrade the account to Pay As You
   Go**, which is what allows charges. Pick a home region near you; it cannot be changed.
2. In the console: **Compute > Instances > Create instance**.
   - Image: **Canonical Ubuntu 24.04** (the aarch64 build).
   - Shape: **Ampere, VM.Standard.A1.Flex**, 2 OCPUs and 12 GB (marked "Always Free
     eligible").
   - Networking: keep "Assign a public IPv4 address" on.
   - SSH keys: upload your public key (for example `~/.ssh/id_ed25519.pub`).
   - If Oracle reports "out of capacity", try another availability domain or try again
     later; free capacity is shared.
3. Open the web ports in Oracle's network: **Networking > Virtual cloud networks** > your
   network > the subnet > its **security list** > **Add ingress rules**: source
   `0.0.0.0/0`, TCP, destination port `80`, and again for `443`.

### 2. Install ResolveIQ on the server

Note the server's public IP from the instance page, then on your computer:

```bash
ssh ubuntu@<PUBLIC_IP>
```

On the server:

```bash
curl -fsSLO https://raw.githubusercontent.com/ManasSaxena14/ResolveIQ/main/deploy/oracle/setup.sh
```
```bash
PUBLIC_IP=<PUBLIC_IP> bash setup.sh
```

This takes about 10 minutes, mostly installing PyTorch. With your own domain pointing at
the server, use `DOMAIN=support.example.com` instead of `PUBLIC_IP`; without one, the site
is served at `https://<ip-with-dashes>.sslip.io`, a free DNS name that points at your IP.

### 3. Copy the models from your computer

The trained models are not in git. From the project folder on your computer:

```bash
make oracle-push HOST=ubuntu@<PUBLIC_IP>
```

This builds the web app, copies `models/classifier/production`, `models/risk` and
`frontend/dist` (never sync-tool duplicates or hidden files) and starts the service. Run it
again after retraining a model.

### 4. Add the language model key

On the server, put your free Groq key (account without billing) in the settings file and
restart:

```bash
sudo nano /etc/resolveiq/resolveiq.env
```
```bash
sudo systemctl restart resolveiq
```

Then open `https://<ip-with-dashes>.sslip.io`.

### Running it

| Task | Command (on the server) |
|---|---|
| Live logs | `journalctl -u resolveiq -f` |
| Status | `systemctl status resolveiq` |
| Update to the latest code | `bash /opt/resolveiq/deploy/oracle/update.sh` |
| Restart after editing settings | `sudo systemctl restart resolveiq` |

Things to know:

- **Idle servers.** Oracle may stop an Always Free server whose CPU, network and memory
  use stay very low for 7 days. It is stopped, not deleted: start it again from the
  console and the service starts on boot.
- **Firewall.** Oracle's Ubuntu images block everything except SSH in the server's own
  firewall; `setup.sh` adds rules for ports 80 and 443 and saves them.
- These scripts were checked for syntax and logic on a development machine, not on a
  live Oracle server; if a step fails, the output names it and the script can be run again.

## Render (free tier)

The no-card option. Render's free web services have 512 MB of memory and sleep after 15 minutes without
traffic. The full service needs about 760 MB, so Render runs the **lite runtime**: the same
models as 8-bit ONNX graphs with a precomputed index and no PyTorch. It uses 347 MB at idle
and 363 MB under load, starts in about 2.6 seconds once the container is up, and agrees
with the full models closely (see [model_card_lite.md](model_card_lite.md)).

`render.yaml` defines two free services:

| Service | What it is | Sleeps |
|---|---|---|
| `resolveiq-api` | FastAPI in the lite runtime; the build installs `requirements-lite.txt`, precompiles the code and downloads the lite bundle | After 15 idle minutes |
| `resolveiq-web` | The React app as a static site | Never; it shows visitors when the API is waking |

1. Build and publish the lite bundle (once, and again after retraining):

   ```bash
   python scripts/export_lite.py
   hf auth login
   python scripts/publish_models.py lite --repo-id <user>/resolveiq-lite
   ```

2. In Render choose **New > Blueprint**, connect the GitHub repository and apply
   `render.yaml`. Fill in the values it asks for:

   | Service | Variable | Value |
   |---|---|---|
   | `resolveiq-api` | `RESOLVEIQ_LITE_REPO` | `<user>/resolveiq-lite` |
   | `resolveiq-api` | `RESOLVEIQ_CORS_ORIGINS` | the static site address, e.g. `https://resolveiq-web.onrender.com` |
   | `resolveiq-api` | `GROQ_API_KEY` | free tier key, account without billing |
   | `resolveiq-web` | `VITE_API_BASE` | the API address, e.g. `https://resolveiq-api.onrender.com` |

   Render adds a suffix when a name is taken; use the addresses it shows. After changing
   `VITE_API_BASE`, redeploy the static site so the build picks it up.

3. Optional, keep the API awake: set the repository variable `RENDER_API_URL` to the API
   address (GitHub > Settings > Secrets and variables > Actions > Variables). The
   `keep-alive` workflow then pings `/health` every 14 minutes. One service running all
   month uses about 744 of Render's 750 free instance hours, so leave other free Render
   services off, or skip this step and let the API sleep.

**How fast is waking up?** Without the keep-alive, the first request after a quiet period
waits for Render to start the container, typically 30 to 60 seconds on the free tier,
plus about 3 seconds for the service itself. The bytecode is compiled at build time and
the models are part of the build, so nothing is downloaded or compiled on wake-up. The web
app loads instantly from the static site, pings the API as soon as it opens, and tells the
visitor the server is waking while they type. Every request after that takes milliseconds
plus the language model.

Try the lite runtime locally exactly as Render runs it:

```bash
RESOLVEIQ_RUNTIME=lite uvicorn resolveiq.api.main:app --port 8000
```

## Hugging Face Spaces (needs a paid plan)

**Not free any more.** Hugging Face now requires a paid plan (PRO for personal
accounts) to create Spaces that run code, Gradio and Docker alike, even on the CPU
basic hardware. Free accounts can only create static Spaces, plus up to two Gradio
Spaces on ZeroGPU, which this project has not been tested on. Because ResolveIQ is
kept free, use the Oracle Cloud or Render deployment above. `scripts/deploy_space.py`
refuses to upload unless it is run with `--paid-plan` (`make deploy-space PAID=1`).

The rest of this section applies only to an account with the paid plan.

The Space runs `space.py`, which serves the React web app and the API from one process on
the Space's port. The trained models are not in git, so they are published to the Hub
first and downloaded when the Space starts; the web app is built locally and uploaded with
the code, because a Space has no Node.js.

1. Log in and publish the models (needs a Hugging Face **write** token):

   ```bash
   hf auth login
   python scripts/publish_models.py classifier --repo-id <user>/resolveiq-ticket-classifier
   python scripts/publish_models.py risk --repo-id <user>/resolveiq-escalation-risk
   ```

2. Build, stage and upload the Space, and set the two model variables in one step:

   ```bash
   make deploy-space PAID=1 SPACE=<user>/resolveiq CLASSIFIER=<user>/resolveiq-ticket-classifier RISK=<user>/resolveiq-escalation-risk
   ```

   `scripts/deploy_space.py` builds the frontend, copies only what the Space needs (the
   package, configs, the help center, the built app, `space.py` and `requirements.txt`;
   never `.env`, models, logs or local data), writes the Space README header and uploads
   it. Add `--dry-run` to list the files without uploading.

3. In the Space settings add the language model:

   | Name | Kind | Value |
   |---|---|---|
   | `RESOLVEIQ_LLM_PROVIDER` | Variable | `groq` (Ollama cannot run on Spaces) |
   | `GROQ_API_KEY` | Secret | your key |
   | `RESOLVEIQ_API_KEYS` | Secret, optional | keys for programmatic clients |

On first start the Space downloads the two models, builds the knowledge base index from
`data/kb/` and downloads the embedding model, which takes a few minutes. The MLflow
registry is not available there, so the service logs a warning and uses the downloaded
folders. On a Space, `space.py` trusts the proxy's `X-Forwarded-For` header so that each
visitor gets their own rate limit.

Try exactly what the Space runs before deploying:

```bash
make web-build && make space     # http://127.0.0.1:7860
```

The older Gradio page (`app.py`) still works as a lighter alternative: set
`app_file: app.py` in the Space README.

## Costs: everything is free

The project uses only free software and free tiers. Nothing in it needs a payment method.

| Part | What is used | Cost |
|---|---|---|
| Code, data and models | Open-source libraries (MIT, BSD, Apache), fonts under the SIL Open Font License, the public ticket dataset (CC BY-NC 4.0) and open models (DistilBERT, bge-small, XGBoost) | Free |
| Language model, local | Ollama with llama3.2 on your own machine | Free |
| Language model, hosted | Groq (`llama-3.1-8b-instant`) or Gemini (`gemini-2.5-flash`) on their free tiers, or an OpenRouter `:free` model | Free, with daily request limits |
| Model storage | Hugging Face Hub, public model repositories | Free |
| Public demo | Oracle Cloud Always Free (a server that never sleeps; card for identity check, not charged), or Render's free web service plus a free static site (sleeps after 15 idle minutes, no card) | Free |
| HTTPS | Caddy with a Let's Encrypt certificate, on a free sslip.io name or your own domain | Free |
| Keep-alive | A scheduled GitHub Actions workflow pinging the Render API | Free on a public repository, within Render's 750 free instance hours |
| Experiment tracking | MLflow with a local SQLite file | Free |
| Continuous integration | GitHub Actions on a public repository | Free |

Guards that keep it free:

- `llm.free_only: true` (the default) makes the config check reject an OpenRouter model
  that is not a `:free` variant, so the service cannot start on a paid model by mistake.
- Groq and Gemini only charge once billing is set up on the account. Create the keys on
  accounts without billing, and the free tier simply returns a rate-limit error (HTTP 429)
  when its daily quota is used; the service then shows the draft as unavailable and still
  returns the category and risk.
- Hugging Face Spaces that run code now need a paid plan, so they are not part of the
  free setup; `scripts/deploy_space.py` only uploads with an explicit `--paid-plan`.
- On Oracle Cloud, stay on the Always Free shapes and do not upgrade the account to Pay
  As You Go; the setup below uses only Always Free resources.

## Security habits

- Keys live in `.env` locally and in Space secrets when hosted. `.env` is in
  `.gitignore`; rotate any key that was ever committed.
- Ticket text is limited to 4,000 characters and batch files to 2 MB and 2,000 rows.
- API routes are rate limited per caller: 30 requests a minute without a key, 300 with a
  key from `RESOLVEIQ_API_KEYS` (`service.rate_limit`). Over the limit the API answers
  429 with `Retry-After`; an unknown key gets 401.
- Ticket text and documents are treated as untrusted; the prompt tells the language model
  to ignore instructions inside them, and a test checks this.
- The request log stores text length and a hash, never the text.
- The demo page tells visitors not to enter personal data, and all demo data is
  synthetic.

## Docker

Not used. Every target above runs from a virtual environment and `requirements.txt`, and
the CI pipeline tests the same way.
