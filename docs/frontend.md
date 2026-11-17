# Web App

The main user interface is a React and TypeScript single-page app in `frontend/`. FastAPI
serves the built files next to the API, so one process and one port run the whole
product. The Gradio page (`app.py`) stays as the lightweight demo for Hugging Face
Spaces.

![ResolveIQ web app analyzing a double-charge ticket](images/web_app.jpg)

## Pages

| Page | What it does | API calls |
|---|---|---|
| Analyze | Ticket form with type, channel and product, a character counter, example tickets and Ctrl or Cmd + Enter to submit. Shows the draft with linked help center sources and similarity bars, the category with its top probabilities and review flag, the risk gauge with the alert threshold and reasons, step timings and feedback buttons | `POST /analyze`, `POST /feedback` |
| Batch | Drag and drop a CSV, download a sample file, see summary tiles and a ranked table with an alerts-only filter, export the ranking as CSV | `POST /batch` |
| Monitoring | Volume, alert rate, review rate, latency and negative feedback, distributions of categories and draft outcomes, and the 50 most recent requests with their feedback | `GET /stats`, `GET /requests` |

The top bar shows a live status from `GET /health` every 30 seconds (with model versions
on hover) and switches between light and dark themes, following the system setting by
default.

Behaviour worth knowing:

- A `503` from `/analyze` is shown as a partial result: the draft card explains that the
  language model is unavailable and still lists the retrieved sources, while the category
  and risk cards render normally.
- Validation errors from the API (empty text, unknown fields) are shown as readable
  messages. Text over 4,000 characters is blocked before it is sent.
- Pages are addressed by hash (`/#/batch`, `/#/monitoring`), so a refresh keeps the page
  and no server-side routing is needed.
- No ticket text is stored in the browser. The theme choice is the only thing saved, in
  local storage.

## Running it

Requires Node.js 20 or later.

```bash
make web-install   # npm ci in frontend/
make web-build     # type check and build into frontend/dist
make api           # http://127.0.0.1:8000 now serves the app and the API
```

For development with hot reload, run the API and the Vite dev server side by side:

```bash
make api           # terminal 1: API on port 8000
make web-dev       # terminal 2: app on http://127.0.0.1:5173, API calls proxied to 8000
```

Set `RESOLVEIQ_API_URL` to proxy to an API elsewhere. `VITE_API_BASE` (at build time)
points the built app at an API on another origin, and `VITE_SOURCE_BASE_URL` changes where
source links point.

## Structure

```
frontend/
  index.html               page shell
  vite.config.ts           dev proxy, build and test settings
  src/
    main.tsx               entry point
    App.tsx                top bar, navigation, health status, theme
    index.css              design tokens for both themes and all component styles
    api/types.ts           response types mirroring src/resolveiq/schemas.py
    api/client.ts          fetch wrappers and ApiError
    lib/format.ts          formatting helpers (percent, durations, source titles, CSV)
    lib/examples.ts        example tickets and form limits
    components/            result cards and the risk gauge
    pages/                 Analyze, Batch and Monitoring
    test/                  test setup and fixtures
```

There is no UI component library and no router: the app has three pages and a small
design system, and staying with React alone keeps the bundle at about 76 KB gzipped.

## Quality checks

```bash
make web-test      # oxlint, TypeScript type check and Vitest
```

Tests use Vitest, Testing Library and jsdom with `fetch` mocked. They cover the
formatting helpers, the API client (partial results, validation messages, an unreachable
server, unknown feedback IDs), the three result cards and the analyze flow from typing a
ticket to storing feedback. CI runs lint, type check, tests and a production build in a
separate `frontend` job.

When the API changes, update `api/types.ts` to match `schemas.py`; the type check then
shows every place that needs to change.
