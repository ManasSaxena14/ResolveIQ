# Web App

The main user interface is a React and TypeScript single-page app in `frontend/`. FastAPI
serves the built files next to the API, so one process and one port run the whole
product. The Gradio page (`app.py`) stays as a lighter alternative.

![Triage page with the inbox ranked by risk and a lockout ticket sorted into Service Outages and Maintenance](images/web_app.jpg)

## Design

The subject is a sorting office: routing a support ticket is sorting mail into one of ten
queues. The full direction, tokens and rules are in
[frontend/DESIGN.md](../frontend/DESIGN.md).

- **The sorting rack** is the one memorable element: ten pigeonholes, one per queue. The
  ticket's slip drops into its queue and every other slot fills in proportion to its
  probability, so the rack is data, not decoration.
- **The inbox comes first.** Triage opens on tickets ranked by risk and shows the riskiest
  one already sorted, so the page explains itself without any typing.
- **One typeface.** Archivo is bundled with the app, so no request goes to a font service.
  Its width axis is condensed on the rack labels and wide in headings.
- **Color with roles.** Violet marks actions and the selection; red, amber and green belong
  to risk and status. Every text pair is at least 4.5:1 in both themes.
- **Plain, honest wording.** Labels say what things are, and the About page puts every
  result next to the simple method it has to beat, followed by the limits.
- **Motion that explains change.** The slip drops once, and controls answer a press.
  Reduced motion turns all of it off.
- **Every screen size.** Two panes on desktop, one at a time on phones with a bottom tab
  bar. Tables become one block per row and the rack lays its slots on their side.

| Night theme, About | Phone, a ticket |
|---|---|
| ![About page in the night theme](images/web_app_night.jpg) | ![A sorted ticket on a phone](images/web_app_phone.jpg) |

## Pages

| Page | What it does | API calls |
|---|---|---|
| Triage | An inbox of sample tickets ranked by risk with search and a flagged-only filter. Opening a ticket shows its message with the deciding words marked, the rack, the risk scale with its reasons, a drafted reply that is written out live with numbered citations and a grounding line, step timings and a vote. New tickets join the inbox | `POST /batch`, `POST /analyze/stream`, `POST /feedback` |
| Batch | Choose or drop a CSV or score the sample, search and filter the ranking, export it as CSV | `POST /batch` |
| Monitoring | Volume, flag rate, double-check rate, latency and votes marked wrong, tickets by queue and the 50 latest requests | `GET /stats`, `GET /requests` |
| About | What happens to a ticket, how well each part works and what it cannot do | none |

The header shows the server's state from `GET /health` every 30 seconds and switches
between the day and night themes.

Keyboard: `/` searches, `n` starts a new ticket, the arrow keys move through the inbox and
Enter moves into the ticket. A skip link leads to the content.

Behaviour worth knowing:

- Opening Triage scores the sample inbox with one `/batch` call. A ticket is only analyzed
  when it has stayed open for 300 ms, once per visit, so moving through the list is cheap.
- A `503` from the API is shown as a partial result: the reply section explains that the
  language model is unavailable and lists the articles found, while the queue and risk
  stand.
- When the server cannot be reached, the app says so, keeps checking and does again
  whatever failed once it is back. On free hosting a slow first answer is reported as the
  server waking up.
- For tickets close to past ones, the XGBoost part of the risk uses its own quick guess at
  the queue, so a reason can name a different chance than the rack. The app says so.
- Pages and tickets are addressed by hash (`/#/triage/<ticket>`, `/#/batch`), so links, a
  refresh and the back button work and no server-side routing is needed.
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
    App.tsx                header, tab bar, server state and theme
    styles/                base (tokens, type, controls), shell, triage and pages
    api/types.ts           response types mirroring src/resolveiq/schemas.py
    api/client.ts          fetch wrappers, the event stream and ApiError
    lib/useTriage.ts       the inbox, its ranking and each ticket's analysis
    lib/inbox.ts           sample tickets and form limits
    lib/router.ts          hash routes
    lib/format.ts          formatting helpers (percent, durations, source titles, CSV)
    lib/highlight.ts       splits a message into marked and plain parts
    components/            rack, risk scale, ticket list and text, draft, composer, logo
    pages/                 Triage, Batch, Monitoring and About
    test/                  test setup, fixtures and a stand-in API
```

There is no UI component library and no router package: the app has four pages and a small
design system, and staying with React alone keeps the script at about 83 KB gzipped.

## Quality checks

```bash
make web-test      # oxlint, TypeScript type check and Vitest
```

Tests use Vitest, Testing Library and jsdom with `fetch` mocked. They cover the helpers
and routes, the API client (partial results, validation messages, streams, an unreachable
server), each component, and whole pages against a stand-in API: the ranked inbox, sorting
a ticket, keyboard use, writing a ticket, voting, failure and recovery, the waking server,
batch scoring, monitoring and About. CI runs lint, type check, tests and a production
build in a separate `frontend` job.

When the API changes, update `api/types.ts` to match `schemas.py`; the type check then
shows every place that needs to change.
