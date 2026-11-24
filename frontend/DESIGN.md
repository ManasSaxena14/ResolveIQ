# ResolveIQ design direction

**Subject: a sorting office.** Routing a support ticket is sorting mail: each ticket goes
into one of ten queues. The interface is built around that, and everything else stays quiet.

**The one memorable element** is the sorting rack: ten pigeonholes, one per queue. The
ticket's slip drops into the queue it belongs to, and each other slot fills in proportion
to its probability. It is data, not decoration: the rack shows all ten probabilities.

## Pages and their patterns

| Page | Pattern | The one job |
|---|---|---|
| Triage | Split pane | Work through tickets, riskiest first |
| Batch | List / index | Score a file of tickets and act on the ranking |
| Monitoring | Dashboard | Glance at what the service has done |
| About | Marketing: about | Understand how it works and how well |

## Color

Painted steel, white paper, bottle-green ink and rubber-stamp violet. Red, amber and green
belong to risk and status only. Every text pair is at least 4.5:1 in both themes.

| Role | Day | Night shift |
|---|---|---|
| Ground | `#EDF2EE` | `#0E2A20` |
| Paper | `#FFFFFF` | `#163A2D` |
| Steel (structure) | `#CFDDD2` | `#1F4838` |
| Bay (inside the rack) | `#B5C8BA` | `#0A2219` |
| Ink | `#123B2C` | `#E6F0E9` |
| Muted ink | `#48665A` | `#A9C2B4` |
| Stamp (actions, selection) | `#4331C4` | `#B0A6FF` |
| High, elevated, low risk | `#B3261E` `#875400` `#1D6B45` | `#FF9A8C` `#F0BF62` `#84D9AA` |

## Type

One family, Archivo, bundled with the app. Its width axis carries meaning: condensed on
the narrow rack labels, normal for text, wide for headings. Three weights (400, 500, 600).
Sizes: 12, 14, 16, 20, 24 and 32, plus 48 for the About headline. Numbers use tabular figures. Sentence case everywhere.

## Layout and motion

- Spacing in multiples of 4 and 8. Text blocks stay under 70 characters.
- Two breakpoints: 768 and 1280. Below 768 the ticket list fills the screen and opening a
  ticket shows it as its own page with a way back; navigation moves to the bottom.
- Tables become one block per row on phones, each value next to its column name, and the
  rack lays its pigeonholes on their side when it is too narrow for five plates.
- Touch targets are at least 44 px. No page scrolls sideways at 375 px.
- Motion is one moment: the slip dropping into its slot (300 ms, transform only), plus
  100 to 200 ms responses to a press or selection. Reduced motion turns all of it off.

## Logo

An R whose leg resolves into a dot, on a ticket-stub tile with notched sides: a ticket
comes in, a decision comes out. The tile is bottle green, the dot stamp violet. Never
below 16 px, never recolored.

## Deliberately avoided

Cream paper with a serif headline, near-black with one acid accent, monospace or all-caps
micro-labels, numbered headings that are not a sequence, a single italic word in a
headline, glass blur, gradients, and cards that fade up on every page.
