# ResolveIQ design direction

**Concept: a triage instrument.** ResolveIQ reads a ticket and gives a verdict, so the
interface borrows from editorial layouts and measuring instruments rather than generic
SaaS dashboards: a calm paper or night surface, one sharp signal color, serif headlines
for judgment, monospaced numerals for measurements, and scales with tick marks instead
of decorative charts.

## Principles

1. **The verdict comes first.** After analysis, one strip answers route, risk and draft
   at a glance; the evidence follows below it.
2. **Numbers are measurements.** Every figure is set in a monospaced face with tabular
   digits, and every meter shows its scale and the alert threshold.
3. **One signal color.** Lime marks what the user should act on (focus, primary action,
   the active page). Risk uses its own semantic scale: green, amber, red.
4. **Quiet surfaces, sharp edges.** Hairline borders, no heavy shadows, a faint grain on
   the background, generous spacing.
5. **Motion explains change.** Results rise in sequence, meters fill to their value and
   numbers count up once. Nothing loops, and everything stops under reduced motion.

## Tokens

| Role | Night (default dark) | Paper (light) |
|---|---|---|
| Background | `#0d0d0b` | `#f2efe7` |
| Surface | `#151512` | `#fbf9f4` |
| Hairline | `#272722` | `#e1dbcd` |
| Ink | `#eceae3` | `#17150f` |
| Muted ink | `#a19e93` | `#615b4e` |
| Signal | `#cdf24a` | `#3d5a10` text, `#d5f25c` fill |
| Low / medium / high risk | `#6fd39a` / `#f2b544` / `#ff7a66` | `#24774a` / `#a86200` / `#bf3420` |

## Type

| Use | Face | Size |
|---|---|---|
| Page titles, verdict values | Instrument Serif | 30 to 48 px, fluid |
| Interface text | Geist | 14.5 px, line height 1.55 |
| Numbers, IDs, eyebrow labels | Geist Mono | 11 to 40 px, tabular |

Fonts are bundled with the app (`@fontsource`), so no request leaves for a font service.

## Logo

The mark is an R whose leg resolves into the lime verdict dot, set on a ticket-stub tile
with notched sides: a ticket comes in, a decision comes out. The tile stays ink in both
themes, with a hairline edge on dark backgrounds, and the name sets "Resolve" in Geist
with "IQ" in Instrument Serif italic in the signal color.

| Asset | Use |
|---|---|
| `src/components/Logo.tsx` | Mark and wordmark in the app |
| `public/favicon.svg` | Browser tab |
| `public/apple-touch-icon.png`, `public/icon-512.png` | Home screen and install icons |
| `public/og-image.png` | Link previews (1200 x 630) |
| `docs/images/logo-light.png`, `logo-dark.png` | README header |

Keep clear space of half the tile around the mark, never recolor the dot and never
place the mark below 16 px.

## Layout

- Desktop (1024 px and wider): a fixed left rail with the brand, numbered navigation,
  live status and theme switch; content up to 1180 px.
- Tablet (640 to 1023 px): a top bar with inline navigation.
- Phone (below 640 px): a compact top bar and a bottom tab bar within thumb reach;
  every grid collapses to one column and tables scroll inside their card.

## Motion

| Element | Motion | Duration |
|---|---|---|
| Result cards | Fade and rise, staggered 70 ms | 420 ms |
| Meters and bars | Width from zero to value | 700 ms |
| Risk number | Count up from zero | 650 ms |
| Buttons | Press scale to 0.98 | 120 ms |

All motion is disabled when the system asks for reduced motion.
