/** Formatting helpers shared by the pages. Pure functions, unit tested. */

export const SOURCE_BASE_URL: string =
  import.meta.env.VITE_SOURCE_BASE_URL ??
  'https://github.com/ManasSaxena14/ResolveIQ/blob/main/data/kb/'

export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return `${(value * 100).toFixed(digits)}%`
}

/** A percentage that never rounds a real chance down to nothing. */
export function share(value: number): string {
  return value > 0 && value < 0.01 ? 'under 1%' : percent(value)
}

export function milliseconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return '-'
  if (value >= 1000) return `${(value / 1000).toFixed(1)} s`
  return `${Math.round(value)} ms`
}

/** Ensure a message ends with a full stop before another sentence follows it. */
export function sentence(text: string): string {
  const trimmed = text.trim()
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`
}

export function sourceUrl(file: string): string {
  return SOURCE_BASE_URL + file.split('/').map(encodeURIComponent).join('/')
}

/** "billing/double-charges-and-pending-authorizations.md" -> "Double charges and pending authorizations" */
export function sourceTitle(file: string): string {
  const name = file.split('/').pop() ?? file
  const stem = name.replace(/\.[^.]+$/, '').replace(/[-_]+/g, ' ').trim()
  return (stem.charAt(0).toUpperCase() + stem.slice(1)).replace(/\baltura( plus)?\b/i, (name) =>
    name.replace(/\b[a-z]/g, (letter) => letter.toUpperCase()),
  )
}

export function sourceSection(file: string): string {
  const parts = file.split('/')
  return parts.length > 1 ? parts[0] : ''
}

export type RiskLevel = 'low' | 'medium' | 'high'

/** Low below the alert threshold, high from twice the threshold (or 0.6) upward. */
export function riskLevel(probability: number, threshold: number): RiskLevel {
  if (probability < threshold) return 'low'
  if (probability >= Math.max(0.6, threshold * 2)) return 'high'
  return 'medium'
}

export function relativeTime(iso: string, now: Date = new Date()): string {
  const then = new Date(iso)
  const seconds = Math.round((now.getTime() - then.getTime()) / 1000)
  if (Number.isNaN(seconds)) return iso
  if (seconds < 45) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  return then.toLocaleDateString()
}

/** Quote a CSV cell when needed. */
function csvCell(value: string | number): string {
  const text = String(value)
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
}

export function toCsv(header: string[], rows: (string | number)[][]): string {
  return [header, ...rows].map((row) => row.map(csvCell).join(',')).join('\n') + '\n'
}

export const SAMPLE_CSV = toCsv(
  ['id', 'subject', 'body', 'type'],
  [
    ['T-1001', 'Invoice copy', 'Could you send me a copy of last month\'s invoice?', 'Request'],
    [
      'T-1002',
      'Platform down',
      'Our whole team has been locked out since this morning and we have a deadline today. This is urgent.',
      'Incident',
    ],
    ['T-1003', 'Return window', 'How many days do I have to return headphones?', 'Request'],
    [
      'T-1004',
      'Charged twice',
      'I was charged twice for my order and nobody has answered my last two emails.',
      'Problem',
    ],
  ],
)
