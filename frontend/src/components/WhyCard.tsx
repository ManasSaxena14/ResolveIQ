import { useState, type CSSProperties } from 'react'
import type { Highlights } from '../api/types'
import { highlightSegments } from '../lib/highlight'

type Lens = 'category' | 'risk'

const LENS_COPY: Record<Lens, { label: string; note: string }> = {
  category: {
    label: 'Category',
    note: 'Words that pushed the classifier toward the predicted queue. Darker means stronger.',
  },
  risk: {
    label: 'Risk',
    note: 'Words that raise the escalation risk (red) or lower it (green), from the text model.',
  },
}

function markStyle(lens: Lens, weight: number, max: number): CSSProperties {
  const strength = Math.min(Math.abs(weight) / (max || 1), 1)
  const alpha = 0.16 + strength * 0.5
  const color =
    lens === 'category' ? 'var(--signal-mark)' : weight >= 0 ? 'var(--high-mark)' : 'var(--low-mark)'
  return { background: `color-mix(in srgb, ${color} ${Math.round(alpha * 100)}%, transparent)` }
}

/** The ticket text with the words behind each decision highlighted. */
export function WhyCard({
  subject,
  text,
  highlights,
  index = 4,
}: {
  subject: string
  text: string
  highlights: Highlights
  index?: number
}) {
  const [lens, setLens] = useState<Lens>('category')
  const terms = highlights[lens]
  const max = Math.max(...terms.map((t) => Math.abs(t.weight)), 0)
  const full = subject.trim() ? `${subject.trim()}\n\n${text}` : text
  const segments = highlightSegments(full, terms)
  const top = [...terms].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 5)

  if (!highlights.category.length && !highlights.risk.length) return null

  return (
    <section className="card reveal" style={{ '--i': index } as CSSProperties} aria-labelledby="why-title">
      <div className="card-header">
        <h2 id="why-title" className="card-title">
          Why these results
        </h2>
        <div className="segmented" role="group" aria-label="Explain">
          {(Object.keys(LENS_COPY) as Lens[]).map((key) => (
            <button
              key={key}
              type="button"
              aria-pressed={lens === key}
              onClick={() => setLens(key)}
              disabled={!highlights[key].length}
            >
              {LENS_COPY[key].label}
            </button>
          ))}
        </div>
      </div>
      <p className="why-text">
        {segments.map((segment, i) =>
          segment.weight === null ? (
            <span key={i}>{segment.text}</span>
          ) : (
            <mark
              key={i}
              style={markStyle(lens, segment.weight, max)}
              title={`${segment.term}: ${segment.weight > 0 ? '+' : ''}${segment.weight.toFixed(2)}`}
            >
              {segment.text}
            </mark>
          ),
        )}
      </p>
      <p className="small faint" style={{ marginTop: 12 }}>
        {LENS_COPY[lens].note}
      </p>
      {top.length > 0 && (
        <ol className="why-terms" aria-label={`Top ${LENS_COPY[lens].label.toLowerCase()} words`}>
          {top.map((t) => (
            <li key={t.term}>
              <span>{t.term}</span>
              <span className="mono">
                {lens === 'risk' && t.weight > 0 ? '+' : ''}
                {t.weight.toFixed(2)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
