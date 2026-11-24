import { useState, type CSSProperties } from 'react'
import type { Highlights } from '../api/types'
import { highlightSegments } from '../lib/highlight'

type Lens = 'category' | 'risk'

const LENSES: Record<Lens, { label: string; note: string }> = {
  category: { label: 'Queue words', note: 'Stronger color, stronger pull toward this queue.' },
  risk: { label: 'Risk words', note: 'Red words raise the risk, green words lower it.' },
}

/** The customer's message, with the words behind each decision marked in place. */
export function TicketText({ text, highlights }: { text: string; highlights: Highlights | null }) {
  const [lens, setLens] = useState<Lens>('category')
  const terms = highlights?.[lens] ?? []
  const max = Math.max(...terms.map((t) => Math.abs(t.weight)), 0)
  const explained = !!highlights && (highlights.category.length > 0 || highlights.risk.length > 0)

  return (
    <section aria-label="Customer message">
      <p className="message">
        {highlightSegments(text, terms).map((segment, i) =>
          segment.weight === null ? (
            <span key={i}>{segment.text}</span>
          ) : (
            <mark
              key={i}
              data-kind={lens === 'category' ? 'queue' : segment.weight >= 0 ? 'raises' : 'lowers'}
              style={{ '--strength': `${Math.round(16 + 44 * (Math.abs(segment.weight) / (max || 1)))}%` } as CSSProperties}
            >
              {segment.text}
            </mark>
          ),
        )}
      </p>
      {explained && (
        <div className="lens">
          <div className="switch" role="group" aria-label="Marked words">
            {(Object.keys(LENSES) as Lens[]).map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={lens === key}
                disabled={!highlights[key].length}
                onClick={() => setLens(key)}
              >
                {LENSES[key].label}
              </button>
            ))}
          </div>
          <p className="meta muted">{LENSES[lens].note}</p>
        </div>
      )}
    </section>
  )
}
