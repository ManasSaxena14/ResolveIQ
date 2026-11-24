import { useRef, useState, type KeyboardEvent } from 'react'
import type { Score, ScoreState } from '../lib/useTriage'
import type { Ticket } from '../lib/inbox'

interface Props {
  tickets: Ticket[]
  scores: Record<string, Score>
  scoreState: ScoreState
  selected: string | null
  onSelect: (id: string) => void
  onEnter: () => void
  onRetry: () => void
}

export function TicketList({ tickets, scores, scoreState, selected, onSelect, onEnter, onRetry }: Props) {
  const [query, setQuery] = useState('')
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const list = useRef<HTMLUListElement>(null)

  const needle = query.trim().toLowerCase()
  const shown = tickets.filter(
    (t) =>
      (!needle || `${t.subject} ${t.text}`.toLowerCase().includes(needle)) &&
      (!flaggedOnly || scores[t.id]?.alert),
  )

  const onKeyDown = (event: KeyboardEvent) => {
    const step = event.key === 'ArrowDown' ? 1 : event.key === 'ArrowUp' ? -1 : 0
    if (event.key === 'Enter' && selected) {
      event.preventDefault()
      const id = (event.target as HTMLElement).dataset.id
      if (id) onSelect(id)
      onEnter()
      return
    }
    if (!step || !shown.length) return
    event.preventDefault()
    const at = shown.findIndex((t) => t.id === selected)
    const next = shown[Math.min(Math.max(at + step, 0), shown.length - 1)]
    onSelect(next.id)
    list.current?.querySelector<HTMLElement>(`[data-id="${next.id}"]`)?.focus()
  }

  return (
    <div className="inbox">
      <div className="inbox-tools">
        <label className="search">
          <span className="visually-hidden">Search tickets</span>
          <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
            <circle cx="9" cy="9" r="5.5" />
            <path d="m13.5 13.5 4 4" />
          </svg>
          <input
            id="ticket-search"
            type="search"
            placeholder="Search tickets"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <button type="button" className="button" aria-pressed={flaggedOnly} onClick={() => setFlaggedOnly((v) => !v)}>
          Flagged only
        </button>
      </div>
      {scoreState === 'error' && (
        <p className="inbox-note" role="alert">
          The risk order could not be loaded. <button type="button" className="link" onClick={onRetry}>Try again</button>
        </p>
      )}
      <ul className="tickets" ref={list} onKeyDown={onKeyDown} aria-label="Tickets, riskiest first">
        {shown.map((ticket) => {
          const score = scores[ticket.id]
          const level = !score ? null : !score.alert ? 'low' : score.probability >= 0.6 ? 'high' : 'medium'
          return (
            <li key={ticket.id}>
              <button
                type="button"
                className="ticket"
                data-id={ticket.id}
                aria-current={ticket.id === selected || undefined}
                onClick={() => onSelect(ticket.id)}
              >
                <span className="ticket-subject">{ticket.subject}</span>
                <span className="ticket-risk" data-level={level ?? undefined}>
                  {score ? (
                    <>
                      <span className="visually-hidden">{score.alert ? 'Flagged, risk ' : 'Risk '}</span>
                      {Math.round(score.probability * 100)}
                    </>
                  ) : scoreState === 'loading' ? (
                    <span className="ticket-waiting" aria-label="Scoring" />
                  ) : null}
                </span>
                <span className="ticket-preview">{ticket.text}</span>
              </button>
            </li>
          )
        })}
        {shown.length === 0 && (
          <li className="tickets-empty">
            <p>{needle ? `No ticket mentions "${query.trim()}".` : 'No flagged tickets right now.'}</p>
            <button
              type="button"
              className="button"
              onClick={() => {
                setQuery('')
                setFlaggedOnly(false)
              }}
            >
              Show all tickets
            </button>
          </li>
        )}
      </ul>
    </div>
  )
}
