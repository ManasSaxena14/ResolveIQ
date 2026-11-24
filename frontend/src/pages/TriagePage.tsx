import { useEffect, useState } from 'react'
import { Composer } from '../components/Composer'
import { SortingRack } from '../components/SortingRack'
import { TicketDetail } from '../components/TicketDetail'
import { TicketList } from '../components/TicketList'
import { EMPTY_DRAFT, type Ticket } from '../lib/inbox'
import { href, navigate } from '../lib/router'
import { NARROW, useMedia } from '../lib/useMedia'
import type { Triage } from '../lib/useTriage'

/** Pause before triaging, so moving through the list with the arrow keys stays cheap. */
const SETTLE_MS = 300

function focusDetail() {
  window.requestAnimationFrame(() => document.getElementById('detail-title')?.focus())
}

function typing(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && !!target.closest('input, textarea, select')
}

export function TriagePage({ triage, item }: { triage: Triage; item: string | null }) {
  const { tickets, scores, scoreState, results, analyze, addTicket, retryScores } = triage
  const narrow = useMedia(NARROW)
  const [draft, setDraft] = useState(EMPTY_DRAFT)
  const composing = item === 'new'
  const ticket = tickets.find((t) => t.id === item)
  const first = tickets[0]?.id
  const status = ticket && results[ticket.id]?.status

  // On wide screens, open the riskiest ticket once the order is known.
  useEffect(() => {
    if (!narrow && !composing && !ticket && scoreState !== 'loading' && first) navigate('triage', first, true)
  }, [narrow, composing, ticket, scoreState, first])

  useEffect(() => {
    if (!ticket || status) return
    const timer = window.setTimeout(() => void analyze(ticket), SETTLE_MS)
    return () => window.clearTimeout(timer)
  }, [ticket, status, analyze])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || typing(event.target)) return
      if (event.key === '/') {
        event.preventDefault()
        document.getElementById('ticket-search')?.focus()
      } else if (event.key === 'n') {
        event.preventDefault()
        navigate('triage', 'new')
        focusDetail()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const create = (created: Ticket) => {
    addTicket(created)
    setDraft(EMPTY_DRAFT)
    navigate('triage', created.id)
    focusDetail()
  }

  const showList = !narrow || (!ticket && !composing)
  const showDetail = !narrow || !!ticket || composing

  return (
    <div className="split">
      {showList && (
        <section className="split-list" aria-label="Inbox">
          <div className="page-head">
            <div>
              <h1>Triage</h1>
              <p className="secondary muted">Sample tickets, riskiest first.</p>
            </div>
            <a className="button primary" href={href('triage', 'new')}>
              New ticket
            </a>
          </div>
          <TicketList
            tickets={tickets}
            scores={scores}
            scoreState={scoreState}
            selected={ticket?.id ?? null}
            onSelect={(id) => {
              navigate('triage', id, !narrow)
              if (narrow) window.scrollTo({ top: 0 })
            }}
            onEnter={focusDetail}
            onRetry={retryScores}
          />
          <p className="keys meta muted">
            <span>
              <kbd>/</kbd> search
            </span>
            <span>
              <kbd>n</kbd> new ticket
            </span>
            <span>
              <kbd>&uarr;</kbd> <kbd>&darr;</kbd> move
            </span>
            <span>
              <kbd>Enter</kbd> open
            </span>
          </p>
        </section>
      )}

      {showDetail && (
        <div className="split-detail">
          {narrow && (
            <a className="back" href={href('triage')}>
              All tickets
            </a>
          )}
          {composing ? (
            <Composer draft={draft} onChange={setDraft} onCreate={create} />
          ) : ticket ? (
            <TicketDetail
              key={ticket.id}
              ticket={ticket}
              result={results[ticket.id]}
              onRetry={() => void analyze(ticket, true)}
            />
          ) : (
            <article className="detail" aria-busy="true">
              <h2 id="detail-title">Scoring the inbox</h2>
              <p className="muted">The riskiest ticket opens as soon as the order is known.</p>
              <SortingRack category={null} subject="" />
            </article>
          )}
        </div>
      )}
    </div>
  )
}
