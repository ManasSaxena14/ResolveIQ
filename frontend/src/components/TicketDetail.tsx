import { useState } from 'react'
import { sendFeedback } from '../api/client'
import type { AnalyzeResponse, Rating } from '../api/types'
import { milliseconds, percent } from '../lib/format'
import { ticketMeta, type Ticket } from '../lib/inbox'
import type { TicketResult } from '../lib/useTriage'
import { Draft } from './Draft'
import { RiskScale } from './RiskScale'
import { SortingRack } from './SortingRack'
import { TicketText } from './TicketText'

function Vote({ requestId }: { requestId: string }) {
  const [rating, setRating] = useState<Rating | null>(null)
  const [error, setError] = useState('')
  const vote = async (value: Rating) => {
    setError('')
    try {
      await sendFeedback(requestId, value)
      setRating(value)
    } catch {
      setError('Your answer was not saved. Try again.')
    }
  }
  return (
    <div className="vote">
      <span>{rating ? 'Answer saved.' : 'Was this ticket sorted right?'}</span>
      <button type="button" className="button" aria-pressed={rating === 'up'} onClick={() => vote('up')}>
        Yes, right
      </button>
      <button type="button" className="button" aria-pressed={rating === 'down'} onClick={() => vote('down')}>
        No, wrong
      </button>
      {error && (
        <span className="field-error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}

const STEPS: Record<string, string> = {
  retrieval: 'Finding articles',
  generation: 'Writing the reply',
  classifier: 'Choosing the queue',
  risk: 'Scoring the risk',
  explain: 'Marking words',
}

function Timing({ final }: { final: AnalyzeResponse }) {
  const known = Object.keys(STEPS).filter((k) => final.timings_ms[k] !== undefined)
  const total = final.timings_ms.total ?? 0
  const other = total - known.reduce((sum, k) => sum + final.timings_ms[k], 0)
  return (
    <details className="timing">
      <summary>Took {milliseconds(total)}</summary>
      <dl>
        {known.map((k) => (
          <div key={k}>
            <dt>{STEPS[k]}</dt>
            <dd>{milliseconds(final.timings_ms[k])}</dd>
          </div>
        ))}
        {other > Math.max(5, total * 0.03) && (
          <div>
            <dt>Waiting and other work</dt>
            <dd>{milliseconds(other)}</dd>
          </div>
        )}
        <div>
          <dt>Request</dt>
          <dd>{final.request_id}</dd>
        </div>
      </dl>
    </details>
  )
}

interface Props {
  ticket: Ticket
  result: TicketResult | undefined
  onRetry: () => void
}

export function TicketDetail({ ticket, result, onRetry }: Props) {
  const analysis = result?.analysis ?? null
  const category = analysis?.category ?? null
  const failed = result?.status === 'error'

  return (
    <article className="detail" aria-labelledby="detail-title" aria-busy={result?.status === 'running'}>
      <header className="detail-head">
        <h2 id="detail-title" tabIndex={-1}>
          {ticket.subject}
        </h2>
        <p className="secondary muted">{ticketMeta(ticket)}</p>
      </header>

      {failed && (
        <div className="notice" role="alert">
          <p>{result.error}</p>
          <button type="button" className="button" onClick={onRetry}>
            Triage again
          </button>
        </div>
      )}

      <TicketText key={ticket.id} text={ticket.text} highlights={analysis?.highlights ?? null} />

      <div className="decision">
        <section className="decision-queue" aria-labelledby="queue-title">
          <h3 id="queue-title" className="decision-title">
            {category ? category.label : failed ? 'Not sorted yet' : 'Sorting'}
          </h3>
          <p className="secondary" aria-live="polite">
            {category
              ? category.needs_review
                ? `${percent(category.confidence)} sure. Check the queue before sending it on.`
                : `${percent(category.confidence)} sure, enough to send it on.`
              : failed
                ? 'Triage this ticket again to choose a queue.'
                : 'Choosing one of ten queues.'}
          </p>
          <SortingRack category={category} subject={ticket.subject} />
        </section>
        <section className="decision-risk" aria-labelledby="risk-title">
          <h3 id="risk-title">Escalation risk</h3>
          <RiskScale risk={analysis?.risk ?? null} waitingText={failed ? 'Not scored yet' : 'Reading the ticket'} />
        </section>
      </div>

      {analysis && (
        <Draft
          key={`draft-${ticket.id}`}
          draft={result?.final?.draft_reply ?? null}
          liveText={result?.text ?? ''}
          liveSources={result?.sources ?? []}
        />
      )}

      {result?.final && (
        <footer className="detail-foot">
          <Vote key={result.final.request_id} requestId={result.final.request_id} />
          <Timing final={result.final} />
        </footer>
      )}
    </article>
  )
}
