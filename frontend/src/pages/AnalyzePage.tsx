import { useState, type CSSProperties, type FormEvent } from 'react'
import { ApiError, analyzeTicket, sendFeedback } from '../api/client'
import { CHANNELS, TICKET_TYPES, type AnalyzeResponse, type Rating, type TicketType } from '../api/types'
import { CategoryCard, DraftCard, RiskCard, Timeline, Verdict } from '../components/ResultCards'
import { EMPTY_FORM, EXAMPLES, MAX_CHARS, type FormState } from '../lib/examples'

function ResultsSkeleton() {
  return (
    <div className="results" aria-busy="true" aria-label="Analyzing">
      <div className="card verdict">
        {[0, 1, 2].map((i) => (
          <div className="verdict-cell" key={i}>
            <div className="skeleton" style={{ height: 10, width: 70 }} />
            <div className="skeleton" style={{ height: 30, width: '75%' }} />
          </div>
        ))}
      </div>
      <div className="card">
        <div className="skeleton" style={{ height: 10, width: 90, marginBottom: 18 }} />
        <div className="skeleton" style={{ height: 96 }} />
      </div>
      <div className="results-row">
        <div className="card">
          <div className="skeleton" style={{ height: 150 }} />
        </div>
        <div className="card">
          <div className="skeleton" style={{ height: 150 }} />
        </div>
      </div>
    </div>
  )
}

function EmptyResults() {
  return (
    <div className="card empty">
      <span className="eyebrow">Waiting for a ticket</span>
      <h2>
        One message in, <em>three decisions out.</em>
      </h2>
      <p className="muted" style={{ maxWidth: 480 }}>
        Paste a customer message or pick an example. ResolveIQ answers where it goes, how likely it
        is to escalate and what to reply, and shows the evidence for each.
      </p>
      <div className="empty-steps">
        <div className="empty-step">
          <span className="eyebrow">01</span>
          <strong>Draft reply</strong>
          Grounded in the help center, every fact linked to its source
        </div>
        <div className="empty-step">
          <span className="eyebrow">02</span>
          <strong>Category</strong>
          The queue to route to, flagged when the model is unsure
        </div>
        <div className="empty-step">
          <span className="eyebrow">03</span>
          <strong>Escalation risk</strong>
          A calibrated probability, an alert and the top reasons
        </div>
      </div>
    </div>
  )
}

function Feedback({ requestId }: { requestId: string }) {
  const [rating, setRating] = useState<Rating | null>(null)
  const [error, setError] = useState('')

  const vote = async (value: Rating) => {
    setError('')
    try {
      await sendFeedback(requestId, value)
      setRating(value)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Feedback could not be saved')
    }
  }

  return (
    <div className="feedback">
      <span className="small muted">{rating ? 'Thanks for the feedback' : 'Was this helpful?'}</span>
      <button type="button" className="btn btn-sm" aria-pressed={rating === 'up'} onClick={() => vote('up')}>
        Helpful
      </button>
      <button type="button" className="btn btn-sm" aria-pressed={rating === 'down'} onClick={() => vote('down')}>
        Not helpful
      </button>
      {error && (
        <span className="small" style={{ color: 'var(--high)' }}>
          {error}
        </span>
      )}
    </div>
  )
}

export function AnalyzePage() {
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const length = form.text.length
  const tooLong = length > MAX_CHARS
  const canSubmit = form.text.trim().length > 0 && !tooLong && !loading

  const update = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((current) => ({ ...current, [key]: value }))

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!canSubmit) return
    setLoading(true)
    setError('')
    try {
      const { data } = await analyzeTicket(form)
      setResult(data)
    } catch (err) {
      setResult(null)
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <span className="eyebrow">01 / Triage</span>
          <h1>
            Read the ticket. <em>Decide faster.</em>
          </h1>
          <p>Draft a grounded reply, route it to the right queue and judge its escalation risk in one step.</p>
        </div>
      </header>

      <div className="grid-analyze">
        <form className="card composer" onSubmit={submit} aria-label="Ticket">
          <div className="card-header">
            <h2 className="card-title">Incoming ticket</h2>
            <span className="badge badge-neutral">Synthetic data only</span>
          </div>
          <div className="field">
            <label htmlFor="subject">Subject (optional)</label>
            <input
              id="subject"
              className="input"
              value={form.subject}
              maxLength={300}
              placeholder="Charged twice"
              onChange={(e) => update('subject', e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="text">Customer message</label>
            <textarea
              id="text"
              className="textarea"
              value={form.text}
              placeholder="Paste the customer's message"
              onChange={(e) => update('text', e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void submit(e)
              }}
            />
            <div className="field-hint">
              <span>
                <kbd>Ctrl</kbd> or <kbd>Cmd</kbd> + <kbd>Enter</kbd> to analyze
              </span>
              <span className={`mono${tooLong ? ' over' : ''}`}>
                {length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
              </span>
            </div>
          </div>
          <div className="field-row">
            <div className="field">
              <label htmlFor="ticket_type">Type</label>
              <select
                id="ticket_type"
                className="select"
                value={form.ticket_type}
                onChange={(e) => update('ticket_type', e.target.value as TicketType | '')}
              >
                <option value="">Unknown</option>
                {TICKET_TYPES.map((type) => (
                  <option key={type}>{type}</option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="channel">Channel</label>
              <select id="channel" className="select" value={form.channel} onChange={(e) => update('channel', e.target.value)}>
                <option value="">Unknown</option>
                {CHANNELS.map((channel) => (
                  <option key={channel} value={channel}>
                    {channel.charAt(0).toUpperCase() + channel.slice(1)}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="product">Product</label>
              <input
                id="product"
                className="input"
                value={form.product}
                maxLength={100}
                placeholder="Optional"
                onChange={(e) => update('product', e.target.value)}
              />
            </div>
          </div>
          <button type="submit" className="btn btn-primary btn-block" disabled={!canSubmit}>
            <span className="dot" aria-hidden="true" />
            {loading ? 'Analyzing...' : 'Analyze ticket'}
          </button>
          <div className="examples" aria-label="Examples">
            <span className="eyebrow">Try</span>
            {EXAMPLES.map((example) => (
              <button key={example.label} type="button" className="chip" onClick={() => setForm(example.form)}>
                {example.label}
              </button>
            ))}
          </div>
        </form>

        <div className="results" aria-live="polite">
          {error && (
            <div className="alert-banner error" role="alert">
              <span className="mark" aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}
          {loading ? (
            <ResultsSkeleton />
          ) : result ? (
            <>
              <Verdict result={result} />
              <DraftCard draft={result.draft_reply} index={1} />
              <div className="results-row">
                <CategoryCard category={result.category} index={2} />
                <RiskCard risk={result.risk} index={3} />
              </div>
              <div className="card reveal meta" style={{ '--i': 4 } as CSSProperties}>
                <Timeline timings={result.timings_ms} />
                <div className="meta-row">
                  <span className="small faint">
                    Request <span className="mono">{result.request_id}</span>
                  </span>
                  <Feedback key={result.request_id} requestId={result.request_id} />
                </div>
              </div>
            </>
          ) : (
            !error && <EmptyResults />
          )}
        </div>
      </div>
    </>
  )
}
