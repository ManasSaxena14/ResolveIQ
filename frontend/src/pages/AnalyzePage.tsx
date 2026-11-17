import { useState, type FormEvent } from 'react'
import { ApiError, analyzeTicket, sendFeedback } from '../api/client'
import { CHANNELS, TICKET_TYPES, type AnalyzeResponse, type Rating, type TicketType } from '../api/types'
import { CategoryCard, DraftCard, RiskCard } from '../components/ResultCards'
import { EMPTY_FORM, EXAMPLES, MAX_CHARS, type FormState } from '../lib/examples'
import { milliseconds } from '../lib/format'

const TIMING_LABELS: Record<string, string> = {
  retrieval: 'Retrieval',
  generation: 'Generation',
  classifier: 'Classifier',
  risk: 'Risk',
  total: 'Total',
}

function ResultsSkeleton() {
  return (
    <div className="results" aria-busy="true" aria-label="Analyzing">
      <div className="card">
        <div className="skeleton" style={{ height: 14, width: 120, marginBottom: 16 }} />
        <div className="skeleton" style={{ height: 70 }} />
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
      <h2>Paste a ticket to get started</h2>
      <p>ResolveIQ returns three things for every ticket, each with its evidence.</p>
      <div className="empty-steps">
        <div className="empty-step">
          <strong>Draft reply</strong>
          Grounded in the help center, with linked sources
        </div>
        <div className="empty-step">
          <strong>Category</strong>
          The queue to route to, flagged when unsure
        </div>
        <div className="empty-step">
          <strong>Escalation risk</strong>
          Probability, alert and the top reasons
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
      <button
        type="button"
        className="btn btn-sm"
        aria-pressed={rating === 'up'}
        onClick={() => vote('up')}
      >
        Helpful
      </button>
      <button
        type="button"
        className="btn btn-sm"
        aria-pressed={rating === 'down'}
        onClick={() => vote('down')}
      >
        Not helpful
      </button>
      {error && <span className="small" style={{ color: 'var(--high)' }}>{error}</span>}
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
      <div className="page-header">
        <div>
          <h1>Analyze a ticket</h1>
          <p>Draft a reply, route it and judge its escalation risk in one step.</p>
        </div>
      </div>

      <div className="grid-analyze">
        <form className="card" onSubmit={submit} aria-label="Ticket">
          <div className="field">
            <label htmlFor="subject">Subject (optional)</label>
            <input
              id="subject"
              className="input"
              value={form.subject}
              maxLength={300}
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
              <span>Press Ctrl or Cmd + Enter to analyze</span>
              <span className={tooLong ? 'over' : undefined}>
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
              <select
                id="channel"
                className="select"
                value={form.channel}
                onChange={(e) => update('channel', e.target.value)}
              >
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
                onChange={(e) => update('product', e.target.value)}
              />
            </div>
          </div>
          <button type="submit" className="btn btn-primary btn-block" disabled={!canSubmit}>
            {loading ? 'Analyzing...' : 'Analyze ticket'}
          </button>
          <div className="examples" aria-label="Examples">
            <span className="small faint">Try:</span>
            {EXAMPLES.map((example) => (
              <button
                key={example.label}
                type="button"
                className="chip"
                onClick={() => setForm(example.form)}
              >
                {example.label}
              </button>
            ))}
          </div>
        </form>

        <div>
          {error && (
            <div className="alert-banner error" role="alert">
              {error}
            </div>
          )}
          {loading ? (
            <ResultsSkeleton />
          ) : result ? (
            <div className="results">
              <DraftCard draft={result.draft_reply} />
              <div className="results-row">
                <CategoryCard category={result.category} />
                <RiskCard risk={result.risk} />
              </div>
              <div className="card meta">
                <div>
                  <div className="timings">
                    {Object.entries(result.timings_ms).map(([key, value]) => (
                      <span key={key}>
                        {TIMING_LABELS[key] ?? key} <strong>{milliseconds(value)}</strong>
                      </span>
                    ))}
                  </div>
                  <div className="small faint" style={{ marginTop: 4 }}>
                    Request <span className="mono">{result.request_id}</span>
                  </div>
                </div>
                <Feedback key={result.request_id} requestId={result.request_id} />
              </div>
            </div>
          ) : (
            !error && <EmptyResults />
          )}
        </div>
      </div>
    </>
  )
}
