import { useState, type CSSProperties, type FormEvent } from 'react'
import { ApiError, analyzeTicketStream, sendFeedback } from '../api/client'
import {
  CHANNELS,
  TICKET_TYPES,
  type AnalysisEvent,
  type AnalyzeResponse,
  type DraftReply,
  type Rating,
  type Source,
  type TicketType,
} from '../api/types'
import { CategoryCard, DraftCard, RiskCard, Timeline, Verdict } from '../components/ResultCards'
import { WhyCard } from '../components/WhyCard'
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

interface Live {
  submitted: { subject: string; text: string }
  analysis: AnalysisEvent | null
  sources: Source[]
  text: string
  final: AnalyzeResponse | null
}

function Results({ live, analysis }: { live: Live; analysis: AnalysisEvent }) {
  const final = live.final
  const streamingDraft: DraftReply = {
    text: live.text,
    status: 'ok',
    refused: false,
    sources: live.sources,
    error: null,
  }
  return (
    <>
      <Verdict
        result={{ category: analysis.category, risk: analysis.risk, draft_reply: final?.draft_reply ?? null }}
      />
      <DraftCard draft={final?.draft_reply ?? streamingDraft} index={1} streaming={!final} />
      <div className="results-row">
        <CategoryCard category={analysis.category} index={2} />
        <RiskCard risk={analysis.risk} index={3} />
      </div>
      <WhyCard
        subject={live.submitted.subject}
        text={live.submitted.text}
        highlights={analysis.highlights}
        index={4}
      />
      {final && (
        <div className="card reveal meta" style={{ '--i': 5 } as CSSProperties}>
          <Timeline timings={final.timings_ms} />
          <div className="meta-row">
            <span className="small faint">
              Request <span className="mono">{final.request_id}</span>
            </span>
            <Feedback key={final.request_id} requestId={final.request_id} />
          </div>
        </div>
      )}
    </>
  )
}

export function AnalyzePage() {
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [live, setLive] = useState<Live | null>(null)
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
    setLive({
      submitted: { subject: form.subject, text: form.text },
      analysis: null,
      sources: [],
      text: '',
      final: null,
    })
    try {
      const { data } = await analyzeTicketStream(form, {
        onAnalysis: (analysis) => setLive((current) => current && { ...current, analysis }),
        onSources: (sources) => setLive((current) => current && { ...current, sources }),
        onToken: (piece) => setLive((current) => current && { ...current, text: current.text + piece }),
      })
      setLive((current) => current && { ...current, final: data, analysis: current.analysis ?? {
        request_id: data.request_id,
        category: data.category,
        risk: data.risk,
        highlights: data.highlights ?? { category: [], risk: [] },
        versions: data.versions,
      } })
    } catch (err) {
      setLive(null)
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
          {live && !live.analysis ? (
            <ResultsSkeleton />
          ) : live?.analysis ? (
            <Results live={live} analysis={live.analysis} />          ) : (
            !error && !loading && <EmptyResults />
          )}
        </div>
      </div>
    </>
  )
}
