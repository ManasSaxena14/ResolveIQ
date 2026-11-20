import { useState, type CSSProperties } from 'react'
import type { Category, DraftReply, Grounding, Risk, Source } from '../api/types'
import { draftParts } from '../lib/highlight'
import { milliseconds, percent, riskLevel, sentence, sourceSection, sourceTitle, sourceUrl } from '../lib/format'
import { useCountUp } from '../lib/useCountUp'
import { RiskMeter } from './RiskMeter'

const reveal = (index: number) => ({ '--i': index }) as CSSProperties

const DRAFT_BADGE = {
  ok: { label: 'Grounded draft', className: 'badge-ok' },
  refused: { label: 'Out of scope', className: 'badge-neutral' },
  unavailable: { label: 'Draft unavailable', className: 'badge-warn' },
} as const

const RISK_BADGE = {
  low: { label: 'Low risk', className: 'badge-low' },
  medium: { label: 'Elevated risk', className: 'badge-medium' },
  high: { label: 'High risk', className: 'badge-high' },
} as const

/** Route, risk and draft at a glance; the evidence follows in the cards below. */
export function Verdict({
  result,
}: {
  result: { category: Category; risk: Risk; draft_reply: DraftReply | null }
}) {
  const { category, risk, draft_reply: draft } = result
  const level = riskLevel(risk.escalation_probability, risk.threshold)
  const shown = useCountUp(risk.escalation_probability * 100)
  const draftLabel = draft
    ? { ok: 'Ready to edit', refused: 'Not drafted', unavailable: 'Sources only' }[draft.status]
    : 'Writing...'

  return (
    <section className="card verdict reveal" style={reveal(0)} aria-label="Verdict">
      <div className="verdict-cell">
        <span className="eyebrow">Route to</span>
        <span className="verdict-value">{category.label}</span>
        <span className="small muted">
          <span className="mono">{percent(category.confidence)}</span> confidence
          {category.needs_review && <span style={{ color: 'var(--medium)' }}>, needs review</span>}
        </span>
      </div>
      <div className="verdict-cell">
        <span className="eyebrow">Escalation risk</span>
        <span className={`verdict-number ${level}`} aria-hidden="true">
          {Math.round(shown)}
          <span style={{ fontSize: '0.5em', marginLeft: 2 }}>%</span>
        </span>
        <span className="small muted">{risk.alert ? 'Above the alert threshold' : 'Below the alert threshold'}</span>
      </div>
      <div className="verdict-cell">
        <span className="eyebrow">Reply</span>
        <span className="verdict-value">{draftLabel}</span>
        <span className="small muted">
          {draft
            ? `${draft.sources.length} source${draft.sources.length === 1 ? '' : 's'} from the help center`
            : 'The draft streams in below'}
        </span>
      </div>
    </section>
  )
}

/** The draft with each citation turned into a numbered chip that points at its source. */
function DraftText({ text, sources, streaming }: { text: string; sources: Source[]; streaming: boolean }) {
  return (
    <p className="draft-text" aria-live={streaming ? 'polite' : undefined}>
      {draftParts(text, sources).map((part, i) =>
        'citation' in part ? (
          part.index !== null ? (
            <a key={i} className="cite" href={`#source-${part.index}`} title={part.citation}>
              {part.index}
            </a>
          ) : (
            <span key={i} className="cite cite-invalid" title={`Not a retrieved source: ${part.citation}`}>
              ?
            </span>
          )
        ) : (
          <span key={i}>{part.text}</span>
        ),
      )}
      {streaming && <span className="caret" aria-hidden="true" />}
    </p>
  )
}

function GroundingLine({ grounding }: { grounding: Grounding }) {
  const ok = grounding.invalid_citations.length === 0 && grounding.coverage >= 0.5
  return (
    <p className={`grounding small ${ok ? 'ok' : 'warn'}`}>
      <span className="mono">
        {grounding.cited_sentences}/{grounding.sentences}
      </span>{' '}
      sentences cite a source
      {grounding.invalid_citations.length > 0
        ? `. Check: ${grounding.invalid_citations.join('; ')} was not retrieved.`
        : grounding.citations > 0
          ? ', every citation points at a retrieved article.'
          : '. No citations: verify every fact before sending.'}
    </p>
  )
}

export function DraftCard({
  draft,
  index = 1,
  streaming = false,
}: {
  draft: DraftReply
  index?: number
  streaming?: boolean
}) {
  const [copied, setCopied] = useState(false)
  const badge = streaming ? { label: 'Writing', className: 'badge-info' } : DRAFT_BADGE[draft.status]

  const copy = async () => {
    if (!draft.text) return
    try {
      await navigator.clipboard.writeText(draft.text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      setCopied(false)
    }
  }

  return (
    <section className="card reveal" style={reveal(index)} aria-labelledby="draft-title">
      <div className="card-header">
        <h2 id="draft-title" className="card-title">
          Draft reply
        </h2>
        <span className={`badge badge-dot ${badge.className}`}>{badge.label}</span>
      </div>

      {streaming && <DraftText text={draft.text ?? ''} sources={draft.sources} streaming />}
      {!streaming && draft.status === 'ok' && (
        <>
          <DraftText text={draft.text ?? ''} sources={draft.sources} streaming={false} />
          {draft.grounding && <GroundingLine grounding={draft.grounding} />}
          <button type="button" className="btn btn-sm" onClick={copy}>
            {copied ? 'Copied' : 'Copy draft'}
          </button>
        </>
      )}
      {!streaming && draft.status === 'refused' && (
        <div className="alert-banner info">
          <span className="mark" aria-hidden="true" />
          <span>
            The help center has nothing on this question, so no reply was drafted. Answer it
            yourself or route it to the right team.
          </span>
        </div>
      )}
      {!streaming && draft.status === 'unavailable' && (
        <div className="alert-banner warn">
          <span className="mark" aria-hidden="true" />
          <span>
            {sentence(draft.error ?? 'The language model is unavailable')} The category and risk
            below are still valid.
          </span>
        </div>
      )}

      {draft.sources.length > 0 && (
        <>
          <p className="eyebrow section-label">Evidence from the help center</p>
          <ol className="sources">
            {draft.sources.map((source, i) => (
              <li key={`${source.file}-${source.page}`} className="source" id={`source-${i + 1}`}>
                <span className="source-index">{String(i + 1).padStart(2, '0')}</span>
                <div>
                  <a className="source-name" href={sourceUrl(source.file)} target="_blank" rel="noreferrer">
                    {sourceTitle(source.file)}
                  </a>
                  <div className="source-meta">
                    {sourceSection(source.file)} / page {source.page}
                  </div>
                </div>
                {source.score !== null && (
                  <div className="source-score" title="Cosine similarity to the ticket">
                    <span>{source.score.toFixed(2)} match</span>
                    <div className="bar signal">
                      <span style={{ width: percent(source.score), animationDelay: `${120 + i * 60}ms` }} />
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  )
}

export function CategoryCard({ category, index = 2 }: { category: Category; index?: number }) {
  return (
    <section className="card reveal" style={reveal(index)} aria-labelledby="category-title">
      <div className="card-header">
        <h2 id="category-title" className="card-title">
          Category
        </h2>
        {category.needs_review ? (
          <span className="badge badge-dot badge-warn">Needs review</span>
        ) : (
          <span className="badge badge-dot badge-ok">Auto-route</span>
        )}
      </div>
      <div className="category-label">{category.label}</div>
      <div className="muted small">
        <span className="mono">{percent(category.confidence)}</span> confidence
        {category.needs_review && ', below the review threshold. A person should confirm.'}
      </div>
      <div className="prob-list">
        {category.top.map((item, i) => (
          <div className="prob-row" key={item.label}>
            <div className="prob-row-head">
              <span>{item.label}</span>
              <span className="mono">{percent(item.probability)}</span>
            </div>
            <div className="bar">
              <span
                style={{
                  width: percent(item.probability),
                  opacity: i === 0 ? 1 : 0.35,
                  animationDelay: `${180 + i * 70}ms`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export function RiskCard({ risk, index = 3 }: { risk: Risk; index?: number }) {
  const level = riskLevel(risk.escalation_probability, risk.threshold)
  const badge = RISK_BADGE[level]
  return (
    <section className="card reveal" style={reveal(index)} aria-labelledby="risk-title">
      <div className="card-header">
        <h2 id="risk-title" className="card-title">
          Escalation risk
        </h2>
        <span className={`badge badge-dot ${badge.className}`}>{badge.label}</span>
      </div>
      <RiskMeter probability={risk.escalation_probability} threshold={risk.threshold} />
      <div className="alert-line">
        {risk.alert ? (
          <strong style={{ color: 'var(--high)', fontWeight: 600 }}>Alert: prioritize this ticket</strong>
        ) : (
          <span className="muted">No alert</span>
        )}
        <span className="mono faint small">p = {risk.escalation_probability.toFixed(2)}</span>
      </div>
      {risk.reasons.length > 0 ? (
        <ol className="reasons" aria-label="Top reasons">
          {risk.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ol>
      ) : (
        <p className="muted small" style={{ marginTop: 16 }}>
          No single feature raises the risk noticeably.
        </p>
      )}
    </section>
  )
}

const STEPS = [
  { key: 'retrieval', label: 'Retrieval', color: 'var(--info)' },
  { key: 'generation', label: 'Generation', color: 'var(--signal-fill)' },
  { key: 'classifier', label: 'Classifier', color: 'var(--ink-2)' },
  { key: 'risk', label: 'Risk', color: 'var(--medium)' },
  { key: 'explain', label: 'Explain', color: 'var(--low)' },
] as const

/** Where the time went, as one proportional bar. */
export function Timeline({ timings }: { timings: Record<string, number> }) {
  const known = STEPS.filter((step) => timings[step.key] !== undefined).map((step) => ({
    ...step,
    ms: timings[step.key],
  }))
  const measured = known.reduce((sum, step) => sum + step.ms, 0)
  const total = timings.total ?? measured
  // Time outside the named steps, for example a language model call that failed.
  const other = total - measured
  const steps =
    other > Math.max(5, total * 0.03)
      ? [...known, { key: 'other', label: 'Other', color: 'var(--line-strong)', ms: other }]
      : known
  const scale = steps.reduce((sum, step) => sum + step.ms, 0) || 1
  return (
    <div className="meta">
      <div className="meta-row">
        <span className="card-title">Response time</span>
        <span className="mono">{milliseconds(total)}</span>
      </div>
      <div className="timeline" aria-hidden="true">
        {steps.map((step) => (
          <span
            key={step.key}
            style={{ width: `${(step.ms / scale) * 100}%`, background: step.color }}
          />
        ))}
      </div>
      <div className="timeline-legend">
        {steps.map((step) => (
          <span key={step.key}>
            <i style={{ background: step.color }} />
            {step.label} <span className="mono">{milliseconds(step.ms)}</span>
          </span>
        ))}
      </div>
    </div>
  )
}
