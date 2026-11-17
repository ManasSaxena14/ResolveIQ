import { useState } from 'react'
import type { Category, DraftReply, Risk } from '../api/types'
import { percent, riskLevel, sentence, sourceSection, sourceTitle, sourceUrl } from '../lib/format'
import { RiskGauge } from './RiskGauge'

const DRAFT_BADGE = {
  ok: { label: 'Grounded draft', className: 'badge-ok' },
  refused: { label: 'Out of scope', className: 'badge-neutral' },
  unavailable: { label: 'Draft unavailable', className: 'badge-warn' },
} as const

export function DraftCard({ draft }: { draft: DraftReply }) {
  const [copied, setCopied] = useState(false)
  const badge = DRAFT_BADGE[draft.status]

  const copy = async () => {
    if (!draft.text) return
    try {
      await navigator.clipboard.writeText(draft.text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }

  return (
    <section className="card" aria-labelledby="draft-title">
      <div className="card-header">
        <h2 id="draft-title" className="card-title">
          Draft reply
        </h2>
        <span className={`badge badge-dot ${badge.className}`}>{badge.label}</span>
      </div>

      {draft.status === 'ok' && (
        <>
          <p className="draft-text">{draft.text}</p>
          <button type="button" className="btn btn-sm" onClick={copy}>
            {copied ? 'Copied' : 'Copy draft'}
          </button>
        </>
      )}
      {draft.status === 'refused' && (
        <div className="alert-banner info">
          The knowledge base has nothing on this question, so no reply was drafted. Answer it
          manually or route it to the right team.
        </div>
      )}
      {draft.status === 'unavailable' && (
        <div className="alert-banner warn">
          {sentence(draft.error ?? 'The language model is unavailable')} The category and risk
          below are still valid.
        </div>
      )}

      {draft.sources.length > 0 && (
        <div style={{ marginTop: draft.status === 'ok' ? 18 : 0 }}>
          <p className="section-label">Sources from the help center</p>
          <ul className="sources">
            {draft.sources.map((source) => (
              <li key={`${source.file}-${source.page}`} className="source">
                <div>
                  <a className="source-name" href={sourceUrl(source.file)} target="_blank" rel="noreferrer">
                    {sourceTitle(source.file)}
                  </a>
                  <div className="source-meta">
                    {sourceSection(source.file)} &middot; page {source.page}
                  </div>
                </div>
                {source.score !== null && (
                  <div className="source-score" title="Cosine similarity to the ticket">
                    <span>{source.score.toFixed(2)} similarity</span>
                    <div className="bar">
                      <span style={{ width: percent(source.score) }} />
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

export function CategoryCard({ category }: { category: Category }) {
  return (
    <section className="card" aria-labelledby="category-title">
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
        {percent(category.confidence)} confidence
        {category.needs_review && ' - below the review threshold, a person should confirm'}
      </div>
      <div className="prob-list">
        {category.top.map((item) => (
          <div className="prob-row" key={item.label}>
            <div className="prob-row-head">
              <span>{item.label}</span>
              <span className="muted">{percent(item.probability)}</span>
            </div>
            <div className="bar">
              <span style={{ width: percent(item.probability) }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

const RISK_BADGE = {
  low: { label: 'Low risk', className: 'badge-low' },
  medium: { label: 'Elevated risk', className: 'badge-medium' },
  high: { label: 'High risk', className: 'badge-high' },
} as const

export function RiskCard({ risk }: { risk: Risk }) {
  const badge = RISK_BADGE[riskLevel(risk.escalation_probability, risk.threshold)]
  return (
    <section className="card" aria-labelledby="risk-title">
      <div className="card-header">
        <h2 id="risk-title" className="card-title">
          Escalation risk
        </h2>
        <span className={`badge badge-dot ${badge.className}`}>{badge.label}</span>
      </div>
      <RiskGauge probability={risk.escalation_probability} threshold={risk.threshold} />
      <p className="small" style={{ textAlign: 'center', margin: '6px 0 0' }}>
        {risk.alert ? (
          <strong style={{ color: 'var(--high)' }}>Alert: prioritize this ticket</strong>
        ) : (
          <span className="muted">No alert</span>
        )}
      </p>
      {risk.reasons.length > 0 ? (
        <ol className="reasons" aria-label="Top reasons">
          {risk.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ol>
      ) : (
        <p className="muted small">No single feature raises the risk noticeably.</p>
      )}
    </section>
  )
}
