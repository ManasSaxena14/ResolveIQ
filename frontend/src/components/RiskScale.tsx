import type { Risk } from '../api/types'
import { riskLevel } from '../lib/format'

const WORDS = { low: 'Low', medium: 'Elevated', high: 'High' } as const

/** Probability of escalation on a 0 to 100 scale with the flag threshold marked. */
export function RiskScale({ risk, waitingText = 'Reading the ticket' }: { risk: Risk | null; waitingText?: string }) {
  if (!risk) {
    return (
      <div className="risk">
        <p className="risk-value" aria-hidden="true">
          &nbsp;
        </p>
        <div className="scale" aria-hidden="true" />
        <p className="secondary muted">{waitingText}</p>
      </div>
    )
  }
  const value = Math.round(risk.escalation_probability * 100)
  const threshold = Math.round(risk.threshold * 100)
  const level = riskLevel(risk.escalation_probability, risk.threshold)
  return (
    <div className="risk" data-level={level}>
      <p className="risk-value">
        {value}%<span className="risk-word"> {WORDS[level].toLowerCase()} risk</span>
      </p>
      <div
        className="scale"
        role="img"
        aria-label={`Escalation risk ${value} percent, flagged from ${threshold} percent`}
      >
        <span className="scale-fill" style={{ transform: `scaleX(${risk.escalation_probability})` }} />
        <span className="scale-mark" style={{ left: `${threshold}%` }} />
      </div>
      <p className="secondary">
        {risk.alert ? 'Flagged for a quick answer.' : `Not flagged. Tickets are flagged from ${threshold}%.`}
      </p>
      {risk.reasons.length > 0 && (
        <ul className="reasons" aria-label="Reasons">
          {risk.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
      {risk.reasons.some((reason) => /\bqueue \(\d+%\)/.test(reason)) && (
        <p className="meta">The risk model makes its own rough guess at the queue, so its numbers differ from the rack.</p>
      )}
    </div>
  )
}
