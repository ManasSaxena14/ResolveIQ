import { riskLevel } from '../lib/format'

const RADIUS = 80
const ARC = Math.PI * RADIUS

interface Props {
  probability: number
  threshold: number
}

/** Half-circle gauge with the alert threshold marked as a tick. */
export function RiskGauge({ probability, threshold }: Props) {
  const value = Math.min(Math.max(probability, 0), 1)
  const level = riskLevel(value, threshold)
  const angle = Math.PI * (1 - threshold)
  const tick = {
    x1: 100 + 66 * Math.cos(angle),
    y1: 100 - 66 * Math.sin(angle),
    x2: 100 + 94 * Math.cos(angle),
    y2: 100 - 94 * Math.sin(angle),
  }
  const color = `var(--${level})`

  return (
    <svg
      className="gauge"
      viewBox="0 0 200 122"
      role="img"
      aria-label={`Escalation probability ${Math.round(value * 100)} percent, alert threshold ${Math.round(threshold * 100)} percent`}
    >
      <path
        d="M20 100 A80 80 0 0 1 180 100"
        fill="none"
        stroke="var(--track)"
        strokeWidth="16"
        strokeLinecap="round"
      />
      <path
        data-testid="gauge-fill"
        d="M20 100 A80 80 0 0 1 180 100"
        fill="none"
        stroke={color}
        strokeWidth="16"
        strokeLinecap="round"
        strokeDasharray={`${(value * ARC).toFixed(1)} ${ARC.toFixed(1)}`}
      />
      <line {...tick} stroke="var(--text-muted)" strokeWidth="2" />
      <text x="100" y="92" textAnchor="middle" className="gauge-value">
        {Math.round(value * 100)}%
      </text>
      <text x="100" y="118" textAnchor="middle" className="gauge-caption">
        alert at {Math.round(threshold * 100)}%
      </text>
    </svg>
  )
}
