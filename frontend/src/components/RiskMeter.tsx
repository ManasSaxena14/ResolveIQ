import { riskLevel } from '../lib/format'

interface Props {
  probability: number
  threshold: number
}

const WIDTH = 300
const TICKS = Array.from({ length: 11 }, (_, i) => i / 10)

/** A ruled scale from 0 to 100% with the alert threshold marked, like a measuring instrument. */
export function RiskMeter({ probability, threshold }: Props) {
  const value = Math.min(Math.max(probability, 0), 1)
  const level = riskLevel(value, threshold)
  const x = (p: number) => 4 + p * (WIDTH - 8)

  return (
    <svg
      className="meter"
      viewBox={`0 0 ${WIDTH} 62`}
      role="img"
      aria-label={`Escalation probability ${Math.round(value * 100)} percent, alert threshold ${Math.round(threshold * 100)} percent`}
    >
      <rect x="4" y="18" width={WIDTH - 8} height="10" rx="5" fill="var(--track)" />
      <rect
        className="meter-fill"
        data-testid="meter-fill"
        x="4"
        y="18"
        width={Math.max(value * (WIDTH - 8), 10)}
        height="10"
        rx="5"
        fill={`var(--${level})`}
      />
      {TICKS.map((tick) => (
        <g key={tick}>
          <line
            x1={x(tick)}
            x2={x(tick)}
            y1="34"
            y2={tick % 0.5 === 0 ? 41 : 38}
            stroke="var(--line-strong)"
            strokeWidth="1"
          />
          {tick % 0.5 === 0 && (
            <text x={x(tick)} y="54" textAnchor={tick === 0 ? 'start' : tick === 1 ? 'end' : 'middle'}>
              {Math.round(tick * 100)}
            </text>
          )}
        </g>
      ))}
      <line x1={x(threshold)} x2={x(threshold)} y1="10" y2="32" stroke="var(--ink)" strokeWidth="1.5" />
      <text x={x(threshold)} y="7" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        alert {Math.round(threshold * 100)}
      </text>
    </svg>
  )
}
