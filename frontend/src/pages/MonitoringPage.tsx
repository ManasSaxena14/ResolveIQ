import { useEffect, useState, type CSSProperties } from 'react'
import { getRecentRequests, getStats } from '../api/client'
import type { RequestSummary, StatsResponse } from '../api/types'
import { milliseconds, percent, relativeTime } from '../lib/format'

const DRAFT_LABEL: Record<string, { label: string; className: string; color: string }> = {
  ok: { label: 'Drafted', className: 'badge-ok', color: 'var(--low)' },
  refused: { label: 'Out of scope', className: 'badge-neutral', color: 'var(--ink-3)' },
  unavailable: { label: 'Unavailable', className: 'badge-warn', color: 'var(--medium)' },
}

const reveal = (index: number) => ({ '--i': index }) as CSSProperties

function Tile({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="tile">
      <div className="eyebrow">{label}</div>
      <div className="tile-value" style={tone ? { color: tone } : undefined}>
        {value}
      </div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  )
}

function Categories({ counts, total }: { counts: Record<string, number>; total: number }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1])
  const max = Math.max(...entries.map(([, n]) => n), 1)
  return (
    <div className="card reveal" style={reveal(1)}>
      <div className="card-header">
        <h2 className="card-title">Predicted categories</h2>
        <span className="mono faint small">{total} tickets</span>
      </div>
      {entries.length === 0 ? (
        <p className="muted small">No requests yet.</p>
      ) : (
        <div className="hbar-list">
          {entries.map(([label, count], i) => (
            <div className="hbar" key={label}>
              <span>{label}</span>
              <div className="bar">
                <span style={{ width: `${(count / max) * 100}%`, animationDelay: `${i * 40}ms` }} />
              </div>
              <span className="count">{count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function DraftOutcome({ counts, total }: { counts: Record<string, number>; total: number }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1])
  return (
    <div className="card reveal" style={reveal(2)}>
      <div className="card-header">
        <h2 className="card-title">Draft outcome</h2>
        <span className="mono faint small">share of tickets</span>
      </div>
      {entries.length === 0 ? (
        <p className="muted small">No requests yet.</p>
      ) : (
        <>
          <div className="stack" aria-hidden="true">
            {entries.map(([status, count]) => (
              <span
                key={status}
                style={{
                  width: `${(count / Math.max(total, 1)) * 100}%`,
                  background: DRAFT_LABEL[status]?.color ?? 'var(--ink-3)',
                }}
              />
            ))}
          </div>
          <div className="legend">
            {entries.map(([status, count]) => (
              <div className="legend-row" key={status}>
                <i style={{ background: DRAFT_LABEL[status]?.color ?? 'var(--ink-3)' }} />
                <span>{DRAFT_LABEL[status]?.label ?? status}</span>
                <span className="mono muted">{count}</span>
                <span className="mono faint" style={{ minWidth: 44, textAlign: 'right' }}>
                  {percent(count / Math.max(total, 1))}
                </span>
              </div>
            ))}
          </div>
          <p className="small faint" style={{ marginTop: 16 }}>
            A high out-of-scope share means tickets ask about topics the help center does not cover.
          </p>
        </>
      )}
    </div>
  )
}

export function MonitoringPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [rows, setRows] = useState<RequestSummary[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [version, setVersion] = useState(0)

  useEffect(() => {
    let active = true
    Promise.all([getStats(), getRecentRequests(50)])
      .then(([summary, recent]) => {
        if (!active) return
        setStats(summary)
        setRows(recent)
        setError('')
      })
      .catch((err: unknown) => {
        if (active) setError(err instanceof Error ? err.message : 'Monitoring data could not be loaded')
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [version])

  const refresh = () => {
    setLoading(true)
    setVersion((v) => v + 1)
  }

  const total = stats?.requests ?? 0
  const alertRate = total && stats ? stats.alerts / total : null

  return (
    <>
      <header className="page-header">
        <div>
          <span className="eyebrow">03 / Monitoring</span>
          <h1>
            What the service <em>has seen.</em>
          </h1>
          <p>Live numbers from the request log. Ticket text is never stored, only its length and a hash.</p>
        </div>
        <button type="button" className="btn btn-sm" onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </header>

      {error && (
        <div className="alert-banner error" role="alert" style={{ marginBottom: 18 }}>
          <span className="mark" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      {stats && (
        <>
          <div className="card tiles reveal" style={reveal(0)}>
            <Tile label="Tickets analyzed" value={total.toLocaleString()} sub="since the log started" />
            <Tile
              label="Alert rate"
              value={percent(alertRate)}
              sub={`${stats.alerts} alerts`}
              tone={alertRate !== null && alertRate > 0.5 ? 'var(--high)' : undefined}
            />
            <Tile
              label="Needs review"
              value={percent(total ? stats.needs_review / total : null)}
              sub="low-confidence routing"
            />
            <Tile
              label="Median response"
              value={milliseconds(stats.latency_median_ms)}
              sub={`p95 ${milliseconds(stats.latency_p95_ms)}`}
            />
            <Tile
              label="Negative feedback"
              value={stats.feedback.total ? percent(stats.feedback.down_rate) : '-'}
              sub={`${stats.feedback.total} votes`}
            />
          </div>
          <div className="grid-2">
            <Categories counts={stats.categories} total={total} />
            <DraftOutcome counts={stats.draft_status} total={total} />
          </div>
        </>
      )}

      <div className="card reveal" style={reveal(3)}>
        <div className="card-header">
          <h2 className="card-title">Recent requests</h2>
          <span className="mono faint small">newest first</span>
        </div>
        {rows.length === 0 ? (
          <p className="muted small">No requests yet. Analyze a ticket and it will appear here.</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Channel</th>
                  <th>Category</th>
                  <th className="num">Confidence</th>
                  <th className="num">Risk</th>
                  <th>Draft</th>
                  <th className="num">Length</th>
                  <th className="num">Latency</th>
                  <th>Feedback</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const draft = DRAFT_LABEL[row.draft_status] ?? { label: row.draft_status, className: 'badge-neutral' }
                  return (
                    <tr key={row.request_id}>
                      <td title={row.created_at} className="muted" style={{ whiteSpace: 'nowrap' }}>
                        {relativeTime(row.created_at)}
                      </td>
                      <td className="muted">{row.channel || '-'}</td>
                      <td style={{ minWidth: 180 }}>
                        {row.category}
                        {row.needs_review && (
                          <span className="badge badge-warn" style={{ marginLeft: 8 }}>
                            review
                          </span>
                        )}
                      </td>
                      <td className="num">{percent(row.confidence)}</td>
                      <td className="num" style={{ color: row.alert ? 'var(--high)' : undefined }}>
                        {percent(row.risk_score)}
                      </td>
                      <td>
                        <span className={`badge ${draft.className}`}>{draft.label}</span>
                      </td>
                      <td className="num muted">{row.text_length}</td>
                      <td className="num muted">{milliseconds(row.latency_ms)}</td>
                      <td className="muted">
                        {row.rating === 'up' ? 'Helpful' : row.rating === 'down' ? 'Not helpful' : '-'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
