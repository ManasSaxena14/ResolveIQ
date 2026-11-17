import { useEffect, useState } from 'react'
import { getRecentRequests, getStats } from '../api/client'
import type { RequestSummary, StatsResponse } from '../api/types'
import { milliseconds, percent, relativeTime } from '../lib/format'

const DRAFT_LABEL: Record<string, { label: string; className: string }> = {
  ok: { label: 'Drafted', className: 'badge-ok' },
  refused: { label: 'Out of scope', className: 'badge-neutral' },
  unavailable: { label: 'Unavailable', className: 'badge-warn' },
}

function Tile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
    </div>
  )
}

function Distribution({ title, counts, total }: { title: string; counts: Record<string, number>; total: number }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1])
  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">{title}</h2>
      </div>
      {entries.length === 0 ? (
        <p className="muted small">No requests yet.</p>
      ) : (
        <div className="hbar-list">
          {entries.map(([label, count]) => (
            <div className="hbar" key={label}>
              <span>{DRAFT_LABEL[label]?.label ?? label}</span>
              <div className="bar">
                <span style={{ width: percent(count / Math.max(total, 1)) }} />
              </div>
              <span className="muted" style={{ textAlign: 'right' }}>
                {count}
              </span>
            </div>
          ))}
        </div>
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

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Monitoring</h1>
          <p>What the service has seen, from the request log. Ticket text is never stored.</p>
        </div>
        <button type="button" className="btn btn-sm" onClick={refresh} disabled={loading}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="alert-banner error" role="alert">
          {error}
        </div>
      )}

      {stats && (
        <>
          <div className="tiles">
            <Tile label="Tickets analyzed" value={total.toLocaleString()} />
            <Tile label="Alert rate" value={percent(total ? stats.alerts / total : null)} sub={`${stats.alerts} alerts`} />
            <Tile
              label="Needs review"
              value={percent(total ? stats.needs_review / total : null)}
              sub="low-confidence routing"
            />
            <Tile
              label="Response time"
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
            <Distribution title="Predicted categories" counts={stats.categories} total={total} />
            <Distribution title="Draft outcome" counts={stats.draft_status} total={total} />
          </div>
        </>
      )}

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Recent requests</h2>
          <span className="small faint">newest first</span>
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
                      <td title={row.created_at}>{relativeTime(row.created_at)}</td>
                      <td className="muted">{row.channel || '-'}</td>
                      <td>
                        {row.category}
                        {row.needs_review && (
                          <span className="badge badge-warn" style={{ marginLeft: 6 }}>
                            review
                          </span>
                        )}
                      </td>
                      <td className="num">{percent(row.confidence)}</td>
                      <td className="num">
                        <span style={{ color: row.alert ? 'var(--high)' : undefined, fontWeight: row.alert ? 600 : 400 }}>
                          {percent(row.risk_score)}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${draft.className}`}>{draft.label}</span>
                      </td>
                      <td className="num muted">{row.text_length}</td>
                      <td className="num muted">{milliseconds(row.latency_ms)}</td>
                      <td className="muted">{row.rating === 'up' ? 'Helpful' : row.rating === 'down' ? 'Not helpful' : '-'}</td>
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
