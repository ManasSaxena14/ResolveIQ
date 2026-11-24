import { useEffect, useState } from 'react'
import { getRecentRequests, getStats } from '../api/client'
import type { RequestSummary, StatsResponse } from '../api/types'
import { milliseconds, percent, relativeTime } from '../lib/format'
import { QUEUES } from '../lib/inbox'

const DRAFTS: Record<string, string> = {
  ok: 'Reply written',
  refused: 'No article found',
  unavailable: 'Articles only',
}

export function MonitoringPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [rows, setRows] = useState<RequestSummary[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [version, setVersion] = useState(0)

  useEffect(() => {
    let active = true
    Promise.all([getStats(), getRecentRequests(12)])
      .then(([summary, recent]) => {
        if (!active) return
        setStats(summary)
        setRows(Array.isArray(recent) ? recent : [])
        setError('')
      })
      .catch((err: unknown) => {
        if (active) setError(err instanceof Error ? err.message : 'The numbers could not be loaded.')
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [version])

  const total = stats?.requests ?? 0
  const share = (n: number | undefined) => (total ? percent((n ?? 0) / total) : '-')
  const most = Math.max(...Object.values(stats?.categories ?? {}), 1)
  const queues = [...QUEUES].sort((a, b) => (stats?.categories[b] ?? 0) - (stats?.categories[a] ?? 0))
  const outcomes = Object.entries(stats?.draft_status ?? {})
    .sort((a, b) => b[1] - a[1])
    .map(([k, n]) => `${(DRAFTS[k] ?? k).toLowerCase()} ${n.toLocaleString()}`)
    .join(', ')

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Monitoring</h1>
          <p className="secondary muted">What the service has sorted so far. Ticket text is never kept.</p>
        </div>
        <button
          type="button"
          className="button"
          disabled={loading}
          onClick={() => {
            setLoading(true)
            setVersion((v) => v + 1)
          }}
        >
          {loading ? 'Loading' : 'Refresh numbers'}
        </button>
      </div>

      {error && (
        <div className="notice" role="alert">
          <p>{error}</p>
        </div>
      )}

      <dl className="figures">
        <div>
          <dt>Tickets sorted</dt>
          <dd>{stats ? total.toLocaleString() : '-'}</dd>
        </div>
        <div>
          <dt>Flagged as risky</dt>
          <dd>{share(stats?.alerts)}</dd>
        </div>
        <div>
          <dt>Queue to double-check</dt>
          <dd>{share(stats?.needs_review)}</dd>
        </div>
        <div>
          <dt>Typical time</dt>
          <dd>{milliseconds(stats?.latency_median_ms)}</dd>
        </div>
        <div>
          <dt>Marked wrong</dt>
          <dd>{stats?.feedback.total ? percent(stats.feedback.down_rate) : '-'}</dd>
        </div>
      </dl>

      <section className="frame" aria-labelledby="by-queue">
        <h2 id="by-queue" className="frame-title">
          Tickets by queue
        </h2>
        <ul className="bars">
          {queues.map((label) => {
            const n = stats?.categories[label] ?? 0
            return (
              <li key={label}>
                <span className="bar-label">{label}</span>
                <span className="bar-track" aria-hidden="true">
                  <span style={{ transform: `scaleX(${n / most})` }} />
                </span>
                <span className="bar-value">{n.toLocaleString()}</span>
              </li>
            )
          })}
        </ul>
        {outcomes && <p className="frame-note secondary muted">Replies: {outcomes}.</p>}
      </section>

      <section className="frame" aria-labelledby="recent">
        <h2 id="recent" className="frame-title">
          Latest tickets
        </h2>
        <div className="table-scroll">
          <table className="stacked">
            <thead>
              <tr>
                <th scope="col">When</th>
                <th scope="col">Queue</th>
                <th scope="col" className="num">
                  Sure
                </th>
                <th scope="col" className="num">
                  Risk
                </th>
                <th scope="col">Reply</th>
                <th scope="col">Marked</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.request_id}>
                  <td data-label="When" className="nowrap">{relativeTime(row.created_at)}</td>
                  <td className="strong stack-title">
                    {row.category}
                    {row.needs_review && <span className="secondary muted">, double-check</span>}
                  </td>
                  <td data-label="Sure" className="num">{percent(row.confidence)}</td>
                  <td data-label="Risk" className="num">
                    <span className="ticket-risk" data-level={!row.alert ? 'low' : row.risk_score >= 0.6 ? 'high' : 'medium'}>
                      {percent(row.risk_score)}
                    </span>
                  </td>
                  <td data-label="Reply">{DRAFTS[row.draft_status] ?? row.draft_status}</td>
                  <td data-label="Marked">{row.rating === 'up' ? 'Right' : row.rating === 'down' ? 'Wrong' : 'Not yet'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!rows.length && (
          <div className="frame-empty">
            <p>{loading ? 'Loading the latest tickets.' : 'Nothing sorted yet. Open a ticket and it appears here.'}</p>
            <a className="button" href="#/triage">
              Open triage
            </a>
          </div>
        )}
      </section>
    </div>
  )
}
