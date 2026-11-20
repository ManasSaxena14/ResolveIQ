import { useRef, useState, type CSSProperties, type DragEvent } from 'react'
import { ApiError, scoreBatch } from '../api/client'
import type { BatchResponse } from '../api/types'
import { SAMPLE_CSV, percent, riskLevel, toCsv } from '../lib/format'

const MAX_BYTES = 2_000_000

function download(name: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv' }))
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  URL.revokeObjectURL(url)
}

export function BatchPage() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<BatchResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [alertsOnly, setAlertsOnly] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const choose = (selected: File | undefined) => {
    setResult(null)
    setError('')
    if (!selected) return
    if (!selected.name.toLowerCase().endsWith('.csv')) {
      setError('Choose a CSV file.')
      return
    }
    if (selected.size > MAX_BYTES) {
      setError('The file is larger than 2 MB.')
      return
    }
    setFile(selected)
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    choose(event.dataTransfer.files[0])
  }

  const run = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      setResult(await scoreBatch(file))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The file could not be scored.')
    } finally {
      setLoading(false)
    }
  }

  const items = result ? result.items.filter((item) => !alertsOnly || item.alert) : []
  // The batch response carries no threshold; the lowest alerting score bounds it.
  const threshold = result
    ? Math.min(...result.items.filter((i) => i.alert).map((i) => i.escalation_probability), 1)
    : 0.5

  const exportResults = () => {
    if (!result) return
    download(
      'resolveiq_ranked.csv',
      toCsv(
        ['rank', 'id', 'escalation_probability', 'alert', 'reasons'],
        result.items.map((i) => [i.rank, i.id, i.escalation_probability, i.alert ? 1 : 0, i.reasons.join('; ')]),
      ),
    )
  }

  return (
    <>
      <header className="page-header">
        <div>
          <span className="eyebrow">02 / Batch</span>
          <h1>
            Rank the queue. <em>Worst first.</em>
          </h1>
          <p>Upload a CSV of tickets and get them ordered by escalation risk, with the reasons for each.</p>
        </div>
        <button type="button" className="btn btn-sm" onClick={() => download('sample_tickets.csv', SAMPLE_CSV)}>
          Download sample CSV
        </button>
      </header>

      <div className="card" style={{ marginBottom: 16 }}>
        <label
          className={`dropzone${dragging ? ' active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={input}
            type="file"
            accept=".csv,text/csv"
            aria-label="Tickets CSV"
            onChange={(e) => choose(e.target.files?.[0])}
          />
          <span className="eyebrow">{file ? 'Ready to score' : 'CSV intake'}</span>
          <span className="dropzone-title">{file ? file.name : 'Drop a CSV here, or click to choose'}</span>
          <span className="small muted">
            Columns: <span className="mono">body</span> or <span className="mono">text</span> required;{' '}
            <span className="mono">subject</span>, <span className="mono">type</span> and{' '}
            <span className="mono">id</span> optional. Up to 2 MB and 2,000 rows.
          </span>
        </label>
        <div className="actions">
          <button type="button" className="btn btn-primary" disabled={!file || loading} onClick={run}>
            <span className="dot" aria-hidden="true" />
            {loading ? 'Scoring...' : 'Score tickets'}
          </button>
          {file && (
            <button
              type="button"
              className="btn"
              onClick={() => {
                setFile(null)
                setResult(null)
                if (input.current) input.current.value = ''
              }}
            >
              Clear
            </button>
          )}
        </div>
        {error && (
          <div className="alert-banner error" role="alert" style={{ marginTop: 14 }}>
            <span className="mark" aria-hidden="true" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {result && (
        <>
          <div className="card tiles reveal">
            <div className="tile">
              <div className="eyebrow">Tickets scored</div>
              <div className="tile-value">{result.count.toLocaleString()}</div>
              <div className="tile-sub">from {file?.name ?? 'the upload'}</div>
            </div>
            <div className="tile">
              <div className="eyebrow">Alerts</div>
              <div className="tile-value" style={{ color: result.alerts ? 'var(--high)' : undefined }}>
                {result.alerts.toLocaleString()}
              </div>
              <div className="tile-sub">{percent(result.alerts / Math.max(result.count, 1))} of tickets</div>
            </div>
            <div className="tile">
              <div className="eyebrow">Highest risk</div>
              <div className="tile-value">{percent(result.items[0]?.escalation_probability ?? 0)}</div>
              <div className="tile-sub">
                ticket <span className="mono">{result.items[0]?.id}</span>
              </div>
            </div>
          </div>

          <div className="card reveal" style={{ '--i': 1 } as CSSProperties}>
            <div className="card-header">
              <h2 className="card-title">Ranked tickets</h2>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="button"
                  className="btn btn-sm"
                  aria-pressed={alertsOnly}
                  onClick={() => setAlertsOnly((v) => !v)}
                >
                  Alerts only
                </button>
                <button type="button" className="btn btn-sm" onClick={exportResults}>
                  Export CSV
                </button>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th className="num">Rank</th>
                    <th>Ticket</th>
                    <th>Risk</th>
                    <th>Alert</th>
                    <th>Top reasons</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => {
                    const level = riskLevel(item.escalation_probability, threshold)
                    return (
                      <tr key={`${item.rank}-${item.id}`}>
                        <td className="num faint">{String(item.rank).padStart(2, '0')}</td>
                        <td className="mono" style={{ whiteSpace: 'nowrap' }}>
                          {item.id}
                        </td>
                        <td>
                          <div className="risk-cell">
                            <span>{percent(item.escalation_probability)}</span>
                            <div className={`bar ${level}`}>
                              <span style={{ width: percent(item.escalation_probability) }} />
                            </div>
                          </div>
                        </td>
                        <td>
                          {item.alert ? (
                            <span className="badge badge-dot badge-high">Alert</span>
                          ) : (
                            <span className="badge badge-neutral">No</span>
                          )}
                        </td>
                        <td className="small">{item.reasons.join('; ') || <span className="faint">None</span>}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  )
}
