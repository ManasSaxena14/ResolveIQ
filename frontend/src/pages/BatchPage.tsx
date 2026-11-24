import { useMemo, useRef, useState, type DragEvent } from 'react'
import { ApiError, scoreBatch } from '../api/client'
import type { BatchResponse } from '../api/types'
import { SAMPLE_CSV, percent, toCsv } from '../lib/format'

const MAX_BYTES = 2_000_000
const PAGE_SIZE = 25

function download(name: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv' }))
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  URL.revokeObjectURL(url)
}

export function BatchPage() {
  const [result, setResult] = useState<BatchResponse | null>(null)
  const [fileName, setFileName] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [query, setQuery] = useState('')
  const [flaggedOnly, setFlaggedOnly] = useState(false)
  const [page, setPage] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  const score = async (file: File | undefined) => {
    setError('')
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.csv')) return setError('That file is not a CSV. Save it as CSV and choose it again.')
    if (file.size > MAX_BYTES) return setError('That file is over 2 MB. Split it and score the parts.')
    setLoading(true)
    setFileName(file.name)
    try {
      setResult(await scoreBatch(file))
      setPage(0)
    } catch (err) {
      setResult(null)
      setError(err instanceof ApiError ? err.message : 'The file could not be scored. Try again.')
    } finally {
      setLoading(false)
      if (input.current) input.current.value = ''
    }
  }

  const items = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return (result?.items ?? []).filter(
      (i) =>
        (!flaggedOnly || i.alert) &&
        (!needle || `${i.id} ${i.reasons.join(' ')}`.toLowerCase().includes(needle)),
    )
  }, [result, query, flaggedOnly])
  const pages = Math.max(1, Math.ceil(items.length / PAGE_SIZE))
  const shown = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    void score(event.dataTransfer.files[0])
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Batch</h1>
          <p className="secondary muted">
            {result
              ? `${fileName}: ${result.count.toLocaleString()} tickets, ${result.alerts.toLocaleString()} flagged.`
              : 'Score a file of tickets and get them back riskiest first.'}
          </p>
        </div>
        <button type="button" className="button primary" onClick={() => input.current?.click()} disabled={loading}>
          {loading ? 'Scoring' : 'Choose CSV'}
        </button>
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          aria-label="Tickets CSV"
          className="visually-hidden"
          tabIndex={-1}
          onChange={(e) => void score(e.target.files?.[0])}
        />
      </div>

      {error && (
        <div className="notice" role="alert">
          <p>{error}</p>
        </div>
      )}

      <div
        className="frame"
        data-dragging={dragging || undefined}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className="toolbar">
          <label className="search">
            <span className="visually-hidden">Search results</span>
            <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
              <circle cx="9" cy="9" r="5.5" />
              <path d="m13.5 13.5 4 4" />
            </svg>
            <input
              type="search"
              placeholder="Search results"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setPage(0)
              }}
            />
          </label>
          <button
            type="button"
            className="button"
            aria-pressed={flaggedOnly}
            onClick={() => {
              setFlaggedOnly((v) => !v)
              setPage(0)
            }}
          >
            Flagged only
          </button>
          <button
            type="button"
            className="button toolbar-end"
            disabled={!result}
            onClick={() =>
              result &&
              download(
                'resolveiq_ranked.csv',
                toCsv(
                  ['rank', 'id', 'escalation_probability', 'flagged', 'reasons'],
                  result.items.map((i) => [i.rank, i.id, i.escalation_probability, i.alert ? 1 : 0, i.reasons.join('; ')]),
                ),
              )
            }
          >
            Export CSV
          </button>
        </div>
        <div className="table-scroll">
          <table className="results">
            <thead>
              <tr>
                <th scope="col" className="num">
                  Rank
                </th>
                <th scope="col">Ticket</th>
                <th scope="col">Risk</th>
                <th scope="col">Why</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((item) => (
                <tr key={`${item.rank}-${item.id}`}>
                  <td className="num">{item.rank}</td>
                  <td className="nowrap">{item.id}</td>
                  <td className="nowrap">
                    <span className="ticket-risk" data-level={!item.alert ? 'low' : item.escalation_probability >= 0.6 ? 'high' : 'medium'}>
                      {percent(item.escalation_probability)}
                    </span>
                    {item.alert && <span className="secondary"> flagged</span>}
                  </td>
                  <td>{item.reasons.join('. ') || 'Nothing raises the risk.'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!shown.length && (
          <div className="frame-empty">
            {result ? (
              <>
                <p>No result matches.</p>
                <button
                  type="button"
                  className="button"
                  onClick={() => {
                    setQuery('')
                    setFlaggedOnly(false)
                  }}
                >
                  Show all results
                </button>
              </>
            ) : (
              <>
                <p>
                  Drop a CSV here, or score the sample. It needs a <code>body</code> column; <code>subject</code>,{' '}
                  <code>type</code> and <code>id</code> are optional. Up to 2 MB and 2,000 rows.
                </p>
                <div className="row">
                  <button
                    type="button"
                    className="button"
                    disabled={loading}
                    onClick={() => void score(new File([SAMPLE_CSV], 'sample_tickets.csv', { type: 'text/csv' }))}
                  >
                    Score the sample
                  </button>
                  <button type="button" className="button" onClick={() => download('sample_tickets.csv', SAMPLE_CSV)}>
                    Download the sample
                  </button>
                </div>
              </>
            )}
          </div>
        )}
        <div className="pager">
          <button type="button" className="button" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
            Previous
          </button>
          <span className="secondary muted">
            Page {page + 1} of {pages}
          </span>
          <button type="button" className="button" disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)}>
            Next
          </button>
        </div>
      </div>
    </div>
  )
}
