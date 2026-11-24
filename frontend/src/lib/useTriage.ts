import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, analyzeTicketStream, scoreBatch } from '../api/client'
import type { AnalysisEvent, AnalyzeResponse, Source } from '../api/types'
import { toCsv } from './format'
import { SAMPLE_TICKETS, type Ticket } from './inbox'

export interface Score {
  probability: number
  alert: boolean
}

export interface TicketResult {
  status: 'running' | 'done' | 'error'
  analysis: AnalysisEvent | null
  sources: Source[]
  text: string
  final: AnalyzeResponse | null
  error: string
}

export type ScoreState = 'loading' | 'ready' | 'error'

const EMPTY: TicketResult = { status: 'running', analysis: null, sources: [], text: '', final: null, error: '' }

function inboxCsv(tickets: Ticket[]): File {
  const rows = tickets.map((t) => [t.id, t.subject, t.text, t.ticket_type])
  return new File([toCsv(['id', 'subject', 'body', 'type'], rows)], 'inbox.csv', { type: 'text/csv' })
}

/** The inbox, its risk ranking and the analysis of every ticket opened so far. */
export function useTriage() {
  const [tickets, setTickets] = useState<Ticket[]>(SAMPLE_TICKETS)
  const [scores, setScores] = useState<Record<string, Score>>({})
  const [scoreState, setScoreState] = useState<ScoreState>('loading')
  const [results, setResults] = useState<Record<string, TicketResult>>({})
  const [attempt, setAttempt] = useState(0)
  const started = useRef(new Set<string>())

  useEffect(() => {
    let active = true
    scoreBatch(inboxCsv(SAMPLE_TICKETS))
      .then((batch) => {
        if (!active) return
        if (!Array.isArray(batch?.items)) throw new Error('No ranking returned')
        const ranked = Object.fromEntries(
          batch.items.map((i) => [i.id, { probability: i.escalation_probability, alert: i.alert }]),
        )
        setScores((current) => ({ ...ranked, ...current }))
        setScoreState('ready')
      })
      .catch(() => {
        if (active) setScoreState('error')
      })
    return () => {
      active = false
    }
  }, [attempt])

  const retryScores = useCallback(() => {
    setScoreState('loading')
    setAttempt((n) => n + 1)
  }, [])

  const analyze = useCallback(async (ticket: Ticket, again = false) => {
    if (started.current.has(ticket.id) && !again) return
    started.current.add(ticket.id)
    const patch = (change: (r: TicketResult) => TicketResult) =>
      setResults((all) => ({ ...all, [ticket.id]: change(all[ticket.id] ?? EMPTY) }))
    patch(() => EMPTY)
    try {
      const { data } = await analyzeTicketStream(
        {
          text: ticket.text,
          subject: ticket.subject,
          ticket_type: ticket.ticket_type,
          channel: ticket.channel,
          product: ticket.product,
        },
        {
          onAnalysis: (analysis) => {
            patch((r) => ({ ...r, analysis }))
            setScores((s) => ({
              ...s,
              [ticket.id]: {
                probability: analysis.risk.escalation_probability,
                alert: analysis.risk.alert,
              },
            }))
          },
          onSources: (sources) => patch((r) => ({ ...r, sources })),
          onToken: (piece) => patch((r) => ({ ...r, text: r.text + piece })),
        },
      )
      patch((r) => ({
        ...r,
        status: 'done',
        final: data,
        analysis: r.analysis ?? {
          request_id: data.request_id,
          category: data.category,
          risk: data.risk,
          highlights: data.highlights ?? { category: [], risk: [] },
          versions: data.versions,
        },
      }))
    } catch (err) {
      started.current.delete(ticket.id)
      const error = err instanceof ApiError ? err.message : 'This ticket could not be triaged.'
      patch((r) => ({ ...r, status: 'error', error }))
    }
  }, [])

  /** Does again whatever failed while the server was away. */
  const recover = useCallback(() => {
    if (scoreState === 'error') retryScores()
    setResults((all) => Object.fromEntries(Object.entries(all).filter(([, r]) => r.status !== 'error')))
  }, [scoreState, retryScores])

  const addTicket = useCallback((ticket: Ticket) => setTickets((all) => [ticket, ...all]), [])

  /** Riskiest first; tickets without a score keep the order they arrived in. */
  const ranked = useMemo(
    () =>
      tickets
        .map((ticket, index) => ({ ticket, index, score: scores[ticket.id] }))
        .sort((a, b) => (b.score?.probability ?? -1) - (a.score?.probability ?? -1) || a.index - b.index)
        .map(({ ticket }) => ticket),
    [tickets, scores],
  )

  return { tickets: ranked, scores, scoreState, results, analyze, addTicket, retryScores, recover }
}

export type Triage = ReturnType<typeof useTriage>
