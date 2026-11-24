import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { SAMPLE_TICKETS } from './lib/inbox'
import { ANALYZE_RESPONSE, HEALTH, apiFetch, eventStreamResponse, jsonResponse, streamOf } from './test/fixtures'

/** Every sample ticket scored, with the last one riskiest. */
const RANKING = {
  count: SAMPLE_TICKETS.length,
  alerts: 1,
  items: SAMPLE_TICKETS.map((t, i) => ({
    rank: i + 1,
    id: t.id,
    escalation_probability: i / 100,
    alert: t.id === 'careers',
    reasons: [],
  })),
}

const ROUTES = {
  '/health': () => jsonResponse(HEALTH),
  '/batch': () => jsonResponse(RANKING),
  '/analyze/stream': () => eventStreamResponse(streamOf(ANALYZE_RESPONSE)),
  '/feedback': () => jsonResponse({ request_id: 'abc123def456', rating: 'up', stored: true }),
  '/stats': () =>
    jsonResponse({
      requests: 10,
      alerts: 4,
      needs_review: 2,
      mean_risk: 0.3,
      mean_confidence: 0.8,
      categories: { 'Billing and Payments': 10 },
      draft_status: { refused: 10 },
      latency_median_ms: 42,
      latency_p95_ms: 90,
      feedback: { total: 4, up: 3, down: 1, down_rate: 0.25 },
    }),
  '/requests': () => jsonResponse([]),
}

function start(routes: Parameters<typeof apiFetch>[0] = ROUTES) {
  const api = apiFetch(routes)
  vi.spyOn(globalThis, 'fetch').mockImplementation(api.handler)
  render(<App />)
  return api
}

beforeEach(() => {
  window.history.replaceState(null, '', '/')
})

afterEach(() => {
  vi.useRealTimers()
})

describe('Triage', () => {
  it('opens the riskiest ticket and sorts it', async () => {
    start()

    expect(await screen.findByRole('heading', { level: 2, name: 'Engineering internships' })).toBeInTheDocument()
    expect(window.location.hash).toBe('#/triage/careers')
    expect(await screen.findByRole('heading', { name: 'Billing and Payments' })).toBeInTheDocument()
    expect(await screen.findByText('Took 905 ms')).toBeInTheDocument()
  })

  it('saves whether the ticket was sorted right', async () => {
    start()

    await userEvent.click(await screen.findByRole('button', { name: 'Yes, right' }))
    expect(await screen.findByText('Answer saved.')).toBeInTheDocument()
  })

  it('triages each ticket once, however often it is opened', async () => {
    const api = start()
    await screen.findByText('Took 905 ms')

    await userEvent.click(screen.getByRole('button', { name: /Charged twice/ }))
    await screen.findByRole('heading', { level: 2, name: 'Charged twice for one order' })
    await screen.findByText('Took 905 ms')
    await userEvent.click(screen.getByRole('button', { name: /Engineering internships/ }))
    await screen.findByText('Took 905 ms')

    expect(api.calls.filter((c) => c === '/analyze/stream')).toHaveLength(2)
  })

  it('writes a new ticket with the keyboard and puts it in the inbox', async () => {
    start()
    await screen.findByText('Took 905 ms')

    await userEvent.keyboard('n')
    await userEvent.type(await screen.findByLabelText('Customer message'), 'My invoice is wrong')
    await userEvent.click(screen.getByRole('button', { name: 'Triage ticket' }))

    expect(await screen.findByRole('heading', { level: 2, name: 'My invoice is wrong' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /My invoice is wrong/ })).toHaveAttribute('aria-current', 'true')
  })

  it('explains a failure and lets the ticket be triaged again', async () => {
    let fail = true
    start({
      ...ROUTES,
      '/analyze/stream': () =>
        fail ? jsonResponse({ detail: 'Too many requests. Try again in 5 seconds.' }, 429) : ROUTES['/analyze/stream'](),
    })

    expect(await screen.findByText(/Too many requests/)).toBeInTheDocument()
    fail = false
    await userEvent.click(screen.getByRole('button', { name: 'Triage again' }))
    expect(await screen.findByText('Took 905 ms')).toBeInTheDocument()
  })
})

describe('Other pages', () => {
  it('scores the sample file and ranks it', async () => {
    window.history.replaceState(null, '', '#/batch')
    start()

    await userEvent.click(await screen.findByRole('button', { name: 'Score the sample' }))
    expect(await screen.findByText('sample_tickets.csv: 12 tickets, 1 flagged.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Flagged only' }))
    expect(screen.getByRole('cell', { name: 'careers' })).toBeInTheDocument()
    expect(screen.queryByRole('cell', { name: 'charged-twice' })).not.toBeInTheDocument()
  })

  it('refuses files that are not CSV with a way forward', async () => {
    window.history.replaceState(null, '', '#/batch')
    start()

    await userEvent.setup({ applyAccept: false }).upload(screen.getByLabelText('Tickets CSV'), new File(['x'], 'a.pdf'))
    expect(await screen.findByRole('alert')).toHaveTextContent('Save it as CSV and choose it again.')
  })

  it('shows the numbers and an empty list of latest tickets', async () => {
    window.history.replaceState(null, '', '#/monitoring')
    start()

    expect(await screen.findByText('40%')).toBeInTheDocument()
    expect(screen.getByText('Replies: no article found 10.')).toBeInTheDocument()
    expect(await screen.findByText(/Nothing sorted yet/)).toBeInTheDocument()
  })

  it('explains itself honestly', async () => {
    window.history.replaceState(null, '', '#/about')
    start()

    expect(await screen.findByText(/right 14 out of 14 times/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'What it cannot do' })).toBeInTheDocument()
  })
})

describe('Server state', () => {
  it('redoes what failed once the server is back', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    let online = false
    const routes = Object.fromEntries(
      Object.entries(ROUTES).map(([path, reply]) => [
        path,
        () => (online ? reply() : Promise.reject(new TypeError('Failed to fetch'))),
      ]),
    )
    start(routes)

    expect(await screen.findByText(/The server cannot be reached/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Not sorted yet' })).toBeInTheDocument()
    expect(screen.getByText('Offline')).toBeInTheDocument()

    online = true
    await act(async () => {
      vi.advanceTimersByTime(5100)
    })
    expect(await screen.findByText('Took 905 ms')).toBeInTheDocument()
    expect(screen.queryByText(/risk order could not be loaded/)).not.toBeInTheDocument()
  })

  it('explains a sleeping server, then clears once it answers', async () => {
    vi.useFakeTimers()
    let answer: (response: Response) => void = () => {}
    start({ ...ROUTES, '/health': () => new Promise<Response>((resolve) => (answer = resolve)) })

    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    await act(async () => {
      vi.advanceTimersByTime(2600)
    })
    expect(screen.getByRole('status')).toHaveTextContent('waking up')

    await act(async () => {
      answer(jsonResponse(HEALTH))
    })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getByText('Ready')).toBeInTheDocument()
  })
})
