import { act, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { jsonResponse } from './test/fixtures'

const HEALTH = {
  status: 'ok',
  components: { retriever: true, classifier: true },
  versions: { risk: 'xgboost' },
  index: { files: 31, chunks: 31 },
  uptime_seconds: 1,
}

afterEach(() => {
  vi.useRealTimers()
})

describe('App wake-up', () => {
  it('explains a sleeping server, then clears once it answers', async () => {
    vi.useFakeTimers()
    let answer: (response: Response) => void = () => {}
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      () => new Promise<Response>((resolve) => (answer = resolve)),
    )
    render(<App />)

    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    await act(async () => {
      vi.advanceTimersByTime(2600)
    })
    expect(screen.getByRole('status')).toHaveTextContent('waking up')
    expect(screen.getAllByText('Waking up').length).toBeGreaterThan(0)

    await act(async () => {
      answer(jsonResponse(HEALTH))
    })
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
    expect(screen.getAllByText('All systems ready').length).toBeGreaterThan(0)
  })

  it('shows no banner when the server answers quickly', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse(HEALTH))
    render(<App />)

    expect((await screen.findAllByText('All systems ready')).length).toBeGreaterThan(0)
    expect(screen.queryByRole('status')).not.toBeInTheDocument()
  })
})
