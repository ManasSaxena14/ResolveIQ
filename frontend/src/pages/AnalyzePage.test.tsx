import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ANALYZE_RESPONSE, eventStreamResponse, jsonResponse, streamOf } from '../test/fixtures'
import { AnalyzePage } from './AnalyzePage'

describe('AnalyzePage', () => {
  it('analyzes a ticket and stores feedback', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(eventStreamResponse(streamOf(ANALYZE_RESPONSE)))
      .mockResolvedValueOnce(jsonResponse({ request_id: 'abc123def456', rating: 'up', stored: true }))
    render(<AnalyzePage />)

    expect(screen.getByRole('button', { name: 'Analyze ticket' })).toBeDisabled()
    await user.type(screen.getByLabelText('Customer message'), 'I was charged twice')
    await user.click(screen.getByRole('button', { name: 'Analyze ticket' }))

    expect(await screen.findByText('abc123def456')).toBeInTheDocument()
    expect(screen.getByText(/disappears within three days/)).toBeInTheDocument()
    expect(fetchMock.mock.calls[0][0]).toBe('/analyze/stream')
    expect(screen.getByRole('link', { name: '1' })).toHaveAttribute('href', '#source-1')
    expect(screen.getByRole('region', { name: 'Why these results' })).toBeInTheDocument()
    expect(screen.getByText('905 ms')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Helpful' }))
    expect(await screen.findByText('Thanks for the feedback')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Helpful' })).toHaveAttribute('aria-pressed', 'true')
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({ rating: 'up' })
  })

  it('fills the form from an example', async () => {
    const user = userEvent.setup()
    render(<AnalyzePage />)

    await user.click(screen.getByRole('button', { name: 'Outage' }))

    expect((screen.getByLabelText('Customer message') as HTMLTextAreaElement).value).toContain(
      'cannot log in',
    )
    expect(screen.getByLabelText('Type')).toHaveValue('Incident')
  })

  it('blocks text over the limit', async () => {
    render(<AnalyzePage />)
    const box = screen.getByLabelText('Customer message')

    await userEvent.click(box)
    await userEvent.paste('x'.repeat(4001))

    expect(screen.getByText('4,001 / 4,000')).toHaveClass('over')
    expect(screen.getByRole('button', { name: 'Analyze ticket' })).toBeDisabled()
  })

  it('shows API errors', async () => {
    const user = userEvent.setup()
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))
    render(<AnalyzePage />)

    await user.type(screen.getByLabelText('Customer message'), 'Hello')
    await user.click(screen.getByRole('button', { name: 'Analyze ticket' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('cannot be reached')
  })
})
