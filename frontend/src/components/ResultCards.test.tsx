import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ANALYZE_RESPONSE } from '../test/fixtures'
import { CategoryCard, DraftCard, RiskCard } from './ResultCards'

describe('DraftCard', () => {
  it('shows the draft and links each source', () => {
    render(<DraftCard draft={ANALYZE_RESPONSE.draft_reply} />)

    expect(screen.getByText(/disappears within three days/)).toBeInTheDocument()
    const link = screen.getByRole('link', { name: 'Double charges and pending authorizations' })
    expect(link).toHaveAttribute('href', expect.stringContaining('billing/double-charges'))
    expect(screen.getByText('Grounded draft')).toBeInTheDocument()
  })

  it('explains an unavailable draft without hiding the sources', () => {
    render(
      <DraftCard
        draft={{
          ...ANALYZE_RESPONSE.draft_reply,
          text: null,
          status: 'unavailable',
          error: 'The language model is unavailable',
        }}
      />,
    )

    expect(screen.getByText(/language model is unavailable\. The category/)).toBeInTheDocument()
    expect(screen.getByRole('link')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Copy draft' })).not.toBeInTheDocument()
  })

  it('marks out-of-scope questions', () => {
    render(
      <DraftCard
        draft={{ text: 'I could not find this.', status: 'refused', refused: true, sources: [], error: null }}
      />,
    )

    expect(screen.getByText('Out of scope')).toBeInTheDocument()
  })
})

describe('CategoryCard', () => {
  it('shows the label, confidence and review state', () => {
    const { rerender } = render(<CategoryCard category={ANALYZE_RESPONSE.category} />)

    expect(screen.getAllByText('Billing and Payments')).toHaveLength(2)
    expect(screen.getByText('Auto-route')).toBeInTheDocument()

    rerender(<CategoryCard category={{ ...ANALYZE_RESPONSE.category, needs_review: true }} />)
    expect(screen.getByText('Needs review')).toBeInTheDocument()
  })
})

describe('RiskCard', () => {
  it('shows the probability, alert and reasons', () => {
    render(<RiskCard risk={ANALYZE_RESPONSE.risk} />)

    expect(screen.getByRole('img', { name: /82 percent, alert threshold 23 percent/ })).toBeInTheDocument()
    expect(screen.getByText('High risk')).toBeInTheDocument()
    expect(screen.getByText('Alert: prioritize this ticket')).toBeInTheDocument()
    expect(screen.getByRole('list', { name: 'Top reasons' }).children).toHaveLength(1)
  })

  it('handles a calm ticket without reasons', () => {
    render(<RiskCard risk={{ escalation_probability: 0.05, alert: false, threshold: 0.23, reasons: [] }} />)

    expect(screen.getByText('Low risk')).toBeInTheDocument()
    expect(screen.getByText('No alert')).toBeInTheDocument()
    expect(screen.getByText(/No single feature/)).toBeInTheDocument()
  })
})
