import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ANALYZE_RESPONSE } from '../test/fixtures'
import { CategoryCard, DraftCard, RiskCard, Timeline, Verdict } from './ResultCards'

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

describe('Verdict and Timeline', () => {
  it('summarizes route, risk and reply', () => {
    render(<Verdict result={ANALYZE_RESPONSE} />)

    const verdict = screen.getByRole('region', { name: 'Verdict' })
    expect(verdict).toHaveTextContent('Billing and Payments')
    expect(verdict).toHaveTextContent('Above the alert threshold')
    expect(verdict).toHaveTextContent('Ready to edit')
    expect(verdict).toHaveTextContent('1 source from the help center')
    expect(verdict).not.toHaveTextContent('needs review')
  })

  it('flags low-confidence routing in the verdict', () => {
    const result = { ...ANALYZE_RESPONSE, category: { ...ANALYZE_RESPONSE.category, needs_review: true } }
    render(<Verdict result={result} />)

    expect(screen.getByRole('region', { name: 'Verdict' })).toHaveTextContent('needs review')
  })

  it('accounts for time outside the named steps', () => {
    render(<Timeline timings={{ retrieval: 20, classifier: 30, risk: 50, total: 1100 }} />)

    expect(screen.getByText('1.1 s')).toBeInTheDocument()
    expect(screen.getByText('Other')).toBeInTheDocument()
    expect(screen.getByText('1.0 s')).toBeInTheDocument()
  })

  it('shows no extra segment when the steps add up', () => {
    render(<Timeline timings={ANALYZE_RESPONSE.timings_ms} />)

    expect(screen.queryByText('Other')).not.toBeInTheDocument()
    expect(screen.getByText('Generation')).toBeInTheDocument()
  })
})

describe('DraftCard while streaming and after', () => {
  it('shows the text as it arrives with a cursor', () => {
    const { container } = render(
      <DraftCard
        streaming
        draft={{ text: 'The second charge is', status: 'ok', refused: false, sources: [], error: null }}
      />,
    )

    expect(screen.getByText('Writing')).toBeInTheDocument()
    expect(container.querySelector('.caret')).not.toBeNull()
    expect(screen.queryByRole('button', { name: 'Copy draft' })).not.toBeInTheDocument()
  })

  it('reports grounding and flags citations to unknown sources', () => {
    render(
      <DraftCard
        draft={{
          ...ANALYZE_RESPONSE.draft_reply,
          text: 'A hold [double-charges-and-pending-authorizations.md, page 1]. Free [ship.md, page 1].',
          grounding: {
            sentences: 2,
            cited_sentences: 2,
            citations: 2,
            coverage: 1,
            invalid_citations: ['ship.md, page 1'],
          },
        }}
      />,
    )

    expect(screen.getByRole('link', { name: '1' })).toHaveAttribute('href', '#source-1')
    expect(screen.getByTitle('Not a retrieved source: ship.md, page 1')).toBeInTheDocument()
    expect(screen.getByText(/ship.md, page 1 was not retrieved/)).toBeInTheDocument()
  })
})

describe('WhyCard', () => {
  it('highlights category words and switches to risk words', async () => {
    const { WhyCard } = await import('./WhyCard')
    const userEvent = (await import('@testing-library/user-event')).default
    const user = userEvent.setup()
    const { container } = render(
      <WhyCard subject="" text="Charged twice, refund please. Thanks" highlights={ANALYZE_RESPONSE.highlights!} />,
    )

    expect([...container.querySelectorAll('mark')].map((m) => m.textContent)).toEqual(['Charged', 'refund'])

    await user.click(screen.getByRole('button', { name: 'Risk' }))
    expect([...container.querySelectorAll('mark')].map((m) => m.textContent)).toEqual(['twice', 'Thanks'])
    expect(screen.getByRole('list', { name: 'Top risk words' })).toHaveTextContent('+0.90')
  })

  it('renders nothing without highlights', async () => {
    const { WhyCard } = await import('./WhyCard')
    const { container } = render(<WhyCard subject="" text="Hi" highlights={{ category: [], risk: [] }} />)

    expect(container).toBeEmptyDOMElement()
  })
})
