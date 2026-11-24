import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { EMPTY_DRAFT, QUEUES, SAMPLE_TICKETS } from '../lib/inbox'
import { ANALYZE_RESPONSE } from '../test/fixtures'
import { Composer } from './Composer'
import { Draft } from './Draft'
import { RiskScale } from './RiskScale'
import { SortingRack } from './SortingRack'
import { TicketList } from './TicketList'
import { TicketText } from './TicketText'

describe('SortingRack', () => {
  it('shows all ten queues and lays the slip in the chosen one', () => {
    const { container } = render(<SortingRack category={ANALYZE_RESPONSE.category} subject="Charged twice" />)

    expect(container.querySelectorAll('.slot')).toHaveLength(QUEUES.length)
    const chosen = container.querySelector('[data-chosen]')!
    expect(chosen).toHaveTextContent('Charged twice')
    expect(chosen).toHaveTextContent('Billing and Payments')
    expect(screen.getByText(/Billing and Payments 94%, Customer Service 3%/)).toBeInTheDocument()
  })

  it('fills other slots by their real probability, with no minimum', () => {
    const { container } = render(<SortingRack category={ANALYZE_RESPONSE.category} subject="x" />)
    const fills = [...container.querySelectorAll<HTMLElement>('.slot-fill')].map((f) => f.style.transform)

    expect(fills).toContain('scaleY(0.03)')
    expect(fills.filter((f) => f === 'scaleY(0)')).toHaveLength(8)
  })

  it('stands empty before a ticket is sorted', () => {
    const { container } = render(<SortingRack category={null} subject="" />)

    expect(screen.getByText('Not sorted yet')).toBeInTheDocument()
    expect(container.querySelector('.slip')).toBeNull()
  })
})

describe('RiskScale', () => {
  it('states the risk, the threshold and the reasons', () => {
    render(<RiskScale risk={ANALYZE_RESPONSE.risk} />)

    expect(screen.getByRole('img', { name: 'Escalation risk 82 percent, flagged from 23 percent' })).toBeInTheDocument()
    expect(screen.getByText('Flagged for a quick answer.')).toBeInTheDocument()
    expect(within(screen.getByRole('list', { name: 'Reasons' })).getAllByRole('listitem')).toHaveLength(1)
  })

  it('says when a ticket is not flagged', () => {
    render(<RiskScale risk={{ escalation_probability: 0.1, alert: false, threshold: 0.23, reasons: [] }} />)

    expect(screen.getByText('Not flagged. Tickets are flagged from 23%.')).toBeInTheDocument()
    expect(screen.queryByRole('list')).not.toBeInTheDocument()
  })
})

describe('TicketText', () => {
  it('marks the words behind the queue, then behind the risk', async () => {
    const { container } = render(
      <TicketText text="Charged twice, refund please. Thanks" highlights={ANALYZE_RESPONSE.highlights!} />,
    )
    const marks = () => [...container.querySelectorAll('mark')].map((m) => `${m.textContent}:${m.dataset.kind}`)

    expect(marks()).toEqual(['Charged:queue', 'refund:queue'])
    await userEvent.click(screen.getByRole('button', { name: 'Risk words' }))
    expect(marks()).toEqual(['twice:raises', 'Thanks:lowers'])
  })

  it('shows the plain message until there is an explanation', () => {
    const { container } = render(<TicketText text="Hello" highlights={null} />)

    expect(container.querySelector('mark')).toBeNull()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})

describe('Draft', () => {
  const props = { liveText: '', liveSources: [] }

  it('links citations to their articles and reports the grounding', async () => {
    const draft = {
      ...ANALYZE_RESPONSE.draft_reply,
      grounding: { sentences: 1, cited_sentences: 1, citations: 1, coverage: 1, invalid_citations: [] },
    }
    render(<Draft draft={draft} {...props} />)

    expect(screen.getByText('1 of 1 sentences cite an article, and every citation is real.')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /^Article 1/ }))
    expect(screen.getByRole('link', { name: /Double charges/ })).toHaveFocus()
  })

  it('warns about a citation that was never retrieved', () => {
    const draft = {
      ...ANALYZE_RESPONSE.draft_reply,
      text: 'Free [ship.md, page 1].',
      grounding: { sentences: 1, cited_sentences: 1, citations: 1, coverage: 1, invalid_citations: ['ship.md, page 1'] },
    }
    render(<Draft draft={draft} {...props} />)

    expect(screen.getByRole('note')).toHaveTextContent('ship.md, page 1 was not among the articles found')
    expect(screen.getByTitle('Not a retrieved article: ship.md, page 1')).toBeInTheDocument()
  })

  it('writes live, and explains missing replies', () => {
    const { container, rerender } = render(<Draft draft={null} liveText="The second" liveSources={[]} />)
    expect(container.querySelector('.caret')).not.toBeNull()
    expect(screen.queryByRole('button', { name: 'Copy reply' })).not.toBeInTheDocument()

    rerender(<Draft draft={{ text: 'x', status: 'refused', refused: true, sources: [], error: null }} {...props} />)
    expect(screen.getByText(/help center has no article on this/)).toBeInTheDocument()

    rerender(
      <Draft draft={{ text: null, status: 'unavailable', refused: false, sources: [], error: 'Model down' }} {...props} />,
    )
    expect(screen.getByText('Model down. The queue and risk above still hold.')).toBeInTheDocument()
  })
})

describe('TicketList', () => {
  const base = {
    tickets: SAMPLE_TICKETS.slice(0, 3),
    scores: {
      'charged-twice': { probability: 0.83, alert: true },
      'locked-out': { probability: 0.1, alert: false },
    },
    scoreState: 'ready' as const,
    selected: 'charged-twice',
    onSelect: vi.fn(),
    onEnter: vi.fn(),
    onRetry: vi.fn(),
  }

  it('moves with the arrow keys and opens with Enter', async () => {
    const onSelect = vi.fn()
    const onEnter = vi.fn()
    render(<TicketList {...base} onSelect={onSelect} onEnter={onEnter} />)

    screen.getByRole('button', { name: /Charged twice/ }).focus()
    await userEvent.keyboard('{ArrowDown}')
    expect(onSelect).toHaveBeenCalledWith('locked-out')
    await userEvent.keyboard('{Enter}')
    expect(onEnter).toHaveBeenCalled()
  })

  it('filters by text and by flag, and offers a way back from nothing', async () => {
    render(<TicketList {...base} />)

    await userEvent.click(screen.getByRole('button', { name: 'Flagged only' }))
    expect(screen.queryByText('Whole team locked out')).not.toBeInTheDocument()
    await userEvent.type(screen.getByRole('searchbox'), 'zebra')
    expect(screen.getByText('No ticket mentions "zebra".')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Show all tickets' }))
    expect(screen.getByText('Whole team locked out')).toBeInTheDocument()
  })

  it('offers to reload the risk order when it failed', async () => {
    const onRetry = vi.fn()
    render(<TicketList {...base} scoreState="error" onRetry={onRetry} />)

    await userEvent.click(screen.getByRole('button', { name: 'Try again' }))
    expect(onRetry).toHaveBeenCalled()
  })
})

describe('Composer', () => {
  it('asks for a message and then clears the error as it is written', async () => {
    const onCreate = vi.fn()
    const onChange = vi.fn()
    const { rerender } = render(<Composer draft={EMPTY_DRAFT} onChange={onChange} onCreate={onCreate} />)

    await userEvent.click(screen.getByRole('button', { name: 'Triage ticket' }))
    expect(screen.getByRole('alert')).toHaveTextContent("Add the customer's message")
    expect(screen.getByLabelText('Customer message')).toHaveFocus()
    expect(onCreate).not.toHaveBeenCalled()

    await userEvent.type(screen.getByLabelText('Customer message'), 'H')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()

    rerender(<Composer draft={{ ...EMPTY_DRAFT, text: 'x'.repeat(4010) }} onChange={onChange} onCreate={onCreate} />)
    await userEvent.click(screen.getByRole('button', { name: 'Triage ticket' }))
    expect(screen.getByRole('alert')).toHaveTextContent('Shorten the message by 10 characters.')
  })

  it('names a ticket after its first words when there is no subject', async () => {
    const onCreate = vi.fn()
    render(<Composer draft={{ ...EMPTY_DRAFT, text: 'Where is my parcel please tell me now' }} onChange={vi.fn()} onCreate={onCreate} />)

    await userEvent.click(screen.getByRole('button', { name: 'Triage ticket' }))
    expect(onCreate.mock.calls[0][0]).toMatchObject({ subject: 'Where is my parcel please tell' })
  })
})
