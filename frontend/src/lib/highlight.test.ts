import { describe, expect, it } from 'vitest'
import { draftParts, highlightSegments } from './highlight'

describe('highlightSegments', () => {
  it('marks whole words without regard to case', () => {
    const segments = highlightSegments('Refund my CHARGE. Recharge later.', [
      { term: 'charge', weight: 1 },
      { term: 'refund', weight: 0.5 },
    ])

    expect(segments.filter((s) => s.weight !== null).map((s) => s.text)).toEqual(['Refund', 'CHARGE'])
    expect(segments.map((s) => s.text).join('')).toBe('Refund my CHARGE. Recharge later.')
  })

  it('prefers a two-word phrase over its parts', () => {
    const segments = highlightSegments('the server is down', [
      { term: 'is down', weight: 0.8 },
      { term: 'down', weight: 0.2 },
    ])

    expect(segments.find((s) => s.weight !== null)).toMatchObject({ text: 'is down', weight: 0.8 })
  })

  it('handles empty input', () => {
    expect(highlightSegments('', [{ term: 'x', weight: 1 }])).toEqual([])
    expect(highlightSegments('plain', [])).toEqual([{ text: 'plain', weight: null }])
  })
})

describe('draftParts', () => {
  const sources = [{ file: 'billing/double-charges.md', page: 1, score: 0.7 }]

  it('turns citations into numbered references', () => {
    const parts = draftParts('A pending hold [double-charges.md, page 1]. Shipping [shipping.md, page 2].', sources)

    expect(parts).toEqual([
      { text: 'A pending hold ' },
      { citation: 'double-charges.md, page 1', index: 1 },
      { text: '. Shipping ' },
      { citation: 'shipping.md, page 2', index: null },
      { text: '.' },
    ])
  })
})
