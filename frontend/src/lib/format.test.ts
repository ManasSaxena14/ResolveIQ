import { describe, expect, it } from 'vitest'
import {
  milliseconds,
  percent,
  relativeTime,
  riskLevel,
  sentence,
  sourceSection,
  sourceTitle,
  sourceUrl,
  toCsv,
} from './format'

describe('format helpers', () => {
  it('formats percentages and missing values', () => {
    expect(percent(0.823)).toBe('82%')
    expect(percent(0.823, 1)).toBe('82.3%')
    expect(percent(null)).toBe('-')
  })

  it('formats durations', () => {
    expect(milliseconds(42.4)).toBe('42 ms')
    expect(milliseconds(1520)).toBe('1.5 s')
    expect(milliseconds(undefined)).toBe('-')
  })

  it('turns knowledge base paths into titles and links', () => {
    const file = 'billing/double-charges-and-pending-authorizations.md'
    expect(sourceTitle(file)).toBe('Double charges and pending authorizations')
    expect(sourceSection(file)).toBe('billing')
    expect(sourceSection('faq.md')).toBe('')
    expect(sourceUrl('returns/refund timelines.md')).toMatch(/returns\/refund%20timelines\.md$/)
  })

  it('places risk in three levels around the threshold', () => {
    expect(riskLevel(0.1, 0.23)).toBe('low')
    expect(riskLevel(0.4, 0.23)).toBe('medium')
    expect(riskLevel(0.82, 0.23)).toBe('high')
  })

  it('ends sentences once', () => {
    expect(sentence('The model is down')).toBe('The model is down.')
    expect(sentence('Done. ')).toBe('Done.')
  })

  it('describes recent times', () => {
    const now = new Date('2026-11-17T12:00:00Z')
    expect(relativeTime('2026-11-17T11:59:50Z', now)).toBe('just now')
    expect(relativeTime('2026-11-17T11:30:00Z', now)).toBe('30 min ago')
    expect(relativeTime('2026-11-17T09:00:00Z', now)).toBe('3 h ago')
  })

  it('quotes CSV cells that need it', () => {
    expect(toCsv(['a', 'b'], [['x, y', 'say "hi"']])).toBe('a,b\n"x, y","say ""hi"""\n')
  })
})
