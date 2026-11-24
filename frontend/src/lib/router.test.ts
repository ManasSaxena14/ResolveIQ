import { describe, expect, it } from 'vitest'
import { SAMPLE_TICKETS, ticketMeta } from './inbox'
import { href, parseHash } from './router'

describe('router', () => {
  it('reads pages and items, falling back to triage', () => {
    expect(parseHash('')).toEqual({ page: 'triage', item: null })
    expect(parseHash('#/batch')).toEqual({ page: 'batch', item: null })
    expect(parseHash('#/triage/charged-twice')).toEqual({ page: 'triage', item: 'charged-twice' })
    expect(parseHash('#/nowhere/x')).toEqual({ page: 'triage', item: null })
    expect(href('triage', 'a b')).toBe('#/triage/a%20b')
  })
})

describe('inbox', () => {
  it('has unique sample tickets within the limit', () => {
    expect(new Set(SAMPLE_TICKETS.map((t) => t.id)).size).toBe(SAMPLE_TICKETS.length)
    expect(SAMPLE_TICKETS.every((t) => t.text.length < 4000)).toBe(true)
  })

  it('describes a ticket in plain words', () => {
    expect(ticketMeta(SAMPLE_TICKETS[0])).toBe('Email, problem, received 9:42')
  })
})
