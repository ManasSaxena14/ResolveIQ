import { describe, expect, it, vi } from 'vitest'
import { ANALYZE_RESPONSE, jsonResponse } from '../test/fixtures'
import { ApiError, analyzeTicket, getStats, sendFeedback } from './client'

describe('API client', () => {
  it('returns the analysis and marks a 503 as partial', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(ANALYZE_RESPONSE))
      .mockResolvedValueOnce(jsonResponse(ANALYZE_RESPONSE, 503))

    const ok = await analyzeTicket({ text: 'Charged twice' })
    const partial = await analyzeTicket({ text: 'Charged twice' })

    expect(ok.partial).toBe(false)
    expect(partial.partial).toBe(true)
    expect(partial.data.risk.alert).toBe(true)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/analyze')
    expect(JSON.parse(String(init?.body))).toEqual({ text: 'Charged twice' })
  })

  it('turns validation errors into readable messages', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ detail: [{ msg: 'Value error, Ticket text must not be empty' }] }, 422),
    )

    await expect(analyzeTicket({ text: ' ' })).rejects.toThrow('Ticket text must not be empty')
  })

  it('reports an unreachable server', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Failed to fetch'))

    const error = await getStats().catch((err: unknown) => err)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(0)
  })

  it('surfaces 404 for unknown feedback requests', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      jsonResponse({ detail: 'Unknown request ID nope' }, 404),
    )

    await expect(sendFeedback('nope', 'up')).rejects.toMatchObject({
      status: 404,
      message: 'Unknown request ID nope',
    })
  })
})
