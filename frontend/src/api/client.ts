import type {
  AnalysisEvent,
  AnalyzeRequest,
  AnalyzeResponse,
  BatchResponse,
  FeedbackResponse,
  HealthResponse,
  Rating,
  RequestSummary,
  Source,
  StatsResponse,
  StreamHandlers,
} from './types'

/** Error with the HTTP status and the API's `detail` message. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

// The API serves this app, so requests go to the same origin. In development
// Vite proxies the API paths to uvicorn (see vite.config.ts).
const BASE = import.meta.env.VITE_API_BASE ?? ''

function detailMessage(body: unknown, status: number): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const detail = (body as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((item) => (item && typeof item === 'object' && 'msg' in item ? String(item.msg) : ''))
        .filter(Boolean)
        .map((msg) => msg.replace(/^Value error, /, ''))
        .join('; ')
    }
  }
  return `Request failed with status ${status}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, init)
  } catch {
    throw new ApiError(0, 'The server cannot be reached. Check your connection and try again.')
  }
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(response.status, detailMessage(body, response.status))
  return body as T
}

export interface AnalyzeResult {
  data: AnalyzeResponse
  /** True when the language model was unavailable (HTTP 503 with a partial result). */
  partial: boolean
}

export async function analyzeTicket(payload: AnalyzeRequest): Promise<AnalyzeResult> {
  let response: Response
  try {
    response = await fetch(`${BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new ApiError(0, 'The server cannot be reached. Check your connection and try again.')
  }
  const body: unknown = await response.json().catch(() => null)
  // 503 still carries the category and risk; only the draft is missing.
  if (response.ok || (response.status === 503 && body && typeof body === 'object' && 'request_id' in body)) {
    return { data: body as AnalyzeResponse, partial: response.status === 503 }
  }
  throw new ApiError(response.status, detailMessage(body, response.status))
}

/** Parse one server-sent event block ("event: x\ndata: {...}"). */
export function parseEvent(block: string): { event: string; data: unknown } | null {
  let event = 'message'
  const data: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) data.push(line.slice(5).trim())
  }
  if (!data.length) return null
  try {
    return { event, data: JSON.parse(data.join('\n')) }
  } catch {
    return null
  }
}

/**
 * Analyze over /analyze/stream: category and risk arrive first, then the sources and
 * the draft piece by piece. Resolves with the complete response, like analyzeTicket.
 */
export async function analyzeTicketStream(
  payload: AnalyzeRequest,
  handlers: StreamHandlers = {},
): Promise<AnalyzeResult> {
  let response: Response
  try {
    response = await fetch(`${BASE}/analyze/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new ApiError(0, 'The server cannot be reached. Check your connection and try again.')
  }
  if (!response.ok || !response.body) {
    const body: unknown = await response.json().catch(() => null)
    throw new ApiError(response.status, detailMessage(body, response.status))
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  let final: AnalyzeResult | null = null
  const handle = (block: string) => {
    const parsed = parseEvent(block)
    if (!parsed) return
    const { event, data } = parsed
    if (event === 'analysis') handlers.onAnalysis?.(data as AnalysisEvent)
    else if (event === 'sources') handlers.onSources?.(data as Source[])
    else if (event === 'token') handlers.onToken?.((data as { text: string }).text)
    else if (event === 'done') {
      const done = data as { response: AnalyzeResult['data']; partial: boolean }
      final = { data: done.response, partial: done.partial }
    } else if (event === 'error') {
      throw new ApiError(500, (data as { detail?: string }).detail ?? 'The analysis failed.')
    }
  }
  for (;;) {
    const { value, done } = await reader.read()
    if (value) buffer += value.replace(/\r\n/g, '\n')
    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      handle(buffer.slice(0, boundary))
      buffer = buffer.slice(boundary + 2)
      boundary = buffer.indexOf('\n\n')
    }
    if (done) break
  }
  if (buffer.trim()) handle(buffer)
  if (!final) throw new ApiError(500, 'The analysis ended before it was complete.')
  return final
}

export function scoreBatch(file: File): Promise<BatchResponse> {
  const form = new FormData()
  form.append('file', file)
  return request<BatchResponse>('/batch', { method: 'POST', body: form })
}

export function sendFeedback(requestId: string, rating: Rating, comment = ''): Promise<FeedbackResponse> {
  return request<FeedbackResponse>('/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, rating, comment }),
  })
}

export const getHealth = () => request<HealthResponse>('/health')
export const getStats = () => request<StatsResponse>('/stats')
export const getRecentRequests = (limit = 50) =>
  request<RequestSummary[]>(`/requests?limit=${limit}`)
