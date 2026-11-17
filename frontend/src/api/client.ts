import type {
  AnalyzeRequest,
  AnalyzeResponse,
  BatchResponse,
  FeedbackResponse,
  HealthResponse,
  Rating,
  RequestSummary,
  StatsResponse,
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
    throw new ApiError(0, 'The ResolveIQ API cannot be reached. Is the server running?')
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
    throw new ApiError(0, 'The ResolveIQ API cannot be reached. Is the server running?')
  }
  const body: unknown = await response.json().catch(() => null)
  // 503 still carries the category and risk; only the draft is missing.
  if (response.ok || (response.status === 503 && body && typeof body === 'object' && 'request_id' in body)) {
    return { data: body as AnalyzeResponse, partial: response.status === 503 }
  }
  throw new ApiError(response.status, detailMessage(body, response.status))
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
