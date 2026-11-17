// Mirrors src/resolveiq/schemas.py. Keep both in sync when the API changes.

export const TICKET_TYPES = ['Incident', 'Problem', 'Request', 'Change'] as const
export const CHANNELS = ['email', 'chat', 'phone', 'web'] as const

export type TicketType = (typeof TICKET_TYPES)[number]
export type DraftStatus = 'ok' | 'refused' | 'unavailable'
export type Rating = 'up' | 'down'

export interface AnalyzeRequest {
  text: string
  subject?: string
  ticket_type?: TicketType | ''
  channel?: string
  product?: string
}

export interface Source {
  file: string
  page: number
  score: number | null
}

export interface DraftReply {
  text: string | null
  status: DraftStatus
  refused: boolean
  sources: Source[]
  error: string | null
}

export interface CategoryScore {
  label: string
  probability: number
}

export interface Category {
  label: string
  confidence: number
  needs_review: boolean
  top: CategoryScore[]
}

export interface Risk {
  escalation_probability: number
  alert: boolean
  threshold: number
  reasons: string[]
}

export interface AnalyzeResponse {
  request_id: string
  draft_reply: DraftReply
  category: Category
  risk: Risk
  timings_ms: Record<string, number>
  versions: Record<string, string>
}

export interface BatchItem {
  rank: number
  id: string
  escalation_probability: number
  alert: boolean
  reasons: string[]
}

export interface BatchResponse {
  count: number
  alerts: number
  items: BatchItem[]
}

export interface FeedbackResponse {
  request_id: string
  rating: Rating
  stored: boolean
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  components: Record<string, boolean>
  versions: Record<string, string>
  index: Record<string, number>
  uptime_seconds: number
}

export interface RequestSummary {
  request_id: string
  created_at: string
  channel: string
  category: string
  confidence: number
  needs_review: boolean
  risk_score: number
  alert: boolean
  text_length: number
  draft_status: DraftStatus
  latency_ms: number
  rating: Rating | null
}

export interface StatsResponse {
  requests: number
  alerts: number
  needs_review: number
  mean_risk: number | null
  mean_confidence: number | null
  categories: Record<string, number>
  draft_status: Record<string, number>
  latency_median_ms: number | null
  latency_p95_ms: number | null
  feedback: { total: number; up: number; down: number; down_rate: number }
}
