import type { AnalyzeResponse } from '../api/types'

export const ANALYZE_RESPONSE: AnalyzeResponse = {
  request_id: 'abc123def456',
  draft_reply: {
    text: 'The second charge is a pending authorization and disappears within three days.',
    status: 'ok',
    refused: false,
    sources: [
      { file: 'billing/double-charges-and-pending-authorizations.md', page: 1, score: 0.68 },
    ],
    error: null,
  },
  category: {
    label: 'Billing and Payments',
    confidence: 0.94,
    needs_review: false,
    top: [
      { label: 'Billing and Payments', probability: 0.94 },
      { label: 'Customer Service', probability: 0.03 },
    ],
  },
  risk: {
    escalation_probability: 0.82,
    alert: true,
    threshold: 0.23,
    reasons: ['Wording typical of high-priority tickets: "twice", "charge"'],
  },
  timings_ms: { retrieval: 14, generation: 820, classifier: 25, risk: 40, total: 905 },
  versions: { embedder: 'bge', llm: 'groq', classifier: 'distilbert', risk: 'xgboost' },
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}
