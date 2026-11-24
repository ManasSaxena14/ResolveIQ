import type { AnalyzeResponse } from '../api/types'

export const ANALYZE_RESPONSE: AnalyzeResponse = {
  request_id: 'abc123def456',
  draft_reply: {
    text: 'The second charge is a pending authorization and disappears within three days [double-charges-and-pending-authorizations.md, page 1].',
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
  highlights: {
    category: [
      { term: 'charged', weight: 1 },
      { term: 'refund', weight: 0.6 },
    ],
    risk: [
      { term: 'twice', weight: 0.9 },
      { term: 'thanks', weight: -0.4 },
    ],
  },
}

/** A Response whose body streams server-sent events, as /analyze/stream does. */
export function eventStreamResponse(events: [string, unknown][]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const [event, data] of events) {
        // Split each event across two chunks to exercise buffering.
        const text = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
        const middle = Math.floor(text.length / 2)
        controller.enqueue(encoder.encode(text.slice(0, middle)))
        controller.enqueue(encoder.encode(text.slice(middle)))
      }
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

export function streamOf(response: AnalyzeResponse, partial = false): [string, unknown][] {
  const words = (response.draft_reply.text ?? '').split(' ')
  return [
    [
      'analysis',
      {
        request_id: response.request_id,
        category: response.category,
        risk: response.risk,
        highlights: response.highlights,
        versions: response.versions,
      },
    ],
    ['sources', response.draft_reply.sources],
    ...words.map((word, i): [string, unknown] => ['token', { text: i ? ` ${word}` : word }]),
    ['done', { response, partial }],
  ]
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

export const HEALTH = {
  status: 'ok',
  components: { retriever: true, classifier: true },
  versions: { risk: 'xgboost' },
  index: { files: 31, chunks: 31 },
  uptime_seconds: 1,
}

type Reply = Response | Promise<Response>

/** A fetch that answers by path, so a whole page can run against a stand-in API. */
export function apiFetch(routes: Record<string, () => Reply>) {
  const calls: string[] = []
  const handler = (input: RequestInfo | URL): Promise<Response> => {
    const path = String(input).replace(/^https?:\/\/[^/]+/, '').split('?')[0]
    calls.push(path)
    const route = routes[path]
    return route ? Promise.resolve(route()) : Promise.resolve(jsonResponse({ detail: 'Not found' }, 404))
  }
  return { handler, calls }
}
