import type { Source, WordWeight } from '../api/types'

export interface Segment {
  text: string
  /** Weight of the matched term, or null for plain text. */
  weight: number | null
  term?: string
}

const WORD = /[\p{L}\p{N}][\p{L}\p{N}'-]*/gu

/**
 * Split `text` into plain and highlighted segments. Terms match whole words without
 * regard to case; two-word terms (from the risk text model) match adjacent words.
 * When both a phrase and one of its words match, the phrase wins.
 */
export function highlightSegments(text: string, terms: WordWeight[]): Segment[] {
  if (!text || !terms.length) return text ? [{ text, weight: null }] : []
  const weights = new Map(terms.map((t) => [t.term.toLowerCase(), t.weight]))
  const words = [...text.matchAll(WORD)].map((m) => ({
    start: m.index ?? 0,
    end: (m.index ?? 0) + m[0].length,
    word: m[0].toLowerCase(),
  }))

  const spans: { start: number; end: number; weight: number; term: string }[] = []
  for (let i = 0; i < words.length; i += 1) {
    const next = words[i + 1]
    const phrase = next ? `${words[i].word} ${next.word}` : ''
    if (phrase && weights.has(phrase) && /^\s+$/.test(text.slice(words[i].end, next.start))) {
      spans.push({ start: words[i].start, end: next.end, weight: weights.get(phrase)!, term: phrase })
      i += 1
    } else if (weights.has(words[i].word)) {
      spans.push({ ...words[i], weight: weights.get(words[i].word)!, term: words[i].word })
    }
  }

  const segments: Segment[] = []
  let cursor = 0
  for (const span of spans) {
    if (span.start > cursor) segments.push({ text: text.slice(cursor, span.start), weight: null })
    segments.push({ text: text.slice(span.start, span.end), weight: span.weight, term: span.term })
    cursor = span.end
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor), weight: null })
  return segments
}

export type DraftPart = { text: string } | { citation: string; index: number | null }

const CITATION = /\[([^[\]]+?),\s*page\s*(\d+)\s*\]/gi

function basename(path: string): string {
  return path.trim().toLowerCase().split('/').pop() ?? ''
}

/**
 * Split a draft into text and citation chips. Each citation is matched to the
 * position of its source in `sources` (1-based) by file name and page, or null when
 * it points at no retrieved source.
 */
export function draftParts(text: string, sources: Source[]): DraftPart[] {
  const parts: DraftPart[] = []
  let cursor = 0
  for (const match of text.matchAll(CITATION)) {
    const start = match.index ?? 0
    if (start > cursor) parts.push({ text: text.slice(cursor, start) })
    const file = basename(match[1])
    const page = Number(match[2])
    const found = sources.findIndex((s) => basename(s.file) === file && s.page === page)
    parts.push({ citation: `${match[1].trim()}, page ${page}`, index: found === -1 ? null : found + 1 })
    cursor = start + match[0].length
  }
  if (cursor < text.length) parts.push({ text: text.slice(cursor) })
  return parts
}
