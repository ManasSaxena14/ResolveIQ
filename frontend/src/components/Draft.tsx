import { useState } from 'react'
import type { DraftReply, Grounding, Source } from '../api/types'
import { draftParts } from '../lib/highlight'
import { sentence, sourceSection, sourceTitle, sourceUrl } from '../lib/format'

function Body({ text, sources, writing }: { text: string; sources: Source[]; writing: boolean }) {
  return (
    <p className="draft-text">
      {draftParts(text, sources).map((part, i) =>
        'citation' in part ? (
          part.index === null ? (
            <span key={i} className="cite cite-unknown" title={`Not a retrieved article: ${part.citation}`}>
              ?
            </span>
          ) : (
            <button
              key={i}
              type="button"
              className="cite"
              title={part.citation}
              aria-label={`Article ${part.index}: ${part.citation}`}
              onClick={() => document.getElementById(`article-${part.index}`)?.focus()}
            >
              {part.index}
            </button>
          )
        ) : (
          <span key={i}>{part.text}</span>
        ),
      )}
      {writing && <span className="caret" aria-hidden="true" />}
    </p>
  )
}

function groundingNote(g: Grounding): { text: string; warn: boolean } {
  if (g.invalid_citations.length)
    return { text: `Check before sending: ${g.invalid_citations.join('; ')} was not among the articles found.`, warn: true }
  if (!g.citations) return { text: 'No article is cited. Check every fact before sending.', warn: true }
  return {
    text: `${g.cited_sentences} of ${g.sentences} sentences cite an article, and every citation is real.`,
    warn: g.coverage < 0.5,
  }
}

interface Props {
  draft: DraftReply | null
  liveText: string
  liveSources: Source[]
}

/** The suggested reply and the help articles it rests on. */
export function Draft({ draft, liveText, liveSources }: Props) {
  const [copied, setCopied] = useState(false)
  const writing = !draft
  const sources = draft?.sources ?? liveSources
  const note = draft?.grounding ? groundingNote(draft.grounding) : null

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(draft?.text ?? '')
      setCopied(true)
      window.setTimeout(() => setCopied(false), 2000)
    } catch {
      setCopied(false)
    }
  }

  return (
    <section aria-labelledby="reply-title">
      <div className="section-head">
        <h3 id="reply-title">Suggested reply</h3>
        {draft?.status === 'ok' && (
          <button type="button" className="button primary" onClick={copy}>
            {copied ? 'Copied' : 'Copy reply'}
          </button>
        )}
      </div>

      {writing && (liveText ? <Body text={liveText} sources={sources} writing /> : <p className="muted">Writing a reply.</p>)}
      {draft?.status === 'ok' && <Body text={draft.text ?? ''} sources={sources} writing={false} />}
      {note && (
        <p className={note.warn ? 'secondary note-warn' : 'secondary muted'} role={note.warn ? 'note' : undefined}>
          {note.text}
        </p>
      )}
      {draft?.status === 'refused' && (
        <p>The help center has no article on this, so no reply was written. Answer it yourself or pass it to the queue above.</p>
      )}
      {draft?.status === 'unavailable' && (
        <p>
          {sentence(draft.error ?? 'The language model is unavailable')} The queue and risk above still hold.
        </p>
      )}

      {sources.length > 0 && (
        <>
          <h4>Help articles found</h4>
          <ol className="articles">
            {sources.map((source, i) => (
              <li key={`${source.file}-${source.page}`}>
                <a id={`article-${i + 1}`} href={sourceUrl(source.file)} target="_blank" rel="noreferrer">
                  <span className="article-number">{i + 1}</span>
                  <span className="article-title">{sourceTitle(source.file)}</span>
                  <span className="meta muted">
                    {sourceSection(source.file) || 'help center'}
                    {source.score !== null && `, ${Math.round(source.score * 100)}% match`}
                  </span>
                </a>
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  )
}
