import { useState, type FormEvent } from 'react'
import { CHANNELS, TICKET_TYPES, type TicketType } from '../api/types'
import { MAX_CHARS, type ComposerDraft, type Ticket } from '../lib/inbox'

interface Props {
  draft: ComposerDraft
  onChange: (draft: ComposerDraft) => void
  onCreate: (ticket: Ticket) => void
}

export function Composer({ draft, onChange, onCreate }: Props) {
  const [error, setError] = useState('')
  const set = <K extends keyof ComposerDraft>(key: K, value: ComposerDraft[K]) => {
    onChange({ ...draft, [key]: value })
    if (key === 'text') setError('')
  }

  const check = (): string => {
    if (!draft.text.trim()) return "Add the customer's message to triage it."
    if (draft.text.length > MAX_CHARS)
      return `Shorten the message by ${(draft.text.length - MAX_CHARS).toLocaleString()} characters.`
    return ''
  }

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const problem = check()
    setError(problem)
    if (problem) {
      document.getElementById('message')?.focus()
      return
    }
    const now = new Date()
    onCreate({
      ...draft,
      subject: draft.subject.trim() || draft.text.trim().split(/\s+/).slice(0, 6).join(' '),
      id: `own-${now.getTime()}`,
      received: `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`,
    })
  }

  return (
    <form className="composer" onSubmit={submit} noValidate aria-labelledby="detail-title">
      <h2 id="detail-title" tabIndex={-1}>
        New ticket
      </h2>
      <p className="muted">
        Use invented data. ResolveIQ keeps only the length of the message and a hash, but a hosted language model may
        read the text to write the reply.
      </p>

      <label htmlFor="subject">Subject (optional)</label>
      <input id="subject" maxLength={300} value={draft.subject} onChange={(e) => set('subject', e.target.value)} />

      <label htmlFor="message">Customer message</label>
      <textarea
        id="message"
        rows={8}
        value={draft.text}
        aria-invalid={!!error || undefined}
        aria-describedby={error ? 'message-error' : 'message-count'}
        onChange={(e) => set('text', e.target.value)}
        onBlur={() => setError(draft.text ? check() : '')}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit(e)
        }}
      />
      {error ? (
        <p id="message-error" className="field-error" role="alert">
          {error}
        </p>
      ) : (
        <p id="message-count" className="meta muted">
          {draft.text.length.toLocaleString()} of {MAX_CHARS.toLocaleString()} characters
        </p>
      )}

      <div className="field-row">
        <div>
          <label htmlFor="type">Type (optional)</label>
          <select id="type" value={draft.ticket_type} onChange={(e) => set('ticket_type', e.target.value as TicketType | '')}>
            <option value="">Not known</option>
            {TICKET_TYPES.map((t) => (
              <option key={t}>{t}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="channel">Channel (optional)</label>
          <select id="channel" value={draft.channel} onChange={(e) => set('channel', e.target.value)}>
            <option value="">Not known</option>
            {CHANNELS.map((c) => (
              <option key={c} value={c}>
                {c.charAt(0).toUpperCase() + c.slice(1)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="product">Product (optional)</label>
          <input id="product" maxLength={100} value={draft.product} onChange={(e) => set('product', e.target.value)} />
        </div>
      </div>
      <button type="submit" className="button primary">
        Triage ticket
      </button>
    </form>
  )
}
