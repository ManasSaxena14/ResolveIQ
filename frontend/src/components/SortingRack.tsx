import type { Category } from '../api/types'
import { QUEUES } from '../lib/inbox'
import { percent, share } from '../lib/format'

interface Props {
  category: Category | null
  subject: string
}

/**
 * The sorting rack: one pigeonhole per queue. The ticket's slip lies in the queue it
 * belongs to; every other slot fills in proportion to its probability.
 */
export function SortingRack({ category, subject }: Props) {
  const scores = new Map(category?.top.map((t) => [t.label, t.probability]))
  const known = new Set<string>(QUEUES)
  const labels: string[] = [...QUEUES, ...(category?.top.map((t) => t.label).filter((l) => !known.has(l)) ?? [])]
  const summary = category
    ? category.top
        .slice(0, 3)
        .map((t) => `${t.label} ${percent(t.probability)}`)
        .join(', ')
    : 'Not sorted yet'

  return (
    <figure className="rack" aria-label="Queues">
      <figcaption className="visually-hidden">{summary}</figcaption>
      <ol className="rack-grid" aria-hidden="true">
        {labels.map((label) => {
          const p = scores.get(label) ?? 0
          const chosen = category?.label === label
          return (
            <li key={label} className="slot" data-chosen={chosen || undefined}>
              <div className="slot-bay">
                {chosen ? (
                  <div className="slip" key={subject}>
                    <span>{subject || 'Ticket'}</span>
                  </div>
                ) : (
                  <div className="slot-fill" style={{ transform: `scaleY(${p})` }} />
                )}
              </div>
              <div className="slot-plate">
                <span className="slot-name">{label}</span>
                <span className="slot-share">{category ? share(p) : ''}</span>
              </div>
            </li>
          )
        })}
      </ol>
    </figure>
  )
}
