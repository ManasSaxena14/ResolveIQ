import { useEffect, useState } from 'react'

export const PAGES = ['triage', 'batch', 'monitoring', 'about'] as const
export type PageKey = (typeof PAGES)[number]

export interface Route {
  page: PageKey
  /** Ticket id on the triage page, "new" for the composer, or null for the list. */
  item: string | null
}

export function parseHash(hash: string): Route {
  const [page, item] = hash.replace(/^#\/?/, '').split('/')
  const known = (PAGES as readonly string[]).includes(page)
  return { page: known ? (page as PageKey) : 'triage', item: known && item ? decodeURIComponent(item) : null }
}

export function href(page: PageKey, item?: string | null): string {
  return `#/${page}${item ? `/${encodeURIComponent(item)}` : ''}`
}

export function navigate(page: PageKey, item?: string | null, replace = false) {
  const url = href(page, item)
  if (replace) window.history.replaceState(null, '', url)
  else window.history.pushState(null, '', url)
  window.dispatchEvent(new HashChangeEvent('hashchange'))
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(window.location.hash))
  useEffect(() => {
    const update = () => setRoute(parseHash(window.location.hash))
    window.addEventListener('hashchange', update)
    window.addEventListener('popstate', update)
    return () => {
      window.removeEventListener('hashchange', update)
      window.removeEventListener('popstate', update)
    }
  }, [])
  return route
}
