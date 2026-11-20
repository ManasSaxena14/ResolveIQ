import { useEffect, useState } from 'react'
import { getHealth } from './api/client'
import type { HealthResponse } from './api/types'
import { Wordmark } from './components/Logo'
import { AnalyzePage } from './pages/AnalyzePage'
import { BatchPage } from './pages/BatchPage'
import { MonitoringPage } from './pages/MonitoringPage'

const PAGES = {
  analyze: { label: 'Analyze', index: '01', component: AnalyzePage },
  batch: { label: 'Batch', index: '02', component: BatchPage },
  monitoring: { label: 'Monitoring', index: '03', component: MonitoringPage },
} as const

type PageKey = keyof typeof PAGES
type Theme = 'light' | 'dark'
const PAGE_KEYS = Object.keys(PAGES) as PageKey[]

function pageFromHash(): PageKey {
  const key = window.location.hash.replace('#/', '')
  return key in PAGES ? (key as PageKey) : 'analyze'
}

function readTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem('resolveiq-theme')
    return stored === 'light' || stored === 'dark' ? stored : null
  } catch {
    return null
  }
}

const WAKING_AFTER_MS = 2500

/**
 * Polls /health. On free hosting the API sleeps when idle and takes up to a minute to
 * wake, so a check that has not answered after a moment is reported as "waking".
 */
function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [down, setDown] = useState(false)
  const [waking, setWaking] = useState(false)

  useEffect(() => {
    let active = true
    let timer = 0
    const schedule = (ms: number) => {
      timer = window.setTimeout(() => void check(), ms)
    }
    const check = async () => {
      const slow = window.setTimeout(() => active && setWaking(true), WAKING_AFTER_MS)
      try {
        const data = await getHealth()
        if (!active) return
        setHealth(data)
        setDown(false)
        setWaking(false)
        schedule(30_000)
      } catch {
        if (!active) return
        setDown(true)
        schedule(5_000)
      } finally {
        window.clearTimeout(slow)
      }
    }
    void check()
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [])

  return { health, down, waking }
}

function Status({
  health,
  down,
  waking,
}: {
  health: HealthResponse | null
  down: boolean
  waking: boolean
}) {
  if (waking && !health) {
    return (
      <span className="status degraded">
        <span className="light" />
        <span className="status-label">Waking up</span>
      </span>
    )
  }
  if (down) {
    return (
      <span className="status down">
        <span className="light" />
        <span className="status-label">API offline</span>
      </span>
    )
  }
  if (!health) {
    return (
      <span className="status">
        <span className="light" />
        <span className="status-label">Connecting</span>
      </span>
    )
  }
  const missing = Object.entries(health.components)
    .filter(([, ok]) => !ok)
    .map(([name]) => name.replace('_', ' '))
  const title = Object.entries(health.versions)
    .map(([k, v]) => `${k}: ${v}`)
    .join('\n')
  return (
    <span className={`status ${health.status}`} title={title}>
      <span className="light" />
      <span className="status-label">
        {health.status === 'ok' ? 'All systems ready' : `Degraded: ${missing.join(', ')}`}
      </span>
    </span>
  )
}

function ThemeButton({ dark, onToggle }: { dark: boolean; onToggle: () => void }) {
  return (
    <button
      type="button"
      className="icon-btn"
      onClick={onToggle}
      aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`}
      title={dark ? 'Paper theme' : 'Night theme'}
    >
      {dark ? (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
      )}
    </button>
  )
}

function Brand() {
  return (
    <a className="brand" href="#/analyze" aria-label="ResolveIQ home">
      <Wordmark />
    </a>
  )
}

export default function App() {
  const [page, setPage] = useState<PageKey>(pageFromHash)
  const [theme, setTheme] = useState<Theme | null>(readTheme)
  const { health, down, waking } = useHealth()

  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
  }, [theme])

  useEffect(() => {
    document.title = `${PAGES[page].label} | ResolveIQ`
  }, [page])

  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
  const dark = (theme ?? (prefersDark ? 'dark' : 'light')) === 'dark'

  const toggleTheme = () => {
    const next: Theme = dark ? 'light' : 'dark'
    setTheme(next)
    try {
      window.localStorage.setItem('resolveiq-theme', next)
    } catch {
      // Storage can be unavailable (private windows); the toggle still works for this visit.
    }
  }

  const go = (key: PageKey) => {
    window.history.pushState(null, '', `#/${key}`)
    setPage(key)
    window.scrollTo({ top: 0 })
  }

  const navItems = (showIndex: boolean) =>
    PAGE_KEYS.map((key) => (
      <button
        key={key}
        type="button"
        className="nav-item"
        aria-current={page === key ? 'page' : undefined}
        onClick={() => go(key)}
      >
        {showIndex && <span className="index">{PAGES[key].index}</span>}
        {PAGES[key].label}
      </button>
    ))

  const Page = PAGES[page].component

  return (
    <div className="shell">
      <aside className="rail">
        <Brand />
        <nav className="nav" aria-label="Main">
          {navItems(true)}
        </nav>
        <div className="rail-foot">
          <Status health={health} down={down} waking={waking} />
          <div className="rail-foot-row">
            <span className="eyebrow">v1.4</span>
            <ThemeButton dark={dark} onToggle={toggleTheme} />
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <Brand />
          <nav className="nav" aria-label="Main">
            {navItems(false)}
          </nav>
          <div className="topbar-end">
            <Status health={health} down={down} waking={waking} />
            <ThemeButton dark={dark} onToggle={toggleTheme} />
          </div>
        </header>

        {waking && !health && (
          <div className="wake-banner" role="status">
            <span className="wake-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            The free server is waking up after a quiet period. This takes up to a minute
            once; everything after that is fast. You can start typing now.
          </div>
        )}

        <main className="page" key={page}>
          <Page />
        </main>

        <footer className="footer">
          Demo on synthetic data. Do not enter personal data. Replies are drafts for a human agent and
          are never sent automatically.
        </footer>
      </div>

      <nav className="tabbar" aria-label="Main">
        {navItems(true)}
      </nav>
    </div>
  )
}
