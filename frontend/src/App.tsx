import { useEffect, useState } from 'react'
import { getHealth } from './api/client'
import type { HealthResponse } from './api/types'
import { AnalyzePage } from './pages/AnalyzePage'
import { BatchPage } from './pages/BatchPage'
import { MonitoringPage } from './pages/MonitoringPage'

const PAGES = {
  analyze: { label: 'Analyze', component: AnalyzePage },
  batch: { label: 'Batch', component: BatchPage },
  monitoring: { label: 'Monitoring', component: MonitoringPage },
} as const

type PageKey = keyof typeof PAGES
type Theme = 'light' | 'dark'

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

function HealthPill() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [down, setDown] = useState(false)

  useEffect(() => {
    let active = true
    const check = async () => {
      try {
        const data = await getHealth()
        if (active) {
          setHealth(data)
          setDown(false)
        }
      } catch {
        if (active) setDown(true)
      }
    }
    void check()
    const timer = window.setInterval(() => void check(), 30_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])

  if (down) return <span className="status-pill down">API offline</span>
  if (!health) return <span className="status-pill">Connecting</span>
  const missing = Object.entries(health.components)
    .filter(([, ok]) => !ok)
    .map(([name]) => name.replace('_', ' '))
  const title = Object.entries(health.versions)
    .map(([k, v]) => `${k}: ${v}`)
    .join('\n')
  return (
    <span className={`status-pill ${health.status}`} title={title}>
      {health.status === 'ok' ? 'All systems ready' : `Degraded: ${missing.join(', ')}`}
    </span>
  )
}

function ThemeIcon({ dark }: { dark: boolean }) {
  return dark ? (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  ) : (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
    </svg>
  )
}

export default function App() {
  const [page, setPage] = useState<PageKey>(pageFromHash)
  const [theme, setTheme] = useState<Theme | null>(readTheme)

  useEffect(() => {
    const onHash = () => setPage(pageFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
  }, [theme])

  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
  const effectiveTheme: Theme = theme ?? (prefersDark ? 'dark' : 'light')

  const toggleTheme = () => {
    const next: Theme = effectiveTheme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    try {
      window.localStorage.setItem('resolveiq-theme', next)
    } catch {
      // Storage can be unavailable (private windows); the toggle still works for this visit.
    }
  }

  const Page = PAGES[page].component

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              RQ
            </span>
            ResolveIQ
          </div>
          <nav className="nav" aria-label="Main">
            {(Object.keys(PAGES) as PageKey[]).map((key) => (
              <button
                key={key}
                type="button"
                aria-current={page === key ? 'page' : undefined}
                onClick={() => {
                  window.location.hash = `/${key}`
                  setPage(key)
                }}
              >
                {PAGES[key].label}
              </button>
            ))}
          </nav>
          <div className="topbar-end">
            <HealthPill />
            <button
              type="button"
              className="icon-btn"
              onClick={toggleTheme}
              aria-label={`Switch to ${effectiveTheme === 'dark' ? 'light' : 'dark'} theme`}
              title="Toggle theme"
            >
              <ThemeIcon dark={effectiveTheme === 'dark'} />
            </button>
          </div>
        </div>
      </header>

      <main className="page">
        <Page />
      </main>

      <footer className="footer">
        <div className="page">
          Demo on synthetic data. Do not enter personal data. Replies are drafts for a human agent
          and are never sent automatically.
        </div>
      </footer>
    </div>
  )
}
