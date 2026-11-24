import { useEffect, useRef, useState } from 'react'
import { getHealth } from './api/client'
import type { HealthResponse } from './api/types'
import { Logo } from './components/Logo'
import { href, useRoute, type PageKey } from './lib/router'
import { useTriage } from './lib/useTriage'
import { AboutPage } from './pages/AboutPage'
import { BatchPage } from './pages/BatchPage'
import { MonitoringPage } from './pages/MonitoringPage'
import { TriagePage } from './pages/TriagePage'

type Theme = 'light' | 'dark'

const NAV: { key: PageKey; label: string }[] = [
  { key: 'triage', label: 'Triage' },
  { key: 'batch', label: 'Batch' },
  { key: 'monitoring', label: 'Monitoring' },
  { key: 'about', label: 'About' },
]

const WAKING_AFTER_MS = 2500

function readTheme(): Theme | null {
  try {
    const stored = window.localStorage.getItem('resolveiq-theme')
    return stored === 'light' || stored === 'dark' ? stored : null
  } catch {
    return null
  }
}

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

  return { health, down, waking: waking && !health }
}

function statusOf({ health, down, waking }: ReturnType<typeof useHealth>): { text: string; tone: string } {
  if (waking) return { text: 'Waking up', tone: 'wait' }
  if (down) return { text: 'Offline', tone: 'bad' }
  if (!health) return { text: 'Connecting', tone: 'wait' }
  return health.status === 'ok' ? { text: 'Ready', tone: 'good' } : { text: 'Partly ready', tone: 'wait' }
}

export default function App() {
  const route = useRoute()
  const server = useHealth()
  const triage = useTriage()
  const [theme, setTheme] = useState<Theme | null>(readTheme)
  const status = statusOf(server)

  const online = !!server.health && !server.down
  const wasOnline = useRef(online)
  const { recover } = triage
  useEffect(() => {
    if (online && !wasOnline.current) recover()
    wasOnline.current = online
  }, [online, recover])

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme
    else delete document.documentElement.dataset.theme
  }, [theme])

  useEffect(() => {
    document.title = `${NAV.find((n) => n.key === route.page)?.label} | ResolveIQ`
  }, [route.page])

  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false
  const dark = (theme ?? (prefersDark ? 'dark' : 'light')) === 'dark'
  const toggleTheme = () => {
    const next: Theme = dark ? 'light' : 'dark'
    setTheme(next)
    try {
      window.localStorage.setItem('resolveiq-theme', next)
    } catch {
      // Storage can be unavailable (private windows); the switch still works for this visit.
    }
  }

  const links = NAV.map(({ key, label }) => (
    <a key={key} href={href(key)} aria-current={route.page === key ? 'page' : undefined}>
      {label}
    </a>
  ))

  return (
    <div className="app">
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus() }}>
        Skip to content
      </a>
      <header className="topbar">
        <a className="brand" href={href('triage')}>
          <Logo />
          <span>ResolveIQ</span>
        </a>
        <nav className="nav" aria-label="Pages">
          {links}
        </nav>
        <span className="status" data-tone={status.tone}>
          {status.text}
        </span>
        <button
          type="button"
          className="theme"
          onClick={toggleTheme}
          aria-label={dark ? 'Use the day theme' : 'Use the night theme'}
          title={dark ? 'Day theme' : 'Night theme'}
        >
          <svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true">
            {dark ? (
              <>
                <circle cx="10" cy="10" r="3.5" />
                <path d="M10 2v2M10 16v2M2 10h2M16 10h2M4.3 4.3l1.4 1.4M14.3 14.3l1.4 1.4M4.3 15.7l1.4-1.4M14.3 5.7l1.4-1.4" />
              </>
            ) : (
              <path d="M16.5 11.6A7 7 0 1 1 8.4 3.5a5.5 5.5 0 0 0 8.1 8.1z" />
            )}
          </svg>
        </button>
      </header>

      {server.waking && (
        <div className="wake" role="status">
          The free server is waking up after a quiet spell. This takes up to a minute, then everything is fast.
        </div>
      )}

      <main id="main" tabIndex={-1}>
        {route.page === 'triage' && <TriagePage triage={triage} item={route.item} />}
        {route.page === 'batch' && <BatchPage />}
        {route.page === 'monitoring' && <MonitoringPage />}
        {route.page === 'about' && <AboutPage />}
      </main>

      <nav className="tabbar" aria-label="Pages">
        {links}
      </nav>
    </div>
  )
}
