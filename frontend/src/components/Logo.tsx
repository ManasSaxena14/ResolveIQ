import { useId } from 'react'

/**
 * The ResolveIQ mark: an R whose leg resolves into the lime verdict dot, set on a
 * ticket-stub tile with notched sides. The tile stays ink in both themes; on dark
 * backgrounds a hairline edge keeps it separate from the page.
 */
export function Logo({ size = 28, title }: { size?: number; title?: string }) {
  const id = useId()
  const stub = `${id}-stub`
  const sheen = `${id}-sheen`
  return (
    <svg
      className="brand-mark"
      width={size}
      height={size}
      viewBox="0 0 32 32"
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
    >
      <defs>
        <mask id={stub}>
          <rect width="32" height="32" fill="#fff" />
          <circle cx="1" cy="16" r="2.5" fill="#000" />
          <circle cx="31" cy="16" r="2.5" fill="#000" />
        </mask>
        <linearGradient id={sheen} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#2a2720" />
          <stop offset="1" stopColor="#14120d" />
        </linearGradient>
      </defs>
      <g mask={`url(#${stub})`}>
        <rect x="1" y="1" width="30" height="30" rx="8.5" fill={`url(#${sheen})`} />
        <rect
          x="1.4"
          y="1.4"
          width="29.2"
          height="29.2"
          rx="8.1"
          fill="none"
          stroke="#fff"
          strokeWidth="0.7"
          style={{ strokeOpacity: 'var(--logo-edge, 0)' }}
        />
      </g>
      <g transform="translate(-0.4 -0.5)">
        <path
          d="M10.8 8.9V23.1M10.8 8.9H16.2A4.35 4.35 0 0 1 16.2 17.6H10.8M15.3 17.6L18.2 20.5"
          fill="none"
          stroke="#f6f3ea"
          strokeWidth="3.05"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="21.55" cy="23.85" r="2.4" fill="#cdf24a" />
      </g>
    </svg>
  )
}

/** Mark and name together, as used in the navigation. */
export function Wordmark({ size = 28 }: { size?: number }) {
  return (
    <span className="wordmark">
      <Logo size={size} />
      <span className="wordmark-name">
        Resolve<em>IQ</em>
      </span>
    </span>
  )
}
