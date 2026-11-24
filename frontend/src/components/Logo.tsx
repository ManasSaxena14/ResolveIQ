import { useId } from 'react'

/**
 * The ResolveIQ mark: an R whose leg resolves into a dot, on a ticket-stub tile with
 * notched sides. A ticket comes in, a decision comes out.
 */
export function Logo({ size = 32 }: { size?: number }) {
  const stub = `${useId()}-stub`
  return (
    <svg className="logo" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <mask id={stub}>
        <rect width="32" height="32" fill="#fff" />
        <circle cx="1" cy="16" r="2.5" fill="#000" />
        <circle cx="31" cy="16" r="2.5" fill="#000" />
      </mask>
      <rect className="logo-tile" x="1" y="1" width="30" height="30" rx="7" mask={`url(#${stub})`} />
      <g transform="translate(-0.4 -0.5)">
        <path
          className="logo-letter"
          d="M10.8 8.9V23.1M10.8 8.9H16.2A4.35 4.35 0 0 1 16.2 17.6H10.8M15.3 17.6L18.2 20.5"
          fill="none"
          strokeWidth="3.05"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle className="logo-dot" cx="21.55" cy="23.85" r="2.4" />
      </g>
    </svg>
  )
}
