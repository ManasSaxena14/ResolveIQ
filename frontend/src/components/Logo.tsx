/** The ResolveIQ mark: a ticket being read, with a signal dot for the verdict. */
export function Logo({ size = 26 }: { size?: number }) {
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="9" fill="var(--ink)" />
      <path d="M9 11h10M9 16h14M9 21h7" stroke="var(--bg)" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="23" cy="21" r="3.2" fill="var(--signal-fill)" />
    </svg>
  )
}
