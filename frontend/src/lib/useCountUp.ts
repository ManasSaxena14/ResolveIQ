import { useEffect, useState } from 'react'

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
}

/** Animate a number from 0 to `target` once; returns the target directly under reduced motion. */
export function useCountUp(target: number, duration = 650): number {
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0))

  useEffect(() => {
    if (prefersReducedMotion() || typeof requestAnimationFrame !== 'function') return
    let frame = 0
    const start = performance.now()
    const tick = (now: number) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      setValue(target * eased)
      if (progress < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target, duration])

  return value
}
