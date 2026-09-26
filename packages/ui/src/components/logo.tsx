import { type ComponentProps, createUniqueId } from "solid-js"
import { geometry as g } from "./logo-geometry"

/**
 * The Olaya mark: a ring, the decision curve crossing it, and the operating point on the curve.
 * The gap around the point is a mask, so it stays a real gap on any background.
 */
const OlayaSymbol = (props: { ink: string; point: string; y?: number }) => {
  const mask = `olaya-point-${createUniqueId()}`
  const y = props.y ?? 0
  return (
    <g transform={`translate(0 ${y})`}>
      <mask id={mask} maskUnits="userSpaceOnUse" x="-10" y="-10" width="140" height="140">
        <rect x="-10" y="-10" width="140" height="140" fill="#fff" />
        <circle cx={g.point.cx} cy={g.point.cy} r={g.point.r + g.point.halo} fill="#000" />
      </mask>
      <g mask={`url(#${mask})`}>
        <circle cx={g.ring.cx} cy={g.ring.cy} r={g.ring.r} fill="none" stroke={props.ink} stroke-width={g.ring.width} />
        <path d={g.curve.d} fill="none" stroke={props.ink} stroke-width={g.curve.width} stroke-linecap="round" />
      </g>
      <circle cx={g.point.cx} cy={g.point.cy} r={g.point.r} fill={props.point} />
    </g>
  )
}

export const Mark = (props: { class?: string }) => {
  return (
    <svg
      data-component="logo-mark"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 120 120"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <OlayaSymbol ink="var(--icon-strong-base)" point={g.coral} />
    </svg>
  )
}

export const Splash = (props: Pick<ComponentProps<"svg">, "ref" | "class">) => {
  return (
    <svg
      ref={props.ref}
      data-component="logo-splash"
      classList={{ [props.class ?? ""]: !!props.class }}
      viewBox="0 0 120 120"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <OlayaSymbol ink="var(--icon-strong-base)" point={g.coral} />
    </svg>
  )
}

export const Logo = (props: { class?: string }) => {
  const w = g.wordmark
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox={`0 0 ${w.width} ${w.height}`}
      fill="none"
      role="img"
      aria-label="Olaya"
      classList={{ [props.class ?? ""]: !!props.class }}
    >
      <OlayaSymbol ink="var(--icon-base)" point={g.coral} y={w.symbolY} />
      <path transform={`translate(${w.x} ${w.y}) scale(${w.scale})`} fill="var(--icon-base)" d={w.d} />
    </svg>
  )
}
