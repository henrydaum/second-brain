import type { ComponentProps, CSSProperties } from "react";
import { cn } from "@/lib/utils";

const GRID = 5;
const CENTER = (GRID - 1) / 2;
const DOT_INDEXES = Array.from({ length: GRID * GRID }, (_, i) => i);

/* Deterministic bit-mixing hash so server and client render identical markup; takes a range in milliseconds and returns seconds. A plain (i * prime) % range correlates indexes a grid-stride apart and renders as column-synchronized waves instead of a twinkle. */
const hash = (n: number, salt: number, range: number) => {
  let h = (Math.imul(n, 374761393) + Math.imul(salt, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) % range) / 1000;
};

type Blink = { duration: number; delay: number; lo: number };

type StateConfig = {
  /** Blink parameters per dot, keyed by index and grid position. */
  blink: (i: number, row: number, col: number) => Blink;
};

/**
 * The two patterns this app draws.
 *
 * `loading` twinkles — the file viewer, where dots lighting at random is the
 * honest picture of waiting on bytes that arrive a window at a time.
 * `connecting` ripples outward from the centre, under the thread's working
 * indicator, because a turn in progress is one thing happening rather than
 * many.
 *
 * The component this was taken from ships eighteen further states and a set of
 * glyph tables (check, cross, bang, ellipsis…). None of them was ever rendered
 * here, so none of them is here. Adding one back means adding its entry to
 * this object and nothing else.
 */
const STATES = {
  loading: {
    blink: (i) => ({
      duration: 0.9 + hash(i, 2, 700),
      delay: -hash(i, 1, 1200),
      lo: 0.15,
    }),
  },
  connecting: {
    blink: (_i, row, col) => ({
      duration: 1.4,
      delay: -Math.max(Math.abs(row - CENTER), Math.abs(col - CENTER)) * 0.18,
      lo: 0.15,
    }),
  },
} satisfies Record<string, StateConfig>;

type DotMatrixState = keyof typeof STATES;

type DotMatrixProps = Omit<ComponentProps<"span">, "children"> & {
  state?: DotMatrixState;
  label?: string;
};

/* The registered hi/lo custom properties carry a transition, because removing or adding an animation never triggers a CSS transition on the animated property itself; transitioning the amplitude bounds is what makes state changes cross-fade. */
const DOT_MATRIX_CSS =
  '@property --aui-dot-matrix-hi{syntax:"<number>";inherits:false;initial-value:1}@property --aui-dot-matrix-lo{syntax:"<number>";inherits:false;initial-value:0.15}@keyframes aui-dot-matrix-blink{0%,100%{opacity:var(--aui-dot-matrix-hi,1)}50%{opacity:var(--aui-dot-matrix-lo,0.15)}}';

/**
 * Tiny 5x5 dot-matrix activity indicator. Dots inherit the surrounding text
 * colour and animate in a per-state pattern; state changes cross-fade per dot.
 *
 * ```tsx
 * <DotMatrix state="connecting" aria-hidden />
 * ```
 */
export function DotMatrix({
  className,
  state = "loading",
  label,
  ...props
}: DotMatrixProps) {
  const config: StateConfig = STATES[state];
  return (
    <span
      data-slot="dot-matrix"
      data-state={state}
      role="status"
      className={cn("inline-block size-4 shrink-0", className)}
      {...props}
    >
      <span className="sr-only">{label ?? state}</span>
      {/* Hoisted and deduplicated across instances by React; must live in HTML scope, inside the SVG it would be an SVG-namespace element React does not hoist. */}
      <style href="aui-dot-matrix" precedence="low">
        {DOT_MATRIX_CSS}
      </style>
      <svg
        aria-hidden
        viewBox="0 0 20 20"
        fill="currentColor"
        className="size-full"
      >
        {DOT_INDEXES.map((i) => {
          const row = Math.floor(i / GRID);
          const col = i % GRID;
          const blink = config.blink(i, row, col);
          return (
            <circle
              key={i}
              data-slot="dot-matrix-dot"
              cx={2 + col * 4}
              cy={2 + row * 4}
              r={1.3}
              className="[transition-property:--aui-dot-matrix-hi,--aui-dot-matrix-lo,opacity] duration-300 [animation-iteration-count:infinite] [animation-name:aui-dot-matrix-blink] [animation-timing-function:ease-in-out] motion-reduce:[animation-name:none]"
              style={
                {
                  opacity: 1,
                  animationDuration: `${blink.duration}s`,
                  animationDelay: `${blink.delay}s`,
                  "--aui-dot-matrix-hi": 1,
                  "--aui-dot-matrix-lo": blink.lo,
                } as CSSProperties
              }
            />
          );
        })}
      </svg>
    </span>
  );
}
