/**
 * The design tokens and the base stylesheet, in one place, because two places
 * is how a widget comes to look nearly right.
 *
 * A widget is a separate document. It inherits **nothing** from this page — not
 * a font, not a colour, not a border radius — so a widget author either invents
 * their own palette or is handed ours. Handing it over is what makes fourteen
 * independently written boxes read as one app, and it is the cheapest part of
 * the whole arrangement: the tokens are plain CSS custom properties, and
 * injecting them is a string.
 *
 * So this file is the table and there is no second copy. The frame applies it
 * to itself (`applyTheme`) and the mount injects the identical text into every
 * widget document before the widget's own styles, which is the order that lets
 * a widget override a token deliberately and stops it doing so by accident.
 *
 * Light and dark are separate strings rather than one block with a media query,
 * because the frame decides the scheme and tells the widget. A widget must not
 * ask the *operating system* what colour to be — the answer would be right
 * until somebody picks a theme in the app, at which point one box out of
 * fourteen disagrees with the rest.
 *
 * **The values are the old UI's, copied rather than re-chosen.** They are
 * `oklch` with zero chroma: Second Brain is an achromatic app, and every
 * apparent colour in it is a lightness. That is worth knowing before adding a
 * token, because a saturated one will look like it came from somewhere else —
 * the three that carry hue (`--sb-bad`, `--sb-caution`, `--sb-notice`) are the
 * whole licensed palette, and each of them means something. `oklch` rather than
 * hex because `color-mix` over it stays neutral where mixing hex drifts, and
 * several rules below rely on that.
 */

/** Tokens that do not change with the scheme: type, shape, and motion.
 *
 *  The ladders are the old UI's and are deliberately short. A radius is picked
 *  by *what the thing is* — a control, a row, a surface, a dialog — not by eye,
 *  and the same for a duration. Rungs are what stop fourteen authors from
 *  producing fourteen slightly different corners. */
const SHAPE = `
  --sb-font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --sb-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    "Liberation Mono", "Courier New", monospace;
  --sb-text: 0.875rem;
  --sb-text-sm: 0.8125rem;
  --sb-text-xs: 0.75rem;
  --sb-leading: 1.6;

  --sb-radius-control: 0.625rem;
  --sb-radius-row: 0.75rem;
  --sb-radius-surface: 0.875rem;
  --sb-radius-dialog: 1rem;
  --sb-radius-message: 1.125rem;
  --sb-radius-pill: 999px;

  --sb-control-sm: 2rem;
  --sb-control-md: 2.25rem;
  --sb-touch: 2.75rem;
  --sb-space: 1rem;

  --sb-ease: cubic-bezier(0.22, 1, 0.36, 1);
  --sb-motion-press: 70ms;
  --sb-motion-fast: 120ms;
  --sb-motion-control: 140ms;
  --sb-motion-reveal: 160ms;
  --sb-motion-enter: 180ms;
  --sb-motion-panel: 220ms;

  --sb-divider-width: 0.5px;
`;

/**
 * Light.
 *
 * `--sb-bg` is the page, `--sb-card` is a surface raised off it, and
 * `--sb-surface` is the quiet fill a hover or a selected row gets. In light the
 * first two are both white and only the hairline separates them; in dark they
 * come apart, which is why they are two tokens rather than one with an opacity.
 */
const LIGHT = `
  --sb-bg: oklch(1 0 0);
  --sb-fg: oklch(0.145 0 0);
  --sb-card: oklch(1 0 0);
  --sb-surface: oklch(0.97 0 0);
  --sb-muted: oklch(0.556 0 0);
  --sb-line: oklch(0.922 0 0);
  --sb-ring: oklch(0.708 0 0);
  --sb-accent: oklch(0.205 0 0);
  --sb-accent-fg: oklch(0.985 0 0);
  --sb-bad: oklch(0.577 0.245 27.325);
  --sb-caution: oklch(0.632 0.194 44);
  --sb-notice: oklch(0.641 0.152 74);
  --sb-selection: oklch(0.94 0 0);
  --sb-tint: oklch(0.59 0 0);
  --sb-ambient: oklch(0.93 0 0 / 65%);
  --sb-edge: rgb(255 255 255 / 75%);
  --sb-shadow: 0 8px 28px rgb(0 0 0 / 7%), 0 2px 6px rgb(0 0 0 / 3%);
  --sb-shadow-subtle: 0 2px 8px rgb(0 0 0 / 4%);
  --sb-shadow-dialog: 0 24px 80px rgb(0 0 0 / 20%);
  --sb-focus-strength: 65%;
  --sb-focus-halo: 12%;
`;

const DARK = `
  --sb-bg: oklch(0.145 0 0);
  --sb-fg: oklch(0.985 0 0);
  --sb-card: oklch(0.205 0 0);
  --sb-surface: oklch(0.269 0 0);
  --sb-muted: oklch(0.708 0 0);
  --sb-line: oklch(1 0 0 / 10%);
  --sb-ring: oklch(0.556 0 0);
  --sb-accent: oklch(0.922 0 0);
  --sb-accent-fg: oklch(0.205 0 0);
  --sb-bad: oklch(0.704 0.191 22.216);
  --sb-caution: oklch(0.783 0.163 48);
  --sb-notice: oklch(0.859 0.187 88);
  --sb-selection: oklch(0.3 0 0);
  --sb-tint: oklch(0.76 0 0);
  --sb-ambient: oklch(0.29 0 0 / 38%);
  --sb-edge: rgb(255 255 255 / 12%);
  --sb-shadow: 0 8px 28px rgb(0 0 0 / 24%), 0 2px 6px rgb(0 0 0 / 12%);
  --sb-shadow-subtle: 0 2px 8px rgb(0 0 0 / 12%);
  --sb-shadow-dialog: 0 24px 80px rgb(0 0 0 / 40%);
  --sb-focus-strength: 40%;
  --sb-focus-halo: 5%;
`;

/**
 * The part a widget gets for free by doing nothing.
 *
 * **Element selectors, not classes**, and that is the whole design of it. A
 * widget is written by an agent in one file, in a hurry, and the realistic
 * failure is not that it picks the wrong class — it is that it picks no class
 * at all and ships a browser-default `<button>` and a Times New Roman table.
 * So the defaults *are* the design system, and a widget that opts into nothing
 * still reads as part of the app.
 *
 * Every rule here is one a widget may overrule: none are `!important` and none
 * are more specific than a single element or a single class.
 * `--sb-divider-color` is derived from `--sb-fg` rather than stated per scheme,
 * because a hairline is a *fraction of the text colour* in both — the same
 * trick the old UI's `.sb-divider-*` helpers use, and the reason its 0.5px
 * seams read identically on white and on near-black.
 */
const BASE = `
*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  padding: var(--sb-space);
  background: var(--sb-bg);
  color: var(--sb-fg);
  font-family: var(--sb-font);
  font-size: var(--sb-text);
  line-height: var(--sb-leading);
  -webkit-font-smoothing: antialiased;
}

::selection { background: var(--sb-selection); }

h1, h2, h3, h4 {
  margin: 0 0 0.5rem;
  font-weight: 600;
  line-height: 1.3;
  letter-spacing: -0.01em;
}
h1 { font-size: 1.25rem; }
h2 { font-size: 1.0625rem; }
h3 { font-size: 0.9375rem; }
h4 { font-size: var(--sb-text); color: var(--sb-muted); }
p { margin: 0 0 0.75rem; }
small, .sb-muted { color: var(--sb-muted); font-size: var(--sb-text-sm); }

a { color: var(--sb-fg); text-decoration-color: var(--sb-line); }
a:hover { text-decoration-color: currentColor; }

hr {
  height: 0;
  margin: var(--sb-space) 0;
  border: 0;
  border-top: var(--sb-divider-width) solid var(--sb-divider-color);
}

/* One focus treatment, on everything, because a ring drawn two ways reads as
 * two apps. Never remove it: pointer users never see it. */
:focus-visible {
  outline: 2px solid var(--sb-ring);
  outline-offset: 2px;
}

button, .sb-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  height: var(--sb-control-md);
  padding: 0 0.875rem;
  border: 0.5px solid transparent;
  border-radius: var(--sb-radius-control);
  background: var(--sb-accent);
  color: var(--sb-accent-fg);
  font: inherit;
  font-size: var(--sb-text-sm);
  font-weight: 500;
  cursor: pointer;
  transition: background var(--sb-motion-control) var(--sb-ease),
    border-color var(--sb-motion-control) var(--sb-ease),
    color var(--sb-motion-control) var(--sb-ease),
    scale var(--sb-motion-press) var(--sb-ease);
}
button:active:not(:disabled) { scale: 0.98; }
button:disabled { opacity: 0.5; cursor: not-allowed; }

/* The two quiet variants, which is what most of a widget is made of. */
button.sb-secondary, .sb-btn.sb-secondary {
  background: var(--sb-surface);
  color: var(--sb-fg);
  border-color: var(--sb-line);
}
button.sb-secondary:hover:not(:disabled) { background: var(--sb-selection); }
button.sb-ghost, .sb-btn.sb-ghost {
  background: transparent;
  color: var(--sb-muted);
}
button.sb-ghost:hover:not(:disabled) {
  background: var(--sb-surface);
  color: var(--sb-fg);
}

input, textarea, select {
  width: 100%;
  min-height: var(--sb-control-md);
  padding: 0.375rem 0.625rem;
  border: 0.5px solid var(--sb-line);
  border-radius: var(--sb-radius-control);
  background: var(--sb-bg);
  color: var(--sb-fg);
  font: inherit;
  font-size: var(--sb-text-sm);
  transition: border-color var(--sb-motion-control) var(--sb-ease),
    box-shadow var(--sb-motion-control) var(--sb-ease);
}
input:focus, textarea:focus, select:focus {
  outline: none;
  border-color: color-mix(in srgb, var(--sb-tint) var(--sb-focus-strength), var(--sb-line));
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--sb-tint) var(--sb-focus-halo), transparent);
}
input[type="checkbox"], input[type="radio"] {
  width: auto;
  min-height: 0;
  accent-color: var(--sb-accent);
}
label {
  display: block;
  margin-bottom: 0.25rem;
  font-size: var(--sb-text-sm);
  color: var(--sb-muted);
}

/* A card is the one raised thing. A widget already sits in a slot with a
 * hairline around it, so a card inside one is a *grouping* and not a frame —
 * hence a border and only the subtle shadow. */
.sb-card {
  padding: var(--sb-space);
  border: 0.5px solid var(--sb-line);
  border-radius: var(--sb-radius-surface);
  background: var(--sb-card);
  box-shadow: var(--sb-shadow-subtle);
}
.sb-row {
  padding: 0.5rem 0.75rem;
  border-radius: var(--sb-radius-row);
  transition: background var(--sb-motion-control) var(--sb-ease);
}
.sb-row:hover { background: var(--sb-surface); }
.sb-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.125rem 0.5rem;
  border-radius: var(--sb-radius-pill);
  background: var(--sb-surface);
  color: var(--sb-muted);
  font-size: var(--sb-text-xs);
}

table { width: 100%; border-collapse: collapse; font-size: var(--sb-text-sm); }
th, td {
  padding: 0.5rem 0.625rem;
  text-align: left;
  border-bottom: var(--sb-divider-width) solid var(--sb-divider-color);
}
th { font-weight: 500; color: var(--sb-muted); }
tbody tr:last-child td { border-bottom: 0; }

code, kbd, samp, pre { font-family: var(--sb-font-mono); font-size: var(--sb-text-sm); }
code {
  padding: 0.0625rem 0.3125rem;
  border-radius: 0.375rem;
  background: var(--sb-surface);
}
pre {
  margin: 0 0 0.75rem;
  padding: 0.75rem;
  overflow: auto;
  border: 0.5px solid var(--sb-line);
  border-radius: var(--sb-radius-control);
  background: var(--sb-surface);
}
pre code { padding: 0; background: none; }

blockquote {
  margin: 0 0 0.75rem;
  padding-left: 0.75rem;
  border-left: 2px solid var(--sb-line);
  color: var(--sb-muted);
}

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
  border: 3px solid transparent;
  border-radius: var(--sb-radius-pill);
  background: color-mix(in srgb, var(--sb-fg) 18%, transparent);
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
  background-color: color-mix(in srgb, var(--sb-fg) 30%, transparent);
}
::-webkit-scrollbar-track { background: transparent; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
    scale: none !important;
  }
}`;

/** The id the injected sheet carries, in the frame's document and in every
 *  widget's. It is the handle a scheme change needs: replacing the *text* of
 *  one element is what makes a widget change palette without reloading, and a
 *  widget that reloaded would lose everything it was holding. */
export const THEME_STYLE_ID = "sb-theme";

/** The three answers a person may give, and the one the app starts on. */
export const THEMES = ["system", "light", "dark"];
export const DEFAULT_THEME = "system";

/** What the OS asks for, which is only the answer while the preference is
 *  "system". */
export function preferredScheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark" : "light";
}

/** A preference resolved to one of the two palettes that actually exist. */
export function resolveScheme(theme) {
  return theme === "light" || theme === "dark" ? theme : preferredScheme();
}

/**
 * The whole stylesheet for one scheme: the tokens, then the base rules.
 *
 * ``color-scheme`` is here rather than assumed: it is what tells the browser to
 * draw its *own* furniture — the native scrollbar, a date picker, a `<select>`
 * dropdown, the flash of background before a stylesheet lands — in the right
 * shade. Without it a dark widget gets a white scrollbar and a white dropdown,
 * which is the detail that makes a theme look half-applied.
 */
export function themeCss(scheme = "light") {
  return tokensCss(scheme) + BASE;
}

/**
 * The tokens alone, with no base rules — what the **frame** gets.
 *
 * The frame is not a widget. It has its own stylesheet, written against these
 * same tokens, and it is chrome rather than content: giving it the base sheet
 * would put a filled primary button under every `.bar-btn` and a 1rem padding
 * on the body of a layout whose whole point is that its panels meet with no
 * gap. So the *table* is shared and the *defaults* are not, which is the split
 * that matters — a token with two values is drift, a default a widget needs
 * and the frame does not is just two different documents.
 */
export function tokensCss(scheme = "light") {
  return `:root {
  color-scheme: ${scheme};
${SHAPE}${scheme === "dark" ? DARK : LIGHT}  --sb-divider-color: color-mix(in srgb, var(--sb-fg) 10%, transparent);
}
`;
}

/**
 * Apply a scheme to the frame's own document.
 *
 * The inline properties are the pre-paint script's, and they have to be moved
 * rather than merely overruled: an inline `color-scheme` beats the one in the
 * token block, so a page opened dark and switched to light kept dark
 * scrollbars and a dark `<select>` — the exact half-applied look `color-scheme`
 * exists to prevent, arriving by the mechanism that was meant to prevent it.
 * The background is handed back to the stylesheet outright, since past first
 * paint it has nothing left to say.
 */
export function applyTheme(scheme) {
  document.documentElement.style.colorScheme = scheme;
  document.documentElement.style.background = "";
  let style = document.getElementById(THEME_STYLE_ID);
  if (!style) {
    style = document.createElement("style");
    style.id = THEME_STYLE_ID;
    document.head.append(style);
  }
  style.textContent = tokensCss(scheme);
}
