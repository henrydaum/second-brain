/**
 * The design tokens, in one place, because two places is how a widget comes to
 * look nearly right.
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
 */

/** Tokens that do not change with the scheme. */
const SHAPE = `
  --sb-font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --sb-font-mono: 13px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sb-radius-control: 0.625rem;
  --sb-radius-surface: 0.875rem;
  --sb-space: 1rem;
  --sb-ease: cubic-bezier(0.22, 1, 0.36, 1);
  --sb-motion-fast: 120ms;
`;

const LIGHT = `
  --sb-bg: #ffffff;
  --sb-surface: #f7f8fa;
  --sb-fg: #252525;
  --sb-muted: #6b7280;
  --sb-line: #e5e7eb;
  --sb-accent: #0f766e;
  --sb-bad: #b91c1c;
  --sb-shadow: 0 2px 8px rgb(0 0 0 / 6%);
`;

const DARK = `
  --sb-bg: #1c1c1c;
  --sb-surface: #242424;
  --sb-fg: #ededed;
  --sb-muted: #9ca3af;
  --sb-line: #333333;
  --sb-accent: #2dd4bf;
  --sb-bad: #f87171;
  --sb-shadow: 0 2px 8px rgb(0 0 0 / 24%);
`;

/** What the OS asks for, which is only ever the *starting* answer. */
export function preferredScheme() {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark" : "light";
}

/**
 * The whole stylesheet for one scheme: the tokens, plus the few rules that
 * make an empty document already look like part of the app.
 *
 * ``color-scheme`` is here rather than assumed: it is what tells the browser
 * to draw its *own* furniture — scrollbars, form controls, the flash of
 * background before a stylesheet lands — in the right shade. Without it a dark
 * widget gets a white scrollbar, which is the sort of detail that reads as
 * cheap without anybody being able to say why.
 */
export function themeCss(scheme = "light") {
  return `:root {
  color-scheme: ${scheme};
${SHAPE}${scheme === "dark" ? DARK : LIGHT}}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--sb-bg);
  color: var(--sb-fg);
  font: var(--sb-font);
}`;
}

/** Apply a scheme to the frame's own document. */
export function applyTheme(scheme) {
  let style = document.getElementById("sb-theme");
  if (!style) {
    style = document.createElement("style");
    style.id = "sb-theme";
    document.head.append(style);
  }
  style.textContent = themeCss(scheme);
}
