/**
 * What a widget is dressed in: the app's tokens, and defaults for the elements
 * it is likely to use.
 *
 * This is the *styling* half of preparing a widget's document. The other half
 * — the bridge — belongs to `lib/html-app.ts`; what is here is the part that
 * makes a widget furniture inside the app rather than a document that happens
 * to be on screen.
 *
 * **The token values are read off the running app, never written down here.**
 * A widget is a separate document and inherits nothing, so the frame has to
 * hand it a stylesheet; the temptation is to keep a copy of the palette in this
 * file, and the way a copy fails is that widgets drift half a shade away from
 * the app around them. What is written down is a list of *names* — the app's
 * variable for each widget-facing token — and the values come from
 * `getComputedStyle(document.documentElement)` at mount.
 *
 * The aliasing is deliberate rather than incidental. A widget should not have
 * to know this app is built on shadcn's `--foreground`/`--border` vocabulary,
 * because a future frame might not be; `--sb-fg` and `--sb-line` are the
 * contract, and this table is where the two meet. `WIDGET_STYLE.md` is that
 * contract written for whoever — or whatever — is authoring one.
 */

/**
 * Widget-facing token ← the app's own variable.
 *
 * **A widget's page background is the surface it sits *on*, which is the
 * panel — `--sidebar` — and not `--background`, which is the chat's.** The two
 * are a shade apart on purpose: the panel is lifted off the conversation, and
 * a widget taking the chat's colour reads as a hole cut in the panel rather
 * than as something in it. In dark that is the difference between the near
 * black the thread uses and the lifted grey everything beside it uses.
 *
 * So the whole surface family comes from the sidebar's: its background, its
 * text, its border, its raised fill and its focus ring. What deliberately does
 * *not* is `--sb-accent`, which stays on `--primary`: `--sidebar-primary` is a
 * saturated blue in the dark palette — shadcn's default, unused by this app —
 * and a widget's one prominent colour must not be the single thing on screen
 * that is off-palette.
 *
 * If a widget slot ever lands somewhere other than the panel, this table is
 * where that decision lives.
 */
const ALIASES: Record<string, string> = {
  "--sb-bg": "--sidebar",
  "--sb-fg": "--sidebar-foreground",
  // Raised *against the panel*. `--card` is the chat's raised fill and in the
  // dark palette it is the panel's own colour, so a card drawn with it would
  // be invisible on exactly the surface a widget is on.
  "--sb-card": "--sidebar-accent",
  "--sb-surface": "--sidebar-accent",
  "--sb-muted": "--muted-foreground",
  "--sb-line": "--sidebar-border",
  "--sb-ring": "--sidebar-ring",
  "--sb-accent": "--primary",
  "--sb-accent-fg": "--primary-foreground",
  "--sb-bad": "--destructive",
  "--sb-selection": "--sb-selection",
};

/** Tokens the frame states outright, because the app emits nothing to read —
 *  see the note in `tokenBlock`. */
const STATED = ["--sb-font", "--sb-font-mono", "--sb-space", "--sb-space-sm"];

/** Tokens the app already spells the way a widget should see them. */
const PASSTHROUGH = [
  "--sb-ease",
  "--sb-motion-fast",
  "--sb-motion-control",
  "--sb-motion-panel",
  "--sb-radius-control",
  "--sb-radius-row",
  "--sb-radius-surface",
  "--sb-radius-dialog",
  "--sb-radius-pill",
  "--sb-shadow",
  "--sb-shadow-subtle",
  "--sb-tint",
];

/**
 * Every token a widget is promised, by name.
 *
 * **This is the contract `WIDGET_STYLE.md` and the template describe**, and it
 * is exported so those two can be checked against it rather than trusted. A
 * doc naming a token the frame does not send is worse than a doc naming none:
 * `var(--sb-text)` resolves to nothing, which makes the whole declaration
 * invalid and drops it, silently — which is exactly how every widget came to
 * be rendered in Times New Roman.
 */
export const WIDGET_TOKENS: readonly string[] = [
  ...Object.keys(ALIASES),
  ...PASSTHROUGH,
  ...STATED,
];

/**
 * The app's current token values, as a `:root` block.
 *
 * Read at call time, which is what makes a theme change a re-read rather than a
 * second source of truth. `getPropertyValue` answers the empty string for a
 * variable that does not exist, and an empty declaration is skipped rather than
 * written — a `--sb-fg: ;` would be invalid and take the whole block with it.
 */
export function tokenBlock(scheme: "light" | "dark"): string {
  const computed = getComputedStyle(document.documentElement);
  const lines: string[] = [];

  for (const [alias, source] of Object.entries(ALIASES)) {
    const value = computed.getPropertyValue(source).trim();
    if (value) lines.push(`  ${alias}: ${value};`);
  }
  for (const name of PASSTHROUGH) {
    const value = computed.getPropertyValue(name).trim();
    if (value) lines.push(`  ${name}: ${value};`);
  }
  /*
   * The tokens the frame states rather than forwards, because there is nothing
   * to forward them from.
   *
   * **Spacing and type are Tailwind's here, and a widget has no Tailwind.**
   * Worse for type: `--font-sans` is declared inside `@theme inline`, and
   * `inline` is precisely the Tailwind directive that says "inline this into
   * the utilities and emit no custom property" — so reading it answers the
   * empty string. `tokenBlock` skips an empty value rather than writing an
   * invalid declaration, so `--sb-font` simply never arrived and every widget
   * fell back to the browser's serif default. Silent, and it looked like a
   * widget authoring mistake rather than a missing token.
   *
   * The sans stack is therefore read off the *running app's body*, which is
   * the resolved value rather than a copy of the declaration — one source, and
   * it follows the app if the stack ever changes. There is no element rendering
   * mono text to read, so that one is stated; it is the only token in here that
   * is a copy, and `test_the_style_contract_is_kept` is what notices if it
   * drifts.
   */
  const body = getComputedStyle(document.body);
  lines.push(`  --sb-font: ${body.fontFamily || "ui-sans-serif, system-ui, sans-serif"};`);
  lines.push('  --sb-font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, '
    + 'Consolas, "Liberation Mono", "Courier New", monospace;');
  lines.push("  --sb-space: 1rem;");
  lines.push("  --sb-space-sm: 0.5rem;");

  // Declared, not inferred: a widget must never ask the operating system what
  // colour to be, or one box out of fourteen disagrees the moment somebody
  // picks a theme in the app.
  lines.push(`  color-scheme: ${scheme};`);

  return `:root {\n${lines.join("\n")}\n}`;
}

/**
 * Defaults for the elements a widget is likely to use.
 *
 * **A widget whose `<style>` block is empty should already look right.** That
 * is the target, and it is the only thing standing between "fourteen widgets"
 * and "fourteen apps" — an author who never reads `WIDGET_STYLE.md` still ships
 * something that belongs. Everything here is plain, semantic selectors for that
 * reason; the handful of classes are the shapes that recur often enough to be
 * worth a name.
 */
const BASE = `
*, *::before, *::after { box-sizing: border-box; }
html, body { height: 100%; }
/* **No padding, and that is the frame's decision rather than an oversight.**
   The widget is given the panel edge to edge — no inset, no border, nothing
   between its first pixel and the box — so a widget that wants to fill the
   zone can, and one that wants breathing room pads its own container. A frame
   that padded for you would be one no full-bleed widget could ever undo. */
body {
  margin: 0;
  padding: 0;
  background: var(--sb-bg);
  color: var(--sb-fg);
  font-family: var(--sb-font, ui-sans-serif, system-ui, sans-serif);
  font-size: 0.875rem;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
h1, h2, h3, h4 { margin: 0 0 0.5rem; font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 1.125rem; } h2 { font-size: 1rem; }
h3 { font-size: 0.9375rem; } h4 { font-size: 0.875rem; }
p { margin: 0 0 0.75rem; }
a { color: var(--sb-fg); text-underline-offset: 2px; }
small, .sb-muted { color: var(--sb-muted); font-size: 0.8125rem; }
hr { height: 0; margin: 1rem 0; border: 0; border-top: 0.5px solid var(--sb-line); }
button, input, textarea, select {
  font: inherit;
  border-radius: var(--sb-radius-control);
  transition: background var(--sb-motion-fast) var(--sb-ease),
    border-color var(--sb-motion-fast) var(--sb-ease);
}
button {
  min-height: 2rem;
  padding: 0 0.75rem;
  border: 0;
  background: var(--sb-accent);
  color: var(--sb-accent-fg);
  cursor: pointer;
}
button:hover { opacity: 0.9; }
button.sb-secondary { background: var(--sb-surface); color: var(--sb-fg); }
button.sb-ghost { background: transparent; color: var(--sb-muted); }
button.sb-ghost:hover { background: var(--sb-surface); color: var(--sb-fg); }
input, textarea, select {
  min-height: 2rem;
  padding: 0.25rem 0.5rem;
  border: 0.5px solid var(--sb-line);
  background: var(--sb-bg);
  color: var(--sb-fg);
}
:focus-visible { outline: 2px solid var(--sb-ring); outline-offset: 1px; }
label { display: inline-block; margin-bottom: 0.25rem; color: var(--sb-muted); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.375rem 0.5rem; border-bottom: 0.5px solid var(--sb-line); text-align: left; }
th { color: var(--sb-muted); font-weight: 500; }
code, pre { font-family: var(--sb-font-mono, ui-monospace, monospace); font-size: 0.8125rem; }
code { padding: 0.1rem 0.3rem; border-radius: 0.375rem; background: var(--sb-surface); }
pre { padding: 0.75rem; border-radius: var(--sb-radius-surface); background: var(--sb-surface); overflow: auto; }
pre code { padding: 0; background: none; }
blockquote {
  margin: 0 0 0.75rem;
  padding-left: 0.75rem;
  border-left: 2px solid var(--sb-line);
  color: var(--sb-muted);
}
.sb-card {
  padding: 0.75rem;
  border: 0.5px solid var(--sb-line);
  border-radius: var(--sb-radius-surface);
  background: var(--sb-card);
  box-shadow: var(--sb-shadow-subtle);
}
.sb-row { padding: 0.5rem 0.625rem; border-radius: var(--sb-radius-row); }
.sb-row:hover { background: var(--sb-surface); }
.sb-pill {
  display: inline-block;
  padding: 0.05rem 0.4rem;
  border-radius: var(--sb-radius-pill);
  background: var(--sb-selection);
  font-size: 0.75rem;
}
::selection { background: var(--sb-selection); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb {
  border: 3px solid transparent;
  border-radius: 999px;
  background: var(--sb-line);
  background-clip: content-box;
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; }
}
`;


/** The whole stylesheet a widget is handed, in the order it is handed it. */
export function widgetStyles(scheme: "light" | "dark"): string {
  return tokenBlock(scheme) + BASE;
}
