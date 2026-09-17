/**
 * Preparing a widget's document: the theme, then the bridge, then its markup.
 *
 * Order is the whole of it. The theme goes into `<head>` *first* so the
 * widget's own styles come after and win — a widget that wants to override a
 * token may, and one that says nothing inherits the app's look rather than the
 * browser's defaults. The bridge goes in before any of the author's scripts,
 * because a widget calling `brain` at the top of its first script would
 * otherwise find nothing there.
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
  "--sb-font": "--font-sans",
  "--sb-font-mono": "--font-mono",
};

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
  // The one token written down rather than forwarded, and the reason is that
  // there is nothing to forward it from: spacing in this app is Tailwind's,
  // and a widget has no Tailwind. A widget still needs *a* unit of room that
  // agrees with the frame's, so the frame states one.
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
  font-family: var(--sb-font);
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
code, pre { font-family: var(--sb-font-mono); font-size: 0.8125rem; }
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

/**
 * The bridge, as source text for a `<script>` in the widget's own document.
 *
 * **It grants no authority of its own.** It relays Requests to the frame,
 * which makes them with the same `sdk()` everything else uses, so the kernel
 * classifies a widget's Request exactly as it classifies a tool call and an
 * unsafe one still raises a dialog. What it withholds is the `frontend.*`
 * family: identity, attendance and approval answers belong to the frame, which
 * holds the stream — a widget answering its own approval dialog would be
 * approving itself.
 *
 * It is a string rather than a module because it has to execute in the
 * *widget's* realm. Nothing here is imported, bundled or type-checked; that is
 * the price of the boundary, and it is why this stays as small as it can be.
 */
function bridgeScript(channel: string, token: string): string {
  return `(function () {
  var pending = new Map();
  var next = 1;
  var listeners = { size: [], scheme: [] };
  var state = { size: { width: 0, height: 0 }, scheme: "light" };

  window.addEventListener("message", function (event) {
    // The frame is the only sender we answer to, and it is recognised by
    // object identity rather than by origin: this document has an opaque
    // origin, so \`event.origin\` on anything it sends is the string "null".
    if (event.source !== window.parent) return;
    var message = event.data;
    if (!message || message.channel !== ${JSON.stringify(channel)}) return;

    if (message.kind === "result") {
      var settle = pending.get(message.id);
      if (!settle) return;
      pending.delete(message.id);
      if (message.error) settle.reject(Object.assign(new Error(message.error), { code: message.code || "" }));
      else settle.resolve(message.data);
      return;
    }
    if (message.kind === "size" || message.kind === "scheme") {
      state[message.kind] = message.value;
      if (message.kind === "scheme" && typeof message.css === "string") {
        var sheet = document.getElementById("sb-theme");
        if (sheet) sheet.textContent = message.css;
      }
      (listeners[message.kind] || []).forEach(function (fn) {
        try { fn(message.value); } catch (error) { console.error(error); }
      });
    }
  });

  window.brain = {
    call: function (type, args) {
      var id = next++;
      return new Promise(function (resolve, reject) {
        pending.set(id, { resolve: resolve, reject: reject });
        // "*" as the target, because an opaque origin has no name to address.
        // That is safe only because the frame checks \`event.source\`.
        window.parent.postMessage({
          channel: ${JSON.stringify(channel)},
          token: ${JSON.stringify(token)},
          kind: "call",
          id: id,
          type: type,
          args: args || {},
        }, "*");
      });
    },
    on: function (kind, fn) {
      if (!listeners[kind]) return function () {};
      listeners[kind].push(fn);
      return function () {
        listeners[kind] = listeners[kind].filter(function (other) { return other !== fn; });
      };
    },
    get size() { return state.size; },
    get scheme() { return state.scheme; },
  };
})();`;
}

/**
 * The finished document, for `srcdoc`.
 *
 * The theme and the bridge are *prepended* rather than inserted into the
 * widget's `<head>`, and the difference matters for a file that has no `<head>`
 * at all — which a hand-written widget very often does not. A browser building
 * the DOM will hoist what belongs in a head; a string replace on a tag that is
 * not there silently drops both.
 */
export function widgetDocument(
  html: string,
  options: { channel: string; token: string; scheme: "light" | "dark" },
): string {
  const head = [
    `<meta charset="utf-8">`,
    `<meta name="viewport" content="width=device-width, initial-scale=1">`,
    `<style id="sb-theme">${tokenBlock(options.scheme)}${BASE}</style>`,
    `<script>${bridgeScript(options.channel, options.token)}</script>`,
  ].join("\n");
  return `${head}\n${html}`;
}
