# Writing a widget that looks like the rest of the app

Hand this to whoever — or whatever — is writing a widget. It is short on
purpose: most of the work is already done for you, and the main thing to learn
is what *not* to write.

A widget is one HTML file. It runs in a sandboxed iframe with its own opaque
origin, so it inherits nothing from the frame — not a font, not a colour, not a
border radius. That is the danger the rest of this file exists to answer: if
every widget invents its own palette, fourteen widgets are fourteen apps.

## The one rule

**The frame injects a stylesheet into your document before your own styles
run.** It sets the token table, sensible defaults for every element you are
likely to use, and `color-scheme`. You do not import it, link it, or copy it.

So:

> Write plain, semantic HTML and it will already look right. Reach for a token
> when you need a value. Never write a hex colour, a font stack, or a hand-typed
> radius.

A widget whose `<style>` block is empty is a correct widget. That is the target.

## What you get for free

These already look like Second Brain with no classes on them:

`body` · `h1`–`h4` · `p` · `a` · `small` · `hr` · `button` · `input` ·
`textarea` · `select` · `label` · `table`/`th`/`td` · `code` · `pre` ·
`blockquote` · the scrollbar · the focus ring · `::selection`

`button` is the filled primary. Two classes give you the quieter ones:

```html
<button>Save</button>
<button class="sb-secondary">Cancel</button>
<button class="sb-ghost">Dismiss</button>
```

Four more classes, for the shapes that recur:

| Class | What it is |
|---|---|
| `.sb-card` | A grouping: hairline border, surface fill, subtle shadow |
| `.sb-row` | A list row with a hover fill and a row radius |
| `.sb-pill` | A small status or count |
| `.sb-muted` | Secondary text |

## The tokens

Every one is a CSS custom property on `:root`. Use `var(--sb-…)`.

**Colour** — the palette is achromatic. Everything that looks like a colour is a
lightness, and the three that carry hue mean something.

| Token | Use it for |
|---|---|
| `--sb-bg` | The page behind everything |
| `--sb-fg` | Body text |
| `--sb-card` | A surface raised off the page |
| `--sb-surface` | A quiet fill: hover, a selected row, a code block |
| `--sb-muted` | Secondary text, labels, icons at rest |
| `--sb-line` | Borders and hairlines |
| `--sb-ring` | The focus outline |
| `--sb-accent` / `--sb-accent-fg` | The one emphatic fill and its text |
| `--sb-bad` | A failure, a destructive action. The only token that carries hue. |
| `--sb-selection` | Selected text, a pressed state |
| `--sb-shadow` / `--sb-shadow-subtle` | Two elevations |

Need a colour between two of these? `color-mix(in srgb, var(--sb-fg) 10%,
transparent)`. The palette is `oklch`, so a mix stays neutral; mixing hex
drifts, which is why there are no hex values in the table.

**Shape and type** — `--sb-font`, `--sb-font-mono`, `--sb-space`,
`--sb-space-sm`, and `--sb-tint`.

Type *sizes* are deliberately not tokens. The frame sets a body size and a
line height and the elements above inherit them; a widget that needs a bigger
heading says `font-size: 1.25rem` and means it. A ladder nobody could remember
was four more names to get subtly wrong.

Radii are picked by *what the thing is*, never by eye:

`--sb-radius-control` (a button, an input) · `--sb-radius-row` · 
`--sb-radius-surface` (a card) · `--sb-radius-dialog` · `--sb-radius-pill`

Control heights are not tokens either: `button`, `input`, `textarea` and
`select` already come out at the app's height, which is the only one that
matters.

**Motion** — one easing, `--sb-ease`, and three durations:
`--sb-motion-fast` (120ms) · `--sb-motion-control` (140ms) ·
`--sb-motion-panel` (220ms).

Pick by what is moving: a press is the fastest thing in the app, a panel is the
slowest. Anything outside the ladder will read as being from somewhere else.
`prefers-reduced-motion` is already honoured for you.

## Light and dark

**Do not ask the operating system.** A `prefers-color-scheme` media query in a
widget is right until somebody picks a theme in the app, at which point your
widget is the one box out of fourteen that disagrees.

The frame decides and tells you. The tokens simply change value, so if you only
used tokens you have nothing to do. If you need to *know*, read `brain.scheme`
(`"light"` or `"dark"`), and listen for it changing:

```js
brain.on("scheme", (scheme) => draw(scheme));   // returns an unsubscribe
```

`brain.on("size", …)` is the same shape, for a widget that has to redraw a
canvas when its slot changes.

## Layout

The frame owns the box. Your widget is resized, moved between slots, slid into a
drawer and hidden without being asked, so:

- Lay out at whatever size you are given. No fixed widths, no `100vw`/`100vh` —
  the viewport is the slot, and the slot changes.
- Do not try to draw outside yourself. Menus, dialogs and tooltips that overhang
  belong to the frame; a popover in a widget is clipped by the iframe.
- `body` has **no padding**: the frame hands over the panel edge to edge, so a
  full-bleed widget can be one. Pad your own container — most widgets should.
-   table or a canvas that should meet its edges.

## Things not to do

- A hex colour, an `rgb()`, or a named colour.
- A font stack. `system-ui` is not close enough; the app has one font.
- `outline: none` on anything focusable, unless you draw a ring yourself.
- A `prefers-color-scheme` query.
- `!important` against an injected rule. Every default here is a single element
  or single class selector, so your own rule already wins.
- A CSS framework. A CDN link is a second design system and a network dependency
  in a document that may be offline.

## A widget, complete

```html
<!doctype html>
<meta charset="utf-8">
<title>Recent files</title>
<style>
  /* Only what the defaults cannot know: this widget's own arrangement. */
  .list { display: grid; gap: 0.25rem; }
</style>
<h2>Recent files</h2>
<p class="sb-muted">The last few things written in the workspace.</p>
<div class="list" id="list"></div>
<button id="refresh" class="sb-secondary">Refresh</button>
<script>
  const list = document.querySelector("#list");
  async function draw() {
    const files = await brain.call("fs.list", { path: ".", limit: 10 });
    list.replaceChildren(...files.map((f) => {
      const row = document.createElement("div");
      row.className = "sb-row";
      row.textContent = f.name ?? String(f);
      return row;
    }));
  }
  document.querySelector("#refresh").addEventListener("click", draw);
  draw();
</script>
```

No colours, no fonts, no radii, and it matches the frame exactly.

## Where this comes from

`src/theme.js` is the table, and there is no second copy of it. The frame reads
the tokens from it and the mount injects the same text into your document. The
values are the old Second Brain UI's, carried over deliberately — if something
here looks arbitrary, it is almost certainly a decision that was made once, over
there, and is worth keeping.
