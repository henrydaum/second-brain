/**
 * The two things a Markdown *file* is that a Markdown *reply* is not.
 *
 * Everything the agent says already renders through `MarkdownTextPrimitive`,
 * and none of this is needed there: a reply has no front matter and no
 * neighbours on disk. A note read off the host has both, and getting either one
 * wrong is visible immediately.
 *
 * **Front matter is not a horizontal rule.** A vault note opens with a `---`
 * fence around some YAML, and CommonMark has never heard of it: the opening
 * fence becomes an `<hr>`, `title: Something` becomes a paragraph, and the
 * *closing* fence turns that paragraph into a setext `<h2>`. So the top of
 * every note renders as a rule followed by a large heading reading
 * `title: Something` — which looks less like a preview than like a bug. Lifting
 * the block out before parsing is what makes the rest of the file the document.
 *
 * **A relative link is a host path, not a URL.** `![](attachments/chart.png)`
 * resolves against this app's origin unless something resolves it against the
 * directory the file came from, which is a broken image; a relative `<a>` is
 * worse, because clicking it navigates the whole single-page app away from the
 * conversation. `hostPathFor` is the one place that judgement is made.
 */

import { dirOf, resolveAgainst } from "@/lib/files";

/* ── Front matter ───────────────────────────────────────────────────── */

/** One line of the block, as close to what was written as parsing allows. A
 *  `key` of `""` is a line this could not read — kept, and shown whole. */
export type FrontMatterRow = { key: string; value: string };

export type SplitMarkdown = {
  /** The block that opened the file, or null when there was none. Empty when
   *  the fences were there and held nothing. */
  front: FrontMatterRow[] | null;
  /** Everything else — the document proper. */
  body: string;
};

/** The opening fence, its contents, and the closing `---` or `...`. Lazy, so
 *  the *first* closing fence ends it rather than the last one in the file. */
const BLOCK = /^\uFEFF?---[ \t]*\r?\n([\s\S]*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/;

/** `key: value`, where the key is the conservative subset every vault writes.
 *  Anything looser starts claiming ordinary prose that opened with a word and
 *  a colon. */
const PAIR = /^([A-Za-z0-9_][A-Za-z0-9 _.-]*?)[ \t]*:[ \t]*(.*)$/;

/** A YAML list item under the key above it. */
const ITEM = /^[ \t]*-[ \t]+(.*)$/;

/**
 * Split a note into its front matter and its body.
 *
 * **Nothing that fails to look like front matter is taken as front matter.** A
 * document that genuinely opens with a thematic break and closes a section with
 * another one is indistinguishable from a YAML block by fences alone, so the
 * block must also contain at least one `key: value` line. That makes the false
 * positive cost a stray `---` rather than a chunk of the document going
 * missing, which is the failure worth avoiding: one is visible, the other is
 * not.
 *
 * This is deliberately not a YAML parser. It reads the shape a vault actually
 * writes — flat keys, inline lists, block lists — and shows anything else
 * verbatim rather than guessing at it.
 */
export function splitFrontMatter(text: string): SplitMarkdown {
  const match = BLOCK.exec(text);
  if (!match) return { front: null, body: text };

  const lines = match[1].split(/\r?\n/);
  if (!lines.some((line) => PAIR.test(line))) return { front: null, body: text };

  const rows: FrontMatterRow[] = [];
  for (const line of lines) {
    if (!line.trim()) continue;

    const pair = PAIR.exec(line);
    if (pair) {
      rows.push({ key: pair[1].trim(), value: unquote(pair[2].trim()) });
      continue;
    }

    // A list item belongs to the key above it: `tags:` with nothing after the
    // colon, then one `- x` per line. Joined rather than nested, because these
    // are read at a glance and a nested list in a two-column strip is not.
    const item = ITEM.exec(line);
    const last = rows.at(-1);
    if (item && last) {
      last.value = last.value ? `${last.value}, ${unquote(item[1])}` : unquote(item[1]);
      continue;
    }

    rows.push({ key: "", value: line.trim() });
  }

  return { front: rows, body: text.slice(match[0].length) };
}

/** Strip one matched pair of quotes. YAML's escaping rules go further than
 *  this; the values that reach a reader's eye do not. */
function unquote(value: string): string {
  const quoted = /^(["'])([\s\S]*)\1$/.exec(value);
  return quoted ? quoted[2] : value;
}

/* ── Where a link points ────────────────────────────────────────────── */

/**
 * The host path a link or image in a note refers to, or null.
 *
 * Null means *this is already addressed somewhere else* — an `http(s)` URL, a
 * `mailto:`, an in-page `#fragment`, a protocol-relative `//host/path` — and
 * the caller has to treat it as a link off the machine rather than as a file
 * beside this one. That distinction is the entire job: everything null passes
 * through to the browser's own handling, and everything else becomes a path
 * `/files` can serve.
 *
 * **A single letter before the colon is a Windows drive, not a scheme.** Host
 * paths come from whatever machine wrote them, and `C:\notes\plan.md` is a file
 * rather than a URL in the `c` protocol.
 */
export function hostPathFor(document: string, href: string): string | null {
  const written = href.trim();
  if (!written || written.startsWith("#")) return null;
  // `//host/path` inherits the page's scheme — a URL, however path-shaped.
  if (written.startsWith("//")) return null;
  if (/^[a-z][a-z0-9+.-]*:/i.test(written) && !/^[a-z]:[\\/]/i.test(written)) {
    return null;
  }

  // Markdown escapes a path the way a URL does, so `my%20note.md` is a file
  // with a space in it. A malformed escape is left as written rather than
  // thrown over: the worst case is a link that does not resolve.
  const bare = written.split("#")[0].split("?")[0];
  let decoded = bare;
  try {
    decoded = decodeURIComponent(bare);
  } catch {
    /* as written */
  }
  if (!decoded) return null;

  return resolveAgainst(dirOf(document), decoded);
}
