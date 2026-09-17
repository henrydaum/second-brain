/**
 * A Markdown file, rendered.
 *
 * **Why not `MarkdownTextPrimitive`, which this app already has.** That
 * component reads its text from `useMessagePartText` — it renders *the message
 * it is inside*, and there is no message here. So this drops to the library
 * underneath it, `react-markdown`, which both of them share, and re-uses the
 * two things that actually matter for consistency: the `.aui-md` stylesheet, so
 * a note and a reply are typeset identically, and `code-block.tsx`, so a fenced
 * block in a file and the same block quoted in a reply are the same object.
 *
 * **Raw HTML in the file is not rendered, and that is not an oversight.**
 * `react-markdown` drops embedded HTML unless `rehype-raw` is added, and adding
 * it here would put arbitrary markup from a file the agent wrote, downloaded,
 * or was handed into this app's own origin — the same hole `file-view.tsx`
 * closes for SVG with `sandbox=""`, and with the same stakes, since the
 * production gateway credentials same-origin `/sdk` calls. A note with a stray
 * `<div>` in it losing that `<div>` is the correct trade.
 *
 * URLs are sanitised on the way in too: `defaultUrlTransform` is what refuses
 * `javascript:` in an `href`, and it stays in the path for everything this does
 * not deliberately rewrite.
 */

import { Fragment, memo, useMemo, type FC } from "react";
import ReactMarkdown, {
  defaultUrlTransform,
  type Components,
  type ExtraProps,
  type UrlTransform,
} from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  CodeHeader,
  HighlightedCode,
} from "@/components/assistant-ui/code-block";
import { fileUrl } from "@/lib/client";
import { hostPathFor, splitFrontMatter, type FrontMatterRow } from "@/lib/markdown";
import { cn } from "@/lib/utils";

/* ── Fenced code ────────────────────────────────────────────────────── */

type Fence = { language: string | undefined; code: string };

/**
 * The language and the source out of a `<pre><code class="language-x">`.
 *
 * Read off the hast node rather than off `children`, because `children` is
 * already React elements by then and the fence's language only survives as a
 * class name on a node nobody renders.
 */
function fenceOf(node: ExtraProps["node"]): Fence | null {
  const only = node?.children.length === 1 ? node.children[0] : undefined;
  if (!only || only.type !== "element" || only.tagName !== "code") return null;

  const names = only.properties.className;
  const tag = (Array.isArray(names) ? names.map(String) : []).find((name) =>
    name.startsWith("language-"),
  );
  const code = only.children
    .map((child) => (child.type === "text" ? child.value : ""))
    .join("");

  // Every fenced block ends with the newline before its closing fence, and
  // rendering it draws an empty last line inside the box.
  return {
    language: tag?.slice("language-".length) || undefined,
    code: code.replace(/\n$/, ""),
  };
}

/* ── Front matter ───────────────────────────────────────────────────── */

/**
 * The block at the top, as a strip rather than as prose.
 *
 * It is metadata about the note and not part of it, so it reads as a label:
 * small, muted, and visibly a different kind of thing from the document below.
 */
const FrontMatter: FC<{ rows: FrontMatterRow[] }> = ({ rows }) => (
  <dl className="bg-muted/40 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 rounded-md border px-3 py-2 text-xs">
    {rows.map((row, index) =>
      row.key ? (
        <Fragment key={index}>
          <dt className="text-muted-foreground font-medium">{row.key}</dt>
          <dd className="min-w-0 break-words">{row.value}</dd>
        </Fragment>
      ) : (
        // A line the parser could not read. Shown whole rather than dropped —
        // it was in the file, and guessing at it is how metadata goes missing.
        <dd key={index} className="text-muted-foreground col-span-2 break-words">
          {row.value}
        </dd>
      ),
    )}
  </dl>
);

/* ── The overrides ──────────────────────────────────────────────────── */

/**
 * An image beside the note, fetched from the host rather than from this origin.
 *
 * Only `src` is rewritten. An `href` is left to `defaultUrlTransform` so the
 * anchor below still receives the path as it was written and can decide for
 * itself what following it means.
 */
function urlTransformFor(path: string): UrlTransform {
  return (url, key) => {
    if (key === "src") {
      const target = hostPathFor(path, url);
      if (target) return fileUrl(target);
    }
    return defaultUrlTransform(url);
  };
}

function componentsFor(
  path: string,
  onOpenFile: ((path: string) => void) | undefined,
): Components {
  return {
    pre: ({ node, children, ...rest }) => {
      const fence = fenceOf(node);
      // Not a fenced block after all — an indented one, or something a plugin
      // built. The library's own `pre` is the right answer for those.
      if (!fence) return <pre {...rest}>{children}</pre>;

      // Header and block as siblings, which is how `MarkdownTextPrimitive`
      // composes them and what `.aui-md`'s one pairing rule expects.
      return (
        <>
          <CodeHeader language={fence.language} code={fence.code} />
          <HighlightedCode
            code={fence.code}
            language={fence.language}
            className="rounded-t-none"
          />
        </>
      );
    },

    /**
     * A link, and where it is allowed to take you.
     *
     * Three cases, and the middle one is the reason this override exists: a
     * relative link is a *file beside this one*, and letting the browser follow
     * it would resolve it against this app and navigate the conversation away.
     * It keeps a real `href` all the same — the `/files` URL — so a
     * middle-click or a ⌘-click still means "somewhere else" rather than
     * nothing at all.
     */
    a: ({ node: _node, href, children, ...rest }) => {
      // `[text]()` — a link somebody meant to finish. An empty `href` reloads
      // the page, which of all the outcomes here is the worst one.
      if (!href?.trim()) return <a {...rest}>{children}</a>;

      const target = hostPathFor(path, href);

      if (!target) {
        const offsite = /^[a-z][a-z0-9+.-]*:/i.test(href);
        return (
          <a
            href={href}
            {...(offsite
              ? { target: "_blank", rel: "noreferrer noopener" }
              : {})}
            {...rest}
          >
            {children}
          </a>
        );
      }

      return (
        <a
          href={fileUrl(target)}
          title={target}
          onClick={(event) => {
            if (!onOpenFile) return;
            if (
              event.button !== 0 ||
              event.metaKey ||
              event.ctrlKey ||
              event.shiftKey ||
              event.altKey
            ) {
              return;
            }
            event.preventDefault();
            onOpenFile(target);
          }}
          {...rest}
        >
          {children}
        </a>
      );
    },

    // Same treatment every other picture in this app gets — and `alt` arrives
    // in `rest`, written by whoever wrote the note. `src` has already been
    // pointed at the host by `urlTransformFor`.
    img: ({ node: _node, alt = "", ...rest }) => (
      <img alt={alt} loading="lazy" decoding="async" {...rest} />
    ),
  };
}

/* ── The component ──────────────────────────────────────────────────── */

type MarkdownPreviewProps = {
  /** The file's text, front matter and all. */
  text: string;
  /** Where it was read from — what relative links and images resolve against. */
  path: string;
  /** Follow a link to a neighbouring file. Omit where there is nowhere to
   *  follow it to, and those links fall back to their `/files` URL. */
  onOpenFile?: (path: string) => void;
  className?: string;
};

/**
 * Memoised, and for a sharper reason than the usual one.
 *
 * A whole file is up to `TEXT_CAP` — two megabytes — and re-rendering this
 * re-parses every byte of it, because `react-markdown` holds no parse between
 * renders. Meanwhile `MarkdownView` reads the file-activity context, which
 * hands out a new value whenever the agent touches a file. Without this, the
 * agent writing anything during a turn would re-parse whatever note happens to
 * be open. The props are all stable by construction, so the comparison is free.
 */
const MarkdownPreviewInner: FC<MarkdownPreviewProps> = ({
  text,
  path,
  onOpenFile,
  className,
}) => {
  const { front, body } = useMemo(() => splitFrontMatter(text), [text]);
  const components = useMemo(
    () => componentsFor(path, onOpenFile),
    [path, onOpenFile],
  );
  const urlTransform = useMemo(() => urlTransformFor(path), [path]);

  return (
    // `react-markdown` renders into a fragment, so every block below — and the
    // front matter strip — is a direct child of `.aui-md`, which is what its
    // sibling-spacing rules are written against.
    <div className={cn("aui-md", className)}>
      {front && front.length > 0 && <FrontMatter rows={front} />}
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={components}
        urlTransform={urlTransform}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
};

export const MarkdownPreview = memo(MarkdownPreviewInner);
