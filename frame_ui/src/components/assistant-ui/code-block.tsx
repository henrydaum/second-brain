/**
 * Fenced code, coloured and copyable.
 *
 * These are the two slots `MarkdownTextPrimitive` leaves open, and the library
 * composes them as *siblings* — a header, then the content, inside one fragment
 * with no wrapper to style. That is why the two halves join themselves: the
 * header wears the top radius and drops its bottom border, and the `pre` below
 * squares off its top. Adjacent siblings, one apparent box.
 *
 * `SyntaxHighlighter` is only reached when the fence declared a language at all
 * — an undecorated ``` block goes to the library's own plain renderer and never
 * arrives here. So the fallback below is for the other case: a language that was
 * named but that Prism has no grammar for.
 */

import {
  useState,
  type ComponentType,
  type CSSProperties,
  type FC,
  type ReactNode,
} from "react";
import { CheckIcon, CopyIcon } from "lucide-react";
import { Highlight, Prism, type PrismTheme } from "prism-react-renderer";
import { themes } from "prism-react-renderer";
import type {
  CodeHeaderProps,
  SyntaxHighlighterProps,
} from "@assistant-ui/react-markdown";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { useResolvedTheme } from "@/lib/theme";

/** How long the tick stays after a copy. Matches the message action bar's own
 *  feedback, so the two do not read as different kinds of button. */
const COPIED_MS = 2000;

/** Prism's own names for the grammars it ships, plus the spellings people
 *  actually write — in a fence, or as a file extension. Anything not here
 *  renders unhighlighted, which is a plain monospace block in the right colours
 *  rather than a broken one. */
const ALIASES: Record<string, string> = {
  yml: "yaml",
  shell: "bash",
  sh: "bash",
  "c++": "cpp",
  "c#": "csharp",
  golang: "go",
  rs: "rust",
  ts: "typescript",
  tsx: "tsx",
  js: "javascript",
  mjs: "javascript",
  cjs: "javascript",
  jsx: "jsx",
  py: "python",
  pyw: "python",
  md: "markdown",
  markdown: "markdown",
  htm: "html",
  kt: "kotlin",
  kts: "kotlin",
  gql: "graphql",
  m: "objectivec",
  h: "c",
  hpp: "cpp",
  cc: "cpp",
  cxx: "cpp",
  text: "plain",
  txt: "plain",
};

/** The grammar to highlight with, or null when Prism has none. */
function grammarFor(language: string): string | null {
  const name = ALIASES[language.toLowerCase()] ?? language.toLowerCase();
  return name in Prism.languages ? name : null;
}

/**
 * How much source is worth colouring.
 *
 * Prism tokenises synchronously on the main thread, and the file viewer serves
 * whole files rather than the handful of lines a reply quotes — up to
 * `TEXT_CAP`, which is two megabytes. Past this the tab would lock up while
 * something nobody is going to read gets syntax-highlighted, so it renders as
 * plain monospace instead. A hundred thousand characters is a couple of
 * thousand lines, which is far more than anybody reads in a pane.
 */
const HIGHLIGHT_CAP = 100_000;

/**
 * Light and dark.
 *
 * Picked from the resolved theme rather than from a `dark:` class, because
 * these are inline styles — Prism themes are JavaScript objects, which is most
 * of why this highlighter was chosen over one that ships stylesheets.
 */
function themeFor(resolved: "light" | "dark"): PrismTheme {
  return resolved === "dark" ? themes.vsDark : themes.github;
}

type Wrapper = ComponentType<{
  className?: string;
  style?: CSSProperties;
  children?: ReactNode;
}>;

/**
 * Source, coloured if that is possible and plain if it is not.
 *
 * **One implementation for both places code is read**: a fenced block in a
 * reply, and a whole file in the viewer. They differ only in what wraps the
 * text — the markdown slots must render through the `Pre`/`Code` the library
 * hands them, and the viewer supplies its own — so that is the one thing this
 * takes as an argument.
 */
export const HighlightedCode: FC<{
  code: string;
  /** A fence tag, a language name, or a bare file extension. */
  language?: string;
  className?: string;
  /**
   * Let whatever is behind this show through.
   *
   * **For a caller that has already drawn the surface.** A fenced block in a
   * reply is its own box and wants the theme's background; the file viewer
   * hands this a bordered frame that is already the right colour, and painting
   * a second background inside it leaves a lighter panel floating in a darker
   * one. The token colours are unaffected either way, so contrast is not what
   * this trades.
   */
  transparent?: boolean;
  Pre?: Wrapper;
  Code?: Wrapper;
}> = ({
  code,
  language,
  className,
  transparent = false,
  Pre = "pre" as never,
  Code = "code" as never,
}) => {
  const resolved = useResolvedTheme();
  const grammar = language ? grammarFor(language) : null;

  if (!grammar || code.length > HIGHLIGHT_CAP) {
    return (
      <Pre className={className}>
        <Code>{code}</Code>
      </Pre>
    );
  }

  return (
    <Highlight theme={themeFor(resolved)} code={code} language={grammar}>
      {({ style, tokens, getLineProps, getTokenProps }) => (
        // The theme's own colours, over the stylesheet's. Everything else about
        // the box — border, radius, padding, scroll — belongs to whoever
        // wrapped this, so a highlighted block and a plain one are the same
        // shape.
        <Pre
          style={
            transparent ? { ...style, backgroundColor: undefined } : style
          }
          className={className}
        >
          <Code>
            {tokens.map((line, index) => (
              // `key` on the index deliberately: these are lines of one
              // immutable string, and they have no identity beyond position.
              <span key={index} {...getLineProps({ line })} className="block">
                {line.map((token, at) => (
                  <span key={at} {...getTokenProps({ token })} />
                ))}
              </span>
            ))}
          </Code>
        </Pre>
      )}
    </Highlight>
  );
};

export const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);

  // A fence that declared nothing arrives here as `"unknown"`. The header still
  // earns its place — the copy button is the reason most people look at it —
  // but naming the language "unknown" is a label pretending to be information.
  const named = language && language !== "unknown" ? language : null;

  const copy = () => {
    void navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), COPIED_MS);
    });
  };

  return (
    <div
      data-slot="code-header"
      className="bg-muted/70 text-muted-foreground -mb-px flex h-9 items-center justify-between rounded-t-(--radius-md) border px-3 font-mono text-xs"
    >
      <span className="truncate">{named}</span>
      <TooltipIconButton
        tooltip={copied ? "Copied" : "Copy code"}
        side="left"
        className="-me-1.5 size-7"
        onClick={copy}
      >
        {copied ? (
          <CheckIcon className="size-3.5" />
        ) : (
          <CopyIcon className="size-3.5" />
        )}
      </TooltipIconButton>
    </div>
  );
};

/**
 * The markdown slot. A language Prism does not know keeps the header, the frame
 * and the copy button, and only loses the colours — the honest picture of "no
 * grammar for this" rather than a failure.
 */
export const SyntaxHighlighter: FC<SyntaxHighlighterProps> = ({
  components: { Pre, Code },
  language,
  code,
}) => (
  <HighlightedCode
    code={code}
    language={language}
    className="rounded-t-none"
    Pre={Pre as Wrapper}
    Code={Code as Wrapper}
  />
);
