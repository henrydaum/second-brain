import type { ComponentPropsWithoutRef, FC } from "react";
import { TextMessagePartProvider } from "@assistant-ui/react";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";

import {
  CodeHeader,
  SyntaxHighlighter,
} from "@/components/assistant-ui/code-block";
import { cn } from "@/lib/utils";

type MarkdownElementProps<Tag extends keyof React.JSX.IntrinsicElements> =
  ComponentPropsWithoutRef<Tag> & { node?: unknown };

/** Tables need their own surface in Settings: command output is operational
 * data, not prose, and should scan like an ordinary application table. */
const CommandTable: FC<MarkdownElementProps<"table">> = ({ node: _node, ...props }) => (
  <div className="my-4 min-w-0 w-full max-w-full overflow-x-auto overscroll-x-contain rounded-lg border [contain:inline-size] [-webkit-overflow-scrolling:touch]">
    {/* `table` is Tailwind's `display: table` utility, and it is here on
        purpose. The chat's `.aui-md` stylesheet makes bare tables `display:
        block` so a wide one scrolls inside itself; this table already has a
        scrolling wrapper, so it needs to stay a real table. Utilities outrank
        that stylesheet's layer, which is what lets one word say so. */}
    <table
      className="table min-w-full w-max max-w-none overflow-visible border-collapse text-sm"
      {...props}
    />
  </div>
);

const TableHead: FC<MarkdownElementProps<"thead">> = ({ node: _node, ...props }) => (
  <thead className="bg-muted/60 text-left" {...props} />
);

const TableHeader: FC<MarkdownElementProps<"th">> = ({ node: _node, ...props }) => (
  <th
    className="text-foreground border-b px-3 py-2.5 text-xs font-semibold tracking-wide whitespace-nowrap"
    {...props}
  />
);

const TableRow: FC<MarkdownElementProps<"tr">> = ({ node: _node, ...props }) => (
  <tr className="border-b last:border-b-0 hover:bg-muted/25" {...props} />
);

const TableCell: FC<MarkdownElementProps<"td">> = ({ node: _node, ...props }) => (
  <td className="px-3 py-2.5 align-top" {...props} />
);

/** Backquotes are the protocol's detail-card convention. Render that meaning
 * directly instead of leaving it as an indented strip of chat markdown. */
const DetailCard: FC<MarkdownElementProps<"blockquote">> = ({
  node: _node,
  ...props
}) => (
  <aside
    className="bg-muted/35 my-4 rounded-lg border px-4 py-3 text-sm [&_p]:my-0"
    {...props}
  />
);

const markdownComponents = {
  table: CommandTable,
  thead: TableHead,
  th: TableHeader,
  tr: TableRow,
  td: TableCell,
  blockquote: DetailCard,
  // Not memoised, unlike the chat's: command output arrives whole rather than a
  // token at a time, so there is no re-parse to protect against.
  CodeHeader,
  SyntaxHighlighter,
};

export const CommandMarkdown: FC<{
  text: string;
  className?: string;
}> = ({ text, className }) => (
  <TextMessagePartProvider text={text}>
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      components={markdownComponents}
      smooth={false}
      className={cn(
        "aui-md min-w-0 max-w-full text-sm [&_h1]:text-xl [&_h2]:text-lg [&_h3]:text-base [&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold",
        className,
      )}
    />
  </TextMessagePartProvider>
);

/** One command may emit several message frames. Give each logical result its
 * own renderer without wrapping the whole workflow in another card. */
export const CommandOutput: FC<{ output: string[] }> = ({ output }) => (
  <div className="min-w-0 max-w-full space-y-5">
    {output.map((text, index) => (
      <section
        key={`${index}-${text.slice(0, 24)}`}
        className={cn("min-w-0 max-w-full", index > 0 && "border-t pt-5")}
      >
        <CommandMarkdown text={text} />
      </section>
    ))}
  </div>
);
