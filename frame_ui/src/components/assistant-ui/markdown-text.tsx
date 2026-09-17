import {
  MarkdownTextPrimitive,
  unstable_memoizeMarkdownComponents as memoizeMarkdownComponents,
} from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";

import { CodeHeader, SyntaxHighlighter } from "@/components/assistant-ui/code-block";

/**
 * Memoised, and it matters here more than anywhere else.
 *
 * A reply's markdown is re-parsed on every streamed token, so without this the
 * finished code blocks above the cursor are re-tokenised and re-highlighted
 * once per character of the block still being written.
 */
const components = memoizeMarkdownComponents({ CodeHeader, SyntaxHighlighter });

export function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm]}
      components={components}
      className="aui-md"
    />
  );
}
