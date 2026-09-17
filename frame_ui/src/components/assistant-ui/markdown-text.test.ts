/// <reference types="node" />
/* `types` in `tsconfig.app.json` is `vite/client` alone, which is right for
   everything that ships — this reference opts one test file into Node’s, and
   nothing else in `src` can reach the filesystem by accident. Reading the file
   rather than importing it: vitest hands back an empty string for a CSS import,
   `?raw` included, so an import here would assert nothing and always pass. */

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const DOT_CSS = "node_modules/@assistant-ui/react-markdown/styles/dot.css";

/**
 * The seam `AssistantMessage` turns the streaming cursor off through.
 *
 * `styles/dot.css` draws the pulsing dot at the end of a running `.aui-md` as
 * `content: var(--aui-content)`, and going through a custom property is the
 * only reason that indirection exists — so setting the property to `none`
 * higher up is how a message declines the cursor. That is an undocumented
 * detail of somebody else’s stylesheet, and if a release ever inlines the
 * glyph the suppression stops working *silently*: the dot returns and pulses
 * beside "Working" again, which is the bug this was written for.
 *
 * So the seam is asserted rather than assumed. This failing after an upgrade
 * means the cursor needs a new off switch, not that the test is wrong.
 */
describe("the markdown streaming cursor", () => {
  const css = readFileSync(DOT_CSS, "utf8");

  it("still draws itself through --aui-content", () => {
    expect(css).toContain("--aui-content");
    expect(css).toMatch(/content:\s*var\(--aui-content\)/);
  });

  it("still keys itself on the part’s own running status", () => {
    expect(css).toContain('.aui-md[data-status="running"]');
  });
});
