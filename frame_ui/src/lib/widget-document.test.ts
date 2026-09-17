/**
 * The style contract, checked rather than trusted.
 *
 * `WIDGET_STYLE.md` and `templates/widget_template.html` both tell an author —
 * often an agent, which cannot go and look — which tokens exist. A name in
 * either that the frame does not actually send is worse than no name at all:
 * `var(--sb-text)` resolves to nothing, which makes the *whole declaration*
 * invalid and drops it, with no error anywhere. That is not hypothetical. Both
 * documents promised nine tokens the frame never sent, and `--sb-font` was
 * missing from the other direction — declared in Tailwind's `@theme inline`,
 * which emits no custom property to forward — so every widget rendered in the
 * browser's serif default and looked like an authoring mistake.
 *
 * Three documents describing one table is the arrangement; this is what keeps
 * them the same table.
 */

import { readFileSync } from "node:fs";
import { expect, it } from "vitest";

import { WIDGET_TOKENS } from "./widget-document";

/** Every `--sb-*` a document names, in prose or in code. */
function tokensNamed(path: string): string[] {
  const text = readFileSync(new URL(path, import.meta.url), "utf8");
  return [...new Set(text.match(/--sb-[a-z0-9-]+/g) ?? [])];
}

const promised = new Set(WIDGET_TOKENS);

it.each([
  ["WIDGET_STYLE.md", "../../WIDGET_STYLE.md"],
  // The kernel's template, which is the authoring contract an agent is handed.
  // It lives a folder up because a widget is not part of this app; it is a file
  // in one of the kernel's trees that this app knows how to mount.
  ["widget_template.html", "../../../templates/widget_template.html"],
])("%s names only tokens the frame sends", (_name, path) => {
  const unsent = tokensNamed(path).filter((token) => !promised.has(token));
  expect(unsent).toEqual([]);
});

it("sends the tokens a widget cannot do without", () => {
  // Not the whole set — that would be a second copy of the list. These are the
  // ones whose absence is invisible until somebody looks at a screenshot: the
  // font (serif everywhere), the surface (a hole cut in the panel) and the unit
  // of room (everything flush against the edge).
  for (const token of ["--sb-font", "--sb-bg", "--sb-fg", "--sb-space"]) {
    expect(promised).toContain(token);
  }
});
