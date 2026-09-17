/**
 * The wall a panel does not have.
 *
 * A widget's contract holds because an iframe *can only* post a message — the
 * machine enforces it and no discipline is required. A panel runs in this page,
 * so its contract holds only as long as nobody takes a shortcut, and the first
 * shortcut will look entirely reasonable: one import of `config.js` to read a
 * size, one import of `frame.js` to nudge a slot. After two of those a panel is
 * not an independent piece of machinery, it is part of the frame with a folder
 * of its own — which is the nested mess this arrangement exists to avoid.
 *
 * So the import graph is the boundary. Everything under `src/panels/` may
 * import from inside `src/panels/`, from `client.js` (the one Request route,
 * which is what `brain.call` is on the other side), and from `node_modules`.
 * Nothing else. A panel that needs something more needs a **wider host**, which
 * is a change to `panels/host.js` and a line in the registry — visible, in one
 * place, and reviewable. An import is none of those things.
 *
 * Run with `npm run check`. It is deliberately a script rather than a comment
 * in a style guide: the same argument `test_kernel_boundary.py` makes for the
 * kernel, at the scale this app justifies.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "..", "src");
const panels = join(src, "panels");

/** What a panel may reach outside its own folder, by resolved path. */
const ALLOWED = new Set([join(src, "client.js")]);

/** Every `import`/`export … from` specifier, and `import(...)`. Regex rather
 *  than a parser because the question is only ever "what string follows from",
 *  and a dependency to answer it would be heavier than the rule it checks. */
const SPECIFIER = /(?:^|\n)\s*(?:import|export)[\s\S]*?from\s*["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']\s*\)/g;

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (entry.endsWith(".js") || entry.endsWith(".jsx")) out.push(full);
  }
  return out;
}

const problems = [];
for (const file of walk(panels)) {
  const source = readFileSync(file, "utf8");
  for (const match of source.matchAll(SPECIFIER)) {
    const spec = match[1] || match[2];
    // A bare specifier is a package. Panels may use libraries — that is the
    // whole reason chat is a panel rather than a widget.
    if (!spec.startsWith(".")) continue;
    const target = resolve(dirname(file), spec);
    if (target.startsWith(panels)) continue;
    if (ALLOWED.has(target) || ALLOWED.has(target + ".js")) continue;
    problems.push(
      `${relative(src, file)} imports ${spec}\n` +
      `    → resolves outside src/panels/. A panel reaches the app only through\n` +
      `      the host it is handed. Widen panels/host.js instead.`);
  }
}

if (problems.length) {
  console.error("panel boundary violated:\n\n" + problems.join("\n\n") + "\n");
  process.exit(1);
}
console.log("panel boundary intact");
