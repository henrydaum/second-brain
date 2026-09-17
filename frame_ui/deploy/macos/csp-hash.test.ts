/**
 * The Caddyfile's script hash still describes the script in `index.html`.
 *
 * **This test is the only thing standing between an edit and a silent
 * regression in production.** The theme bootstrap has to run before the first
 * paint — that is the whole reason it is inline rather than a file — and the
 * production Content-Security-Policy therefore admits it by hash. Change one
 * character of that script and the hash no longer matches: the browser refuses
 * to run it, and every cold load of the app flashes white before going dark.
 * Nothing throws, nothing logs anywhere anybody looks, and the only symptom is
 * the exact flash the inline script was written to prevent.
 *
 * `build-release.sh` runs `npm test` before it activates a release, so a
 * mismatch stops the deployment rather than shipping.
 *
 * **Line endings are normalised, and that is load-bearing.** A browser hashes
 * the bytes it received. The Mac Mini builds from a checkout with LF endings,
 * while a Windows working copy has CRLF — so hashing this file verbatim would
 * compute one answer here and a different one on the machine that serves it.
 * `.gitattributes` pins `index.html` to LF for exactly this reason; normalising
 * again here means the test is right even in a working copy that predates it.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const read = (relative: string) =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8")
    // See the note above: the bytes the browser hashes end in LF.
    .replace(/\r\n/g, "\n");

/** Every `<script>` in the page that has no `src` — that is, every one the
 *  policy has to name explicitly. */
function inlineScripts(html: string): string[] {
  const scripts = html.matchAll(
    /<script(?![^>]*\ssrc=)[^>]*>([\s\S]*?)<\/script>/g,
  );
  return [...scripts].map((match) => match[1]);
}

const cspHash = (source: string) =>
  `sha256-${createHash("sha256").update(source, "utf8").digest("base64")}`;

describe("the production Content-Security-Policy", () => {
  const html = read("../../index.html");
  const caddyfile = read("./Caddyfile");

  it("admits the inline theme script by its current hash", () => {
    const [script] = inlineScripts(html);
    expect(script).toBeDefined();
    expect(caddyfile).toContain(`'${cspHash(script!)}'`);
  });

  /**
   * A second inline script would be blocked in production and work perfectly
   * everywhere else — `npm run dev` serves no policy at all, so the failure
   * only appears on the deployed Mac. Naming the count here is what turns that
   * into a test failure at the moment the script is added.
   */
  it("has exactly one inline script to admit", () => {
    expect(inlineScripts(html)).toHaveLength(1);
  });

  /** The policy exists at all, and still refuses the two things this app has
   *  no use for: being framed, and having its `<base>` rewritten. */
  it("keeps the directives the app depends on", () => {
    expect(caddyfile).toContain("frame-ancestors 'none'");
    expect(caddyfile).toContain("base-uri 'none'");
    expect(caddyfile).toContain("connect-src 'self'");
  });

  it("isolates App script permission from the UI policy", () => {
    const policies = [...caddyfile.matchAll(/header Content-Security-Policy "([^"]+)"/g)]
      .map(match => match[1]);
    const host = policies.find(policy => policy.startsWith("sandbox allow-scripts;"));
    expect(host).toContain("script-src 'unsafe-inline'");
    expect(host).toContain("connect-src 'none'");
    expect(host).not.toContain("allow-same-origin");
    expect(host).not.toContain("sha256-");
    expect(policies.at(-1)?.match(/script-src ([^;]+)/)?.[1]).not.toContain("unsafe-inline");
    expect(caddyfile).toContain("handle /html-app-host.html");
  });
});
