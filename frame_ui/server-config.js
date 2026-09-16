/**
 * Second Brain's own config, read from disk by the dev server.
 *
 * **Node-only.** Nothing here reaches the browser, and that is the point: the
 * bearer token lives in this file's world and never in the bundle. The page
 * talks to its own origin, the dev server adds the credential on the hop to
 * Second Brain, and no browser ever holds it.
 *
 * `http_client_url` and `secret_http_token` are kernel settings, so both are in
 * `config.json` beside everything else `/config` edits. Moving this UI to a
 * Tailscale address is therefore one line in that file and a dev-server
 * restart — nothing to copy, nothing to keep in step.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * Where Second Brain keeps its data.
 *
 * **Mirrored from `paths.py`, which is the source of truth.** Shelling out to
 * Python would be authoritative but would make `npm run dev` depend on the
 * server's interpreter; these are three branches that have not moved in the
 * life of the project, and getting them wrong fails loudly (no config found,
 * printed with the path it looked in) rather than quietly.
 *
 * `SB_DATA_DIR` overrides, for a checkout pointed somewhere unusual.
 */
export function dataDir() {
  if (process.env.SB_DATA_DIR) return process.env.SB_DATA_DIR;
  if (process.platform === "win32") {
    return path.join(process.env.LOCALAPPDATA || "", "Second Brain");
  }
  if (process.platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support", "Second Brain");
  }
  const xdg = process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share");
  return path.join(xdg, "Second Brain");
}

export const configPath = () => path.join(dataDir(), "config.json");

/**
 * The token, re-read when `config.json` changes.
 *
 * **Deliberately not read once at startup**, which is what it used to do and
 * which fails in a way nobody can diagnose from the browser. The token is
 * minted by the kernel at boot and can move between files during an upgrade,
 * so a dev server started at the wrong moment holds `""` for its whole life
 * and proxies without a credential — every Request comes back `unauthorized`
 * while the page itself loads perfectly. The value is a *fact about a file*,
 * so it is read from the file.
 *
 * `mtimeMs` rather than a timer: stat is cheap, a proxied request is not
 * frequent enough for it to matter, and an interval would still be wrong for
 * however long it happened to be.
 */
let cached = { mtime: -1, config: {} };

export function config() {
  try {
    const stamp = fs.statSync(configPath()).mtimeMs;
    if (stamp !== cached.mtime) {
      cached = {
        mtime: stamp,
        config: JSON.parse(fs.readFileSync(configPath(), "utf-8")),
      };
    }
  } catch {
    /* Missing or unparseable. Reported by the caller, with the path; not
       fatal, because a dev server that starts and says 401 tells you more
       than one that refuses to start. */
  }
  return cached.config;
}

/** The bearer credential, as of now. */
export function token() {
  return (process.env.VITE_SB_TOKEN || config().secret_http_token || "").trim();
}

/**
 * `{url, configPath, found}` — where to proxy.
 *
 * The URL *is* read once, unlike the token: it is the proxy's `target`, which
 * Vite resolves when the server is configured and cannot be changed under a
 * running one. Changing `http_client_url` therefore needs a restart, and says
 * so in `/config`.
 *
 * Environment wins when it is set, because an explicit override that is
 * silently ignored is worse than no override at all. `.env.local` ships with
 * neither, so in the ordinary case `config.json` is the only answer.
 */
export function backend() {
  const settings = config();
  return {
    url:
      process.env.VITE_SB_URL ||
      settings.http_client_url ||
      "http://127.0.0.1:8787",
    configPath: configPath(),
    found: Object.keys(settings).length > 0,
  };
}
