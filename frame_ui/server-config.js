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

/**
 * `{url, token, source}` — where to proxy, and what to prove it with.
 *
 * Read once at startup rather than watched: both values change about as often
 * as a machine moves house, and a proxy that re-targets underneath a running
 * page would be harder to reason about than restarting the dev server.
 *
 * Environment wins when it is set, because an explicit override that is
 * silently ignored is worse than no override at all. `.env.local` ships with
 * neither, so in the ordinary case `config.json` is the only answer.
 */
export function backend() {
  const configPath = path.join(dataDir(), "config.json");
  let config = {};
  let found = false;
  try {
    config = JSON.parse(fs.readFileSync(configPath, "utf-8"));
    found = true;
  } catch {
    /* Reported by the caller, with the path. Not fatal: a dev server that
       refuses to start tells you less than one that starts and says 401. */
  }
  return {
    url:
      process.env.VITE_SB_URL ||
      config.http_client_url ||
      "http://127.0.0.1:8787",
    token: (process.env.VITE_SB_TOKEN || config.secret_http_token || "").trim(),
    configPath,
    found,
  };
}
