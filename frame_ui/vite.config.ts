import path from "node:path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { backend, token, uiHosts } from "./server-config.js";

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // `loadEnv` rather than `import.meta.env`: this file runs in Node, before any
  // of that exists.
  const env = loadEnv(mode, import.meta.dirname, "VITE_");
  const target = env.VITE_SB_URL || backend().url;
  const authorize: NonNullable<import("vite").ProxyOptions["configure"]> = (proxy) => {
    proxy.on("proxyReq", request => {
      const bearer = token();
      if (bearer) request.setHeader("Authorization", `Bearer ${bearer}`);
    });
  };

  return {
  plugins: [react(), tailwindcss()],

  /**
   * No DOM unless a file asks for one.
   *
   * The reducers are pure and want none — that is the whole reason they are
   * reducers, and running them in a jsdom would be slower for nothing. The
   * dialog is the opposite: what needs pinning about it is that Escape and its
   * corner button reach a real *cancel*, which is a fact about Radix's
   * dismissal wiring rather than about any function written here, and nothing
   * but a DOM can answer it. Those files say `@vitest-environment jsdom` at the
   * top, which keeps the requirement next to the code that has it.
   */
  test: { environment: "node" },

  resolve: {
    // Components import each other as "@/components/...", so the alias is not a
    // convenience here — without it those files do not resolve at all.
    // TypeScript is told the same thing in tsconfig.app.json; both are needed,
    // because Vite resolves at build time and tsc only type-checks.
    alias: { "@": path.resolve(import.meta.dirname, "./src") },
  },

  server: {
    port: Number(env.VITE_UI_PORT) || 5174,
    strictPort: true,

    /**
     * The names this server will answer to, beyond its own localhost.
     *
     * Vite refuses an unrecognised `Host` outright — a plain-text 403 that
     * names `vite.config.js` as the fix. Reaching this app through a gateway
     * (a Tailscale address, say) is exactly the case that produces one, and
     * the refusal is indistinguishable from a broken deployment: the page
     * never loads, and the kernel's own probe reports the bridge as refused
     * while pointing at a token that was never involved.
     *
     * The name comes from `ui_url`, which is already the setting that says
     * where this app lives, so there is nothing to keep in step. A checkout
     * with no gateway contributes none and keeps Vite's default.
     */
    allowedHosts: uiHosts(),

    /**
     * The Second Brain endpoints, served from this app's own origin.
     *
     * **This is what keeps CORS out of the picture entirely.** The alternative
     * is `http_allowed_origins`, and it is a sharper edge than it looks: the
     * server echoes that setting into `Access-Control-Allow-Origin` verbatim,
     * so a trailing slash or `localhost` where the browser says `127.0.0.1`
     * fails the match — and a failed preflight tells you almost nothing.
     *
     * Proxying also makes development match production. Caddy serves the build
     * and proxies these same paths there, so "the browser always talks to its
     * own origin" is one rule that holds in both.
     */
    proxy: {
      "/sdk": { target, changeOrigin: true, configure: authorize },
      // Host files, as bytes with a `Content-Type`. Nothing special is needed
      // here beyond existing: `Range` and `206` pass through untouched, which
      // is what lets a `<video>` seek instead of downloading everything before
      // the point you clicked. Without this entry `fileUrl` resolves against
      // Vite, which has no such route and answers the index page — an `<img>`
      // that fails for reasons no status code explains.
      "/files": { target, changeOrigin: true, configure: authorize },
      "/events": {
        target,
        changeOrigin: true,
        // Server-sent events must not be buffered, or the stream only arrives
        // once it ends — which for a live render stream is never. Two things
        // are needed: no compression on the way in, and the response headers
        // pushed out the moment they arrive rather than held until the first
        // body chunk. Without the flush, `EventSource` never even opens.
        configure: (proxy) => {
          authorize(proxy, {});
          proxy.on("proxyReq", (request) => {
            request.setHeader("Accept-Encoding", "identity");
          });
          // `setImmediate`, not a direct call: this listener runs *before* the
          // proxy has copied the upstream headers onto the response, so
          // flushing here sends them empty — and a stream without
          // `Content-Type: text/event-stream` is one `EventSource` refuses,
          // which presents as a client stuck reconnecting forever. One tick
          // later the headers are in place and the flush does what it says.
          proxy.on("proxyRes", (_proxyRes, _request, response) => {
            setImmediate(() => response.flushHeaders?.());
          });
        },
      },
    },
  },
  };
});
