import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // `loadEnv` rather than `import.meta.env`: this file runs in Node, before any
  // of that exists.
  const env = loadEnv(mode, import.meta.dirname, "VITE_");
  const target = env.VITE_SB_URL || "http://127.0.0.1:8787";

  return {
    server: {
      port: Number(env.VITE_UI_PORT) || 5174,
      strictPort: true,

      /**
       * Second Brain's endpoints, served from this app's own origin.
       *
       * **This is what keeps CORS out of the picture entirely**, and it is also
       * what makes the backend URL a one-line change: the browser only ever
       * talks to the dev server, and the dev server talks to whatever
       * `VITE_SB_URL` names — loopback now, a Tailscale host later, with
       * nothing in the page to update.
       *
       * The alternative is `http_allowed_origins`, and it is a sharper edge
       * than it looks: the server echoes that setting into
       * `Access-Control-Allow-Origin` verbatim, so a trailing slash or
       * `localhost` where the browser says `127.0.0.1` fails the match — and a
       * failed preflight explains almost nothing.
       */
      proxy: {
        "/sdk": { target, changeOrigin: true },
        // Host files, as bytes with a `Content-Type`. `Range` and `206` pass
        // through untouched, which is what lets a `<video>` seek.
        "/files": { target, changeOrigin: true },
        "/events": {
          target,
          changeOrigin: true,
          // Server-sent events must not be buffered, or the stream only
          // arrives once it ends — which for a live render stream is never.
          // Two things are needed: no compression on the way in, and the
          // response headers pushed out the moment they arrive rather than
          // held until the first body chunk. Without the flush, `EventSource`
          // never even opens.
          configure: (proxy) => {
            proxy.on("proxyReq", (request) => {
              request.setHeader("Accept-Encoding", "identity");
            });
            // `setImmediate`, not a direct call: this listener runs *before*
            // the proxy has copied the upstream headers onto the response, so
            // flushing here sends them empty — and a stream without
            // `Content-Type: text/event-stream` is one `EventSource` refuses,
            // which presents as a client stuck reconnecting forever.
            proxy.on("proxyRes", (_proxyRes, _request, response) => {
              setImmediate(() => response.flushHeaders?.());
            });
          },
        },
      },
    },
  };
});
