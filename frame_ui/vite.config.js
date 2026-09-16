import { defineConfig, loadEnv } from "vite";
import { backend, token } from "./server-config.js";

export default defineConfig(({ mode }) => {
  // `loadEnv` rather than `import.meta.env`: this file runs in Node, before any
  // of that exists.
  const env = loadEnv(mode, import.meta.dirname, "VITE_");
  const { url: target, configPath, found } = backend();

  if (!found) {
    console.warn(
      `\n  No Second Brain config at ${configPath}.\n` +
        `  Falling back to ${target} with no token, which answers 401 to\n` +
        `  everything. Start Second Brain once to create it.\n`,
    );
  } else {
    // Not warning about an empty token here: it is re-read per request, so
    // one that is missing right now is one that starts working the moment
    // the kernel mints it, with nothing to restart.
    console.log(`\n  Second Brain: ${target}\n`);
  }

  return {
    server: {
      port: Number(env.VITE_UI_PORT) || 5174,
      strictPort: true,

      /**
       * Second Brain's endpoints, served from this app's own origin.
       *
       * **This is what keeps both CORS and the credential out of the page.**
       * The browser only ever talks to the dev server; the dev server talks to
       * `http_client_url` and adds the bearer header itself. So the token is
       * never in the bundle, and moving the backend to a Tailscale host is one
       * line in `config.json` with nothing in the page to update.
       *
       * The alternative — pointing a browser straight at the server — means
       * `http_allowed_origins`, which is a sharper edge than it looks: the
       * server echoes that setting into `Access-Control-Allow-Origin`
       * verbatim, so a trailing slash or `localhost` where the browser says
       * `127.0.0.1` fails the match, and a failed preflight explains almost
       * nothing.
       */
      proxy: {
        "/sdk": { target, changeOrigin: true, configure: authorize },
        // Host files, as bytes with a `Content-Type`. `Range` and `206` pass
        // through untouched, which is what lets a `<video>` seek.
        "/files": { target, changeOrigin: true, configure: authorize },
        "/events": {
          target,
          changeOrigin: true,
          configure: (proxy) => {
            authorize(proxy);
            // Server-sent events must not be buffered, or the stream only
            // arrives once it ends — which for a live render stream is never.
            // Two things are needed: no compression on the way in, and the
            // response headers pushed out the moment they arrive rather than
            // held until the first body chunk. Without the flush,
            // `EventSource` never even opens.
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

  /**
   * Add the bearer header on the hop the browser cannot see.
   *
   * `token()` per request, not a value captured when the server started: the
   * kernel mints the token at boot, so a dev server that happened to start
   * first would otherwise proxy without a credential for the rest of its
   * life — and the symptom is a page that loads perfectly and answers
   * `unauthorized` to everything, which looks like a bug in the app.
   */
  function authorize(proxy) {
    proxy.on("proxyReq", (request) => {
      const bearer = token();
      if (bearer) request.setHeader("Authorization", `Bearer ${bearer}`);
    });
  }
});
