/**
 * The bridge to Second Brain: `POST /sdk/<request.type>` out, `GET /events` in.
 *
 * Second Brain's protocol is Requests. There is no REST surface beside this and
 * no database — whatever is not expressible as a Request cannot be done from
 * here at all. The body is a Request's arguments as JSON, the answer is its
 * `Result` as JSON, and the other half — the renders that arrive unasked — is
 * one SSE stream of `{kind, session_key, payload}` frames.
 *
 * See `docs/HTTP_PROTOCOL.md` in the server repo for the twelve render kinds
 * and the rules a working demo cannot show.
 */

/**
 * **There is no token in this file, and that is deliberate.**
 *
 * Second Brain's HTTP frontend wants `Authorization: Bearer <secret_http_token>`
 * on everything. The page never sends one: it talks only to its own origin, and
 * the dev server adds the header on the hop the browser cannot see
 * (`vite.config.ts`, reading `config.json` through `server-config.js`). So the
 * credential is never in the bundle, never in a query string, and never in
 * whatever the browser caches or a devtools tab shows a guest.
 *
 * It also means an `EventSource` needs no special case. It cannot set headers,
 * which is why the usual arrangement puts the token in the query string — and
 * a token in a URL is one that ends up in logs.
 *
 * A 401 here is therefore never this file's problem: it is an empty or stale
 * `secret_http_token` in `config.json`, or a dev server started before it was
 * minted.
 */

/**
 * Which session this browser is.
 *
 * `?thread=main` selects the session keyed `http:main`. Two threads are two
 * independent conversations, and this is **the only way** a client names a
 * session — a `session_key`, `token` or `key` in a request body is stripped and
 * replaced by the server, because identity is the server's to state rather than
 * ours to claim. So never put one in `args`; it is silently overwritten.
 *
 * The page's own `?thread=` wins over the configured default, which is what
 * makes a second window a second conversation rather than a fight over the
 * first. **Only one stream may be open per thread** — a second `GET /events`
 * replaces the first — so two tabs on one thread take turns being connected.
 */
// Share the app's selected identity, including its browser-local fallback.
// Do not open another stream here: the React runtime owns the active stream.
import { THREAD } from "./lib/client";
export { THREAD };

/** A URL against the server, with the thread already attached. */
export function serverUrl(path) {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("thread", THREAD);
  return url;
}

/** A Request that came back with something other than 200. `code` is the
 *  kernel's own `ERROR_*` token, so a caller can branch without parsing prose —
 *  a `403 approval_declined` means the person said no, or nobody was there to
 *  ask, and both are things to show rather than retry. */
export class RequestFailed extends Error {
  constructor(type, status, body) {
    super(body?.error || `${type} failed (${status})`);
    this.name = "RequestFailed";
    this.type = type;
    this.status = status;
    this.code = body?.code || "";
  }
}

/** Run one Request and answer with its data. */
export async function call(type, args = {}) {
  const response = await fetch(serverUrl(`/sdk/${type}`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
  let body = null;
  try {
    body = await response.json();
  } catch {
    /* A non-JSON body is a transport problem; the status says enough. */
  }
  if (!response.ok || body?.ok === false) {
    throw new RequestFailed(type, response.status, body);
  }
  return body?.data;
}

/**
 * The render stream.
 *
 * **Opening it is the attendance signal.** The server reads a live stream as
 * "somebody is watching", and attendance is what decides whether an unsafe
 * Request raises an approval dialog or is refused outright — so closing the tab
 * is what switches that authority back off.
 *
 * `EventSource`, deliberately: it reconnects on its own and sends back the last
 * `id:` it saw as `Last-Event-ID`, which the server replays from.
 *
 * Returns a function that closes the stream.
 */
export function openStream(onFrame, onState) {
  const source = new EventSource(serverUrl("/events"));
  source.onopen = () => onState?.("open");
  source.onerror = () => onState?.("retrying");
  source.onmessage = (event) => {
    try {
      onFrame(JSON.parse(event.data));
    } catch {
      /* A frame we cannot parse is not worth taking the stream down for. */
    }
  };
  return () => source.close();
}
