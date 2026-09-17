/**
 * The outbound half of the bridge: `POST /sdk/<request.type>`.
 *
 * Second Brain's protocol is Requests, and this file is how one is made; the
 * other half, the answers that arrive unasked, is `events.ts`. There is no REST
 * surface beside this and no database — whatever is not expressible as a
 * Request cannot be done from here at all.
 *
 * The body is a Request's arguments as JSON and the answer is its result. That
 * is the whole protocol; the interesting parts are all in what the failures
 * mean.
 *
 * `fileUrl` at the bottom is the one exception, and it is not really a third
 * protocol: `GET /files` reads through the same `fs.read_bytes` as everything
 * else, with the same policy check and the same ledger row. What it adds is an
 * HTTP body with a real `Content-Type`, which is the one thing a Request cannot
 * give an `<img>` or a `<video>`.
 */

/**
 * Where the server is, and what proves we are allowed to talk to it.
 *
 * **The browser always talks to its own origin.** In development Vite proxies
 * `/sdk` and `/events` through to the server (see `vite.config.ts`); in
 * production Caddy serves the build and proxies the same paths. One rule for
 * both, and CORS never enters into it — which
 * matters because the server echoes `http_allowed_origins` into
 * `Access-Control-Allow-Origin` verbatim, and a mismatch as small as a trailing
 * slash fails a preflight that then explains almost nothing.
 *
 * Vite reads the kernel configuration and adds the bearer header upstream.
 * Production uses the loopback gateway. Neither needs a browser credential.
 */

/**
 * Which session this browser is.
 *
 * `?thread=main` selects the session keyed `http:main`. Two threads are two
 * independent conversations, and this is **the only way** the client names a
 * session — a `session_key`, `token` or `key` in a request body is stripped and
 * replaced by the server, because identity is the server's to state, not ours
 * to claim. So never put one in `args`; it will be silently overwritten.
 *
 * The page's own `?thread=` wins over the configured default, which is what
 * makes a second window a second conversation rather than a fight over the
 * first. **Only one stream may be open per thread** — a second `GET /events`
 * replaces the first — so two tabs on one thread take turns being connected and
 * both sit there reconnecting.
 */
import { browserStorage, selectThread } from "@/lib/thread";

export const THREAD = selectThread({
  search: window.location.search,
  configured: import.meta.env.VITE_SB_THREAD,
  storage: browserStorage(),
  randomUUID: () => crypto.randomUUID(),
});

/** Authentication belongs to the same-origin proxy, including in development. */
export const authHeaders = (): Record<string, string> => ({});

/** Build a URL against the server, with the thread already attached. */
export function serverUrl(path: string): URL {
  const url = new URL(path, window.location.origin);
  url.searchParams.set("thread", THREAD);
  return url;
}

/**
 * A path on the host, as a URL the browser can actually fetch.
 *
 * Everything the agent hands over — an `attachments` frame, a `paths` entry in
 * a ledger row — is a **filesystem path on the host**, and there is nothing to
 * link to. This is the route that turns one into a link.
 *
 * The same-origin proxy authenticates media requests too, so URLs contain no
 * credential and work directly in images, audio, and video elements.
 *
 * Prefer handing this to an element over fetching it. The route honours `Range`
 * and a large file answers `206` even when nothing asked it to, so the browser's
 * own loader gets the whole file right where a naive `fetch` gets a fragment.
 * `fetchWhole` in `lib/files.ts` is the fetching version, for the renderers that
 * genuinely need the bytes in hand.
 *
 * **The query is built by hand, and it has to be.** `URLSearchParams` — which
 * is what `serverUrl` uses, and what the obvious version of this function used
 * — serialises as `application/x-www-form-urlencoded`, where a space becomes
 * `+` rather than `%20`. The server percent-decodes, so every path under
 * `AppData\Local\Second Brain\` arrived as `Second+Brain` and answered `404`.
 * `encodeURIComponent` is the encoding this route is specified in.
 */
export function fileUrl(hostPath: string): string {
  const query = [
    `thread=${encodeURIComponent(THREAD)}`,
    `path=${encodeURIComponent(hostPath)}`,
  ].join("&");
  return new URL(`/files?${query}`, window.location.origin).toString();
}

/**
 * A Request that came back with something other than 200.
 *
 * **A refusal is an ordinary answer, not a transport failure.** The kernel
 * classifies every Request the same way it classifies a tool call, and a
 * `403 approval_declined` means either "the person said no" or "nobody was
 * there to ask" — both of which are things to show, not to retry. The `code` is
 * the kernel's own, so callers can distinguish those cases without parsing
 * prose.
 */
export class RequestFailed extends Error {
  /** The Request type that failed, e.g. `"conv.load"`. */
  readonly type: string;
  /** HTTP status: 400 bad args, 403 refused, 404 missing, 499 cancelled,
   *  503 subsystem absent, 504 timed out, 500 anything else. */
  readonly status: number;
  /** The kernel's error code, e.g. `"approval_declined"`. May be empty. */
  readonly code: string;

  // Written out rather than declared as constructor parameter properties: this
  // project compiles with `erasableSyntaxOnly`, which allows only TypeScript
  // syntax that erases to nothing, and parameter properties emit real code.
  constructor(type: string, status: number, code: string, message: string) {
    super(message);
    this.name = "RequestFailed";
    this.type = type;
    this.status = status;
    this.code = code;
  }

  /** True when this was a policy refusal rather than a broken call. The two
   *  causes are indistinguishable from out here on purpose: the difference is
   *  whether anyone was watching, which `attended` on the session answers. */
  get isDeclined() {
    return this.code === "approval_declined";
  }

  /**
   * The session now belongs to a different frontend, and nothing this client
   * does can take it back.
   *
   * Every Request runs through `frontend.act`, which refuses a session another
   * frontend owns — so once this happens the thread is finished, not merely
   * failing. It is worth naming because the recovery is not "try again": it is
   * a new thread, or a server restart, and a person staring at a raw
   * `frontend.act: denied` has no way to know that.
   */
  get isSessionTaken() {
    return /belongs to the .* frontend/.test(this.message);
  }
}

/**
 * Make one Request and hand back its result.
 *
 * **This call can legitimately take as long as a person takes.** Anything
 * consequential — `config.write`, `conv.delete`, a gated `command.call` — does
 * not simply fail: it raises a real approval dialog, which arrives on the
 * *event stream* while this POST is still open, and the POST completes once the
 * dialog is answered. So there is deliberately no timeout here, and callers
 * must not serialise these behind one another — a request blocked on a human
 * would stall every request queued behind it, including the `frontend.resolve`
 * that would unblock it.
 *
 * That also means the event stream must already be open before the first
 * interesting Request goes out. Attendance follows the stream; without one,
 * unsafe Requests come back `403 approval_declined` with nobody having been
 * asked.
 */
export async function sdk<T = unknown>(
  type: string,
  args: Record<string, unknown> = {},
): Promise<T> {
  const response = await fetch(serverUrl(`/sdk/${type}`), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify(args),
  });

  // A body is not guaranteed on every status, so a parse failure must not
  // become the error the caller sees — the status is the real information.
  const body = (await response.json().catch(() => ({}))) as {
    data?: T;
    error?: string;
    code?: string;
  };

  if (!response.ok) {
    throw new RequestFailed(
      type,
      response.status,
      body.code ?? "",
      body.error ?? `${type} failed with ${response.status}`,
    );
  }

  return body.data as T;
}
