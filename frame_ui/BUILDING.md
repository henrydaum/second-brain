# Building this UI

Notes for whoever picks this up next, written at the point where the plumbing
works and none of the app exists. It is the things that cost something to find
out, not a plan — the plan is yours.

## Where things stand

`frame_ui/` is ~390 lines: [`src/client.js`](src/client.js) (the protocol, both
halves) and [`src/main.js`](src/main.js) (a page that proves the bridge and
dumps render frames). The client is meant to survive; `main.js` is a probe and
should be deleted the moment something real renders.

What works and is verified: the SSE stream, `POST /sdk/<type>`, the dev-server
proxy that adds the bearer token, the kernel starting the dev server and
announcing it. What does not exist: any rendering at all, any input, any state.

## The one resource that matters

**`Z:\My Code\Second Brain UI` is a finished, working client for this exact
protocol** — 27,000 lines of TypeScript, 57 components, against the same
fourteen render kinds. It is not legacy to be avoided; it is the reference
implementation, and most questions that come up here were answered there
already, usually with a comment saying why.

Its `docs/` is worth reading before designing anything:

| | |
|---|---|
| `CHAT_RENDERING.md` | how a transcript is actually assembled from frames |
| `NOTIFICATIONS.md` | the panel, and why persistence matters to it |
| `AGENT_FILE_ACTIVITY.md` | showing what the agent touched |
| `HTML_APPS.md` | rendering agent-authored HTML safely |
| `MACOS_DEPLOYMENT.md` | the Caddy gateway, which is the only production story that exists |

And its `src/lib/` holds the solved problems, each in its own file with tests:
`events.ts` (EventSource reconnection, `Last-Event-ID` replay, and the failure
modes browsers do *not* report), `thread.ts` (which session a browser is),
`history.ts` (`conv.read` returns a **page**, bounded by bytes — see its
`history-paging.test.ts`), `input-requests.ts`, `markdown.ts`,
`scroll-memory.ts`, `conversations.ts`, `ledger.ts`. Staged attachments live in
`src/runtime/staged-attachments.ts` instead, with the rest of the state.

Copying from it wholesale would recreate what you are replacing, which is
presumably not the point. Reading it before deciding is free.

## Protocol facts that are not guessable

Read [`docs/HTTP_PROTOCOL.md`](../docs/HTTP_PROTOCOL.md) in the server repo —
it is written for exactly this job, and
[`docs/http_reference_client.html`](../docs/http_reference_client.html) is a
working demo to check the bridge against when the client misbehaves. The parts
that bite:

- **The client names a session with `?thread=` and nothing else.** A
  `session_key`, `token` or `key` in a request body is stripped and replaced by
  the server — identity is its to state, not yours to claim. Never put one in
  `args`; it is silently overwritten.
- **One stream per thread.** A second `GET /events` replaces the first, so two
  tabs on one thread take turns being connected and both sit there
  reconnecting. `thread.ts` in the old UI is how that was made survivable.
- **Opening the stream is the attendance signal.** Attendance decides whether
  an unsafe Request raises an approval dialog or is refused outright, so a
  client that holds no stream quietly loses the ability to be asked anything.
- **`frontend.act` is asynchronous.** A box serves one call at a time and a
  dialog has to render back into it, so waiting inline deadlocks. Requests come
  back through the stream, not the POST.
- **An approval arrives as a frame and is answered with a Request**
  (`POST /sdk/frontend.resolve`). Until something renders `approval`, every
  unsafe Request looks exactly like a hang — which is the current state of
  this app and the first real gap to close.

## The channels are not decoration

The kernel splits what reaches a person across several render kinds, and the
whole reason that split exists is that a client could not tell them apart
otherwise. The table in `CLAUDE.md` under **Notifications** is the authority;
the short version:

- `messages` — the conversation. The agent's reply and the person's own words.
- `callable_output` — what a command returned. Not conversation.
- `notification` — the system telling the user something. Belongs in a panel.
- `error`, `tool_status`, `stream_delta`, `typing`, `turn_activity` — status.

**None of this is visible from a terminal.** The REPL declares neither
`supports_callable_output` nor `supports_notifications`, so `BaseFrontend`
flattens both into chat and the output is byte-identical whichever channel it
travelled on. Getting the channel wrong is therefore never caught by trying it
in the REPL — only a client that draws them apart notices, which is this one.

Fourteen kinds today: `messages`, `attachments`, `form_field`, `approval`,
`approval_settled`, `buttons`, `error`, `typing`, `turn_activity`,
`tool_status`, `stream_delta`, `notification`, `callable_output`,
`conversation`. The set is `sandbox/frontends.py::KINDS` and it grows; two were
added without this repo noticing, which is why the test that covers them is
pinned against `KINDS` itself rather than against a written list.

## How this connects, and what not to reinvent

The dev server proxies `/sdk`, `/events` and `/files` to `http_client_url` and
adds `Authorization: Bearer <secret_http_token>`, both read from Second Brain's
`config.json`. Consequences worth keeping:

- **The browser holds no credential**, so nothing in a bundle is secret and
  `EventSource` needs no query-string token. Do not "simplify" this by putting
  the token back in the page.
- **The page only ever talks to its own origin**, so CORS never enters into it.
  `http_allowed_origins` exists for clients that bypass the proxy and is a
  sharper edge than it looks — it is echoed into `Access-Control-Allow-Origin`
  verbatim, so a trailing slash or `localhost` where the browser said
  `127.0.0.1` fails a preflight that explains almost nothing.
- **The token is re-read per request**, so one minted after the dev server
  started is picked up with nothing to restart.
- `ui_url` decides the dev server's port (passed in as `VITE_UI_PORT`), and
  `http_client_url` decides where it proxies. They point opposite ways.

## Things that will look like bugs and are not

- **A 401 on the page is never the page's fault.** It sends no credential. It
  is the dev server not adding the header — nearly always one running code from
  before a change. Restart it.
- **No start-up notification** means the UI was announced as unreachable or the
  bridge check failed; `runtime/web_ui.py` says which, and `web_ui.log` in the
  data directory is the dev server's own output.
- **A frame for a session you are not watching is dropped, not queued.** The
  kernel renders to live sessions only.
- **`npm run dev` may already be running** — the kernel starts one and adopts
  anything already serving, so two people can both think they own it.

## Open, in rough order

1. Render the fourteen kinds. `approval` first: without it an unsafe Request is
   indistinguishable from a hang, and that is the only gap that makes the app
   *wrong* rather than merely empty.
2. Input: `frontend.submit`, attachments (write bytes to scratch with
   `fs.write_bytes`, then submit the path — there is no upload route).
3. Conversation list and history, which is `conv.list` and the **paged**
   `conv.read`.
4. Decide about a framework. The old UI is React + assistant-ui + Tailwind;
   this is currently plain ES modules with no build step beyond Vite, which is
   a real advantage worth losing deliberately rather than by default.
5. A production story. Today there is none: `npm run build` produces a `dist`
   that `http_static_dir` can serve, but a build served that way has to get its
   token from somewhere, and putting it in the bundle hands the credential to
   every browser that loads the page. The macOS Caddy deployment in the old
   repo is the only worked answer.
