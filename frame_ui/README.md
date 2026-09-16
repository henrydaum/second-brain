# frame_ui

The web face of Second Brain, in this repo rather than a separate one.

Right now it is deliberately almost nothing: a page that opens the render
stream and makes one read-only Request, so that "does the browser reach the
kernel" is answered before anything is built on top of it. What is worth
keeping is `src/client.js` — the whole protocol is `POST /sdk/<request.type>`
out and one SSE stream of render frames in, and that file is both halves.

## Running it

Second Brain has to be *running*, with its HTTP frontend on:

```
/frontends enable http
/restart
```

Then, in this folder:

```bash
npm install
npm run dev
```

Opens at http://localhost:5174. There is nothing to configure and no token to
copy — `npm run dev` prints the backend it found.

`/setup`'s web UI phase does the server half for you.

## Where the settings come from

Both are **kernel settings in Second Brain's own `config.json`**, edited with
`/config` like anything else, and read off disk by the dev server at startup
(`server-config.js`). Nothing is duplicated into a `.env` file, because two
copies of one value is how they end up disagreeing.

| Setting | What it does |
|---|---|
| `http_client_url` | Where this UI looks for Second Brain. Default `http://127.0.0.1:8787`. |
| `secret_http_token` | The bearer credential. Minted at first boot; you never type it. |

So pointing the UI at another machine is one line and a dev-server restart:

```
/config   →  http_client_url = http://my-box.tail1234.ts.net:8787
```

`.env.example` exists only for the two things that are genuinely about *this
browser* — which thread it talks to, and which port the dev server listens on —
plus commented-out overrides for the two above, if you ever want to ignore
`config.json`.

## The token is never in the page

The browser sends no credential at all. It talks only to its own origin; the
dev server proxies `/sdk`, `/events` and `/files` onward and adds
`Authorization: Bearer …` on the hop the browser cannot see. Three things fall
out of that, and each of them is a bug avoided rather than a nicety:

* **Nothing in the bundle is secret**, so a build is not a credential leak
  waiting for somewhere to be served from.
* **`EventSource` needs no special case.** It cannot set headers, so the usual
  arrangement puts the token in the query string — and a token in a URL is one
  that ends up in logs.
* **CORS never comes into it.** The alternative is `http_allowed_origins`,
  which is echoed into `Access-Control-Allow-Origin` verbatim, so a trailing
  slash or `localhost` where the browser said `127.0.0.1` fails a preflight
  that then explains almost nothing.

A 401 is therefore never the page's fault: it is an empty or stale
`secret_http_token`, or a dev server started before the kernel minted one.

## Do we still need the token at all?

Yes, and bundling the frontend is not an argument against it — that changed
where the *code* lives, not who can open a socket. It matters most exactly
where this is heading: the moment the port is reachable over a tailnet, the
token is the only thing between whatever else is on that network and
`POST /sdk/proc.run`. What was worth removing was the *chore*, not the
credential, which is why the kernel mints one instead of asking.
