# frame_ui

The web face of Second Brain, in this repo rather than a separate one.

Right now it is deliberately almost nothing: a page that opens the render
stream and makes one read-only Request, so that "does the browser reach the
kernel" is answered before anything is built on top of it. What is worth
keeping is `src/client.js` — the whole protocol is `POST /sdk/<request.type>`
out and one SSE stream of render frames in, and that file is both halves.

**Building on it: [BUILDING.md](BUILDING.md)** — what the protocol will not tell
you, what the old UI already solved, and what is missing here.

## Running it

Once, in this folder:

```bash
npm install
```

After that **Second Brain starts the dev server itself**, on the port in
`ui_url`, and raises a notification the moment it answers:

```
UI is reachable at: http://localhost:5174
```

It needs the HTTP frontend on (`/frontends enable http`, then `/restart`).
`/setup`'s web UI phase does that for you.

Anything already serving `ui_url` is adopted rather than replaced, so a dev
server you keep in your own terminal still works and a `/restart` does not
start a second one. `ui_autostart = false` turns the spawning off and keeps the
notification. The dev server's own output goes to `web_ui.log` in your data
directory, which is where a start-up failure explains itself.

To run it by hand instead, `npm run dev` as usual.

## Where the settings come from

Both are **kernel settings in Second Brain's own `config.json`**, edited with
`/config` like anything else, and read off disk by the dev server at startup
(`server-config.js`). Nothing is duplicated into a `.env` file, because two
copies of one value is how they end up disagreeing.

| Setting | What it does |
|---|---|
| `http_client_url` | Where this UI looks for Second Brain. Default `http://127.0.0.1:8787`. |
| `ui_url` | Where the UI itself is served — what gets started, probed, and named in the notification. Default `http://localhost:5174`. |
| `ui_autostart` | Whether Second Brain runs `npm run dev` for you. Default on. |
| `secret_http_token` | The bearer credential. Minted at first boot; you never type it. |

The two URLs point opposite ways and are easy to confuse: `http_client_url` is
where the *app* looks for Second Brain, `ui_url` is where *you* look for the
app. On this machine they are different ports; behind a gateway that serves the
build and proxies the API they can be one origin.

`ui_url` is also the only place the dev server's port is written down — the
kernel passes it in as `VITE_UI_PORT` when it starts one. A port in
`config.json` and a port in `.env.local` would agree right up until somebody
changed one, and the symptom of that is a notification pointing at nothing.

So pointing the UI at another machine is one line and a `/restart`:

```
/config   →  http_client_url = http://my-box.tail1234.ts.net:8787
```

`.env.example` exists only for the one thing that is genuinely about *this
browser* — which thread it talks to — plus commented-out overrides, if you ever
want to ignore `config.json`. `VITE_UI_PORT` is among them and is normally
supplied by the kernel from `ui_url`; setting it by hand only matters when you
are running `npm run dev` yourself.

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

A 401 is therefore never the page's fault: it is the dev server not adding
the header. It reads the token from `config.json` per request rather than
once at startup, so one minted after the server started is picked up with
nothing to restart — but a server running code from before that change still
needs restarting once.

## Do we still need the token at all?

Yes, and bundling the frontend is not an argument against it — that changed
where the *code* lives, not who can open a socket. It matters most exactly
where this is heading: the moment the port is reachable over a tailnet, the
token is the only thing between whatever else is on that network and
`POST /sdk/proc.run`. What was worth removing was the *chore*, not the
credential, which is why the kernel mints one instead of asking.
