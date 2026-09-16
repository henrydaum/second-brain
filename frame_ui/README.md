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
/config            # set secret_http_token to a long random string
/restart
```

Then, in this folder:

```bash
npm install
cp .env.example .env.local   # copy .env.example .env.local on Windows
npm run dev
```

Paste the token into `VITE_SB_TOKEN` in `.env.local`. Opens at
http://localhost:5174.

`/setup`'s web UI phase does the server half of this for you and prints the
token.

## Pointing it somewhere else

`VITE_SB_URL` is the only thing that knows where Second Brain is, and the
browser never reads it — the dev server proxies `/sdk`, `/events` and `/files`
there, so the page only ever talks to its own origin and CORS never comes into
it. Change that one line to move between a loopback port and a Tailscale
hostname:

```
VITE_SB_URL=http://my-box.tail1234.ts.net:8787
```

There is **no kernel setting for this**. Which address the *server* listens on
is `http_port` (loopback only, by design); which address the *client* dials is
a fact about the client, and putting it in the server's config would be the
server guessing on the client's behalf.

If you ever do point a browser straight at the server instead of proxying,
`http_allowed_origins` has to match the browser's `Origin` header exactly — no
trailing slash, and `localhost` and `127.0.0.1` are different origins.

## The token

`VITE_SB_TOKEN` ends up in the dev bundle. That is fine for a dev server on
your own machine and is not fine for anything you serve to others: a build
behind a gateway should leave it empty and have the gateway add the bearer
header on its own hop, so no browser bundle holds the credential.
