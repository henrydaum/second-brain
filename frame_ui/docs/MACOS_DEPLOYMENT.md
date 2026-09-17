# macOS production deployment

The production deployment has two local services:

```text
Browser -> Caddy (127.0.0.1:4173) -> frontend_http (127.0.0.1:8787)
              |                              |
              +-- compiled React files       +-- Second Brain HTTP/SSE API
```

Caddy serves a tested, compiled release and supplies the Second Brain bearer
token only on its loopback request to `frontend_http`. Production JavaScript
contains no token. Vite is used for development and builds, not as a long-lived
production server.

## Prerequisites

- Second Brain is already running as a user LaunchAgent.
- The `frontend_http` store frontend is installed and enabled.
- [Homebrew](https://brew.sh/) and Node.js are installed.
- The repository is cloned under the same macOS user that runs Second Brain.

Configure `frontend_http` through Second Brain's configuration UI or REPL:

| Setting | Value |
| --- | --- |
| `http_port` | `8787` |
| `secret_http_token` | A long random token |
| `http_static_dir` | Empty |
| `http_allowed_origins` | Empty |

The token must be the same one entered during frontend installation. The HTTP
frontend remains bound to loopback; do not expose port 8787.

Use a long token made from letters, digits, and the conventional token punctuation
`. _ ~ + / = -`. The installer rejects whitespace and configuration syntax so
the value can be substituted into Caddy safely.

## Install

From the repository root:

```bash
sh deploy/macos/install.sh
```

The installer:

1. Installs Caddy with Homebrew if necessary.
2. Prompts for `secret_http_token` without echoing it.
3. Stores it in `~/Library/Application Support/Second Brain UI/runtime.env`
   with mode `0600`.
4. Runs tests, lint, type-checking, and the production build.
5. Activates the build atomically under the application-support directory.
6. Installs and starts `~/Library/LaunchAgents/com.secondbrain.ui.plist`.

Open <http://127.0.0.1:4173>. Caddy starts at login and restarts if it exits.
It may start before Second Brain; requests return `502` until `frontend_http`
is ready, then recover without restarting Caddy.

The installer first probes port 8787 without credentials. A working
`frontend_http` answers `401`; `503` means the kernel listener exists but no
frontend owns it, and a connection failure means Second Brain is not listening.

For unattended installation, supply the token in the process environment. Do
not put it on the command line, where it would enter shell history:

```bash
read -s SB_HTTP_TOKEN
export SB_HTTP_TOKEN
sh deploy/macos/install.sh
unset SB_HTTP_TOKEN
```

## Operate and update

All operational commands run from the repository clone:

```bash
# Show launchd state and probe Caddy
sh deploy/macos/manage.sh status

# Restart Caddy
sh deploy/macos/manage.sh restart

# Test, build, and atomically activate the checked-out source
sh deploy/macos/manage.sh update

# Swap the current and previous successful releases
sh deploy/macos/manage.sh rollback

# Replace the gateway token after changing secret_http_token in Second Brain
sh deploy/macos/manage.sh set-token
```

The optional store package `command_update_ui` provides `/update_ui`.
Install it with `/packages install command_update_ui`. It discovers the checkout
from the installed LaunchAgent, pulls with `git pull --ff-only`, and runs the
deployment command with progress messages in the normal command interface.
The regular **Update** action continues to invoke the kernel's `/update` command.

To perform the same update from a terminal in the checkout:

```bash
git pull --ff-only
sh deploy/macos/manage.sh update
```

A failed test, lint, type-check, or build never changes the active release.
Successful updates retain the immediately previous release for rollback.

Logs are written to:

```text
~/Library/Logs/Second Brain UI/caddy.stdout.log
~/Library/Logs/Second Brain UI/caddy.stderr.log
```

## Uninstall

```bash
sh deploy/macos/manage.sh uninstall
```

This stops and removes only the Caddy LaunchAgent. It deliberately preserves
the private token, releases, and logs under `~/Library` so uninstall is
recoverable. Remove those directories manually only if their contents are no
longer needed.

## Security boundary

- Caddy and `frontend_http` listen only on `127.0.0.1`.
- Caddy overwrites, rather than trusts, a browser-supplied Authorization header.
- Static files and the browser contain no backend bearer token.
- `http_allowed_origins` stays empty because the browser uses one origin.
- Do not bind port 4173 to the LAN or forward it through a router.

### `/sdk` is refused unless the request came from this app

Caddy attaches the bearer token to everything that reaches `/sdk`, so *what
reaches it* is the perimeter. Loopback binding is not that perimeter: a page on
any site, open in a tab of the same browser, can POST to
`http://127.0.0.1:4173/sdk/…` — such a request needs no preflight and no token,
because the gateway supplies the token itself. The reply is unreadable to
whoever sent it and the action has already happened.

The gateway therefore refuses any `/sdk` request whose `Origin` is not this
origin. Browsers send `Origin` on every POST, including same-origin ones, so
the app is unaffected. Two consequences worth knowing:

- `curl http://127.0.0.1:4173/sdk/...` answers `403` unless you send a matching
  `Origin` header. Talk to `127.0.0.1:8787` directly with the bearer token when
  you want to drive the backend by hand.
- Putting another proxy in front — Cloudflare Access, say — must preserve the
  `Host` header, or the check compares the browser's `Origin` against the wrong
  name and refuses everything.

`/events` and `/files` carry no such check. Both are GETs, which send no
`Origin`, and a cross-origin read of either is already refused by the browser
because no CORS headers ever come back.

### Files are served, never executed

`/files` responses carry `Content-Security-Policy: sandbox`, and the file
viewer loads anything it frames inside a `sandbox=""` iframe. Both exist for
one case: an SVG is a *document*, not a picture, whenever a framing element
loads it, so a script inside one would otherwise run at this origin — with the
gateway attaching the backend credential to whatever it then called. Neither
mechanism affects images, audio, video, or the PDF viewer, which fetches bytes
and renders them from a blob.

### Content-Security-Policy

The app document is served with a policy that keeps scripts, styles, XHR and
framing on this origin. Two directives are deliberately looser:

- `style-src 'unsafe-inline'` is required — Radix positions popovers, tooltips
  and dialogs with inline styles computed at runtime.
- `img-src` additionally allows `https:`, so a remote picture in an agent's
  reply still draws. Tighten it to `'self'` if you would rather no reply could
  ever cause an outbound image request.

The one inline script — the theme bootstrap in `index.html`, which must run
before the first paint — is admitted by SHA-256 hash.
`deploy/macos/csp-hash.test.ts` fails the test run, and therefore the release,
if the script is edited without updating the hash in the Caddyfile.

## Private remote access with Tailscale

Install Tailscale on the Mac Mini, iPhone, and Windows PC and sign all three in
to the same tailnet. On the Mac Mini, publish Caddy as a persistent private
HTTPS service:

```bash
tailscale serve --bg http://127.0.0.1:4173
tailscale serve status
```

The status command prints the canonical `https://...ts.net` URL. Open that URL
on the other two devices while Tailscale is connected. Use that URL for normal
access and for installing the UI as an app; an installed web app is tied to its
origin, so do not install it from the loopback URL first.

`--bg` makes the Serve configuration survive Tailscale and machine restarts.
This is **Tailscale Serve**, which is private to the tailnet. Do not enable
Tailscale Funnel, bind Caddy to the LAN, or forward either local port through a
router. To inspect or remove the remote endpoint later:

```bash
tailscale serve status
tailscale serve reset
```

If the Tailscale command is not found when using the graphical macOS app, use
the app's CLI installation option or run the binary from its documented app
location. Tailscale Serve may prompt once to enable HTTPS certificates.

### Install as an app

- **iPhone:** Open the `https://...ts.net` URL in Safari, use Share, choose
  **Add to Home Screen**, keep **Open as Web App** enabled if iOS offers it,
  then tap **Add**.
- **Windows:** Open the same URL in Microsoft Edge, choose **Apps > Install
  Second Brain** (or use the install icon in the address bar), then choose the
  taskbar/Start options you want.

The first PWA release intentionally has no service worker. The manifest gives
it a standalone window and app icon, while every UI and API request remains
live. This avoids caching private conversations, files, SSE, or stale frontend
code. Consequently the app requires the Mac Mini and Tailscale connection to be
available; offline mode can be designed separately if it ever becomes useful.

Each browser installation stores its own session identity locally. This lets
the Mac, iPhone, and Windows clients keep independent event streams while still
seeing the same persisted conversations. An explicit `?thread=name` URL remains
available for an intentional extra session; do not open that exact URL in two
places at once, because the backend permits one event stream per thread.

For a later Cloudflare deployment, put Access in front of the tunnel before
forwarding to `http://127.0.0.1:4173`. It does not require changing the frontend
build.

## Port 8787 troubleshooting

If Second Brain reports that `frontend_http` could not take port 8787, identify
the listener before changing any ports:

```bash
lsof -nP -iTCP:8787 -sTCP:LISTEN
curl -i --max-time 3 http://127.0.0.1:8787/events
```

- One Second Brain process plus `HTTP/1.1 401` means the listener is healthy;
  the error may be from a second, manually launched Second Brain process or an
  older log entry.
- Two Second Brain processes means the LaunchAgent copy and a manual copy are
  competing. Keep the LaunchAgent instance and stop the manually started one.
- `503` means the kernel has the socket but `frontend_http` failed to claim it.
  Use `/frontends` to disable any duplicate HTTP-serving frontend, then restart
  Second Brain so ownership is established cleanly.
- A different process name means that application owns 8787. Stop it or move
  `frontend_http` with `http_port`; if the backend port changes, update all three
  Caddy upstreams to match before reinstalling.
