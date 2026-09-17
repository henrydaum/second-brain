# HTML renders but buttons do nothing

The original viewer used `srcdoc`, which inherited the main UI's CSP. That
policy admitted only same-origin scripts and the theme bootstrap's hash, so it
blocked both the injected bridge and author inline scripts. Adding
`'unsafe-inline'` alongside that hash does not fix this: browsers ignore it when
a hash or nonce appears in the same script source list.
See [MDN's CSP guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CSP).

The viewer now loads `/html-app-host.html` as a real document. Caddy gives that
route its own policy with `sandbox allow-scripts` and inline script permission.
The viewer delivers the prepared App once by message. The host writes it into
its own document, preserving the opaque sandbox and dedicated policy. The
main UI retains its hash-based policy, and `/files` retains its script-disabled
sandbox. Relay source, origin, and per-document token checks remain intact.

## Deploy on the Mac Mini

First transfer/sync the corrected UI source from the Windows checkout to
`/Users/henry/Projects/second-brain-ui/second-brain-ui`. Include the new
`public/html-app-host.html`, the viewer change, and `deploy/macos/Caddyfile`.
Do not transfer Windows `node_modules`. Merge local Mac changes deliberately;
the handoff described changes that were not present in the Windows checkout.
The main UI's script directive should retain its hash and omit the attempted
`unsafe-inline` workaround.

Then run on the Mac:

```sh
cd /Users/henry/Projects/second-brain-ui/second-brain-ui
sh deploy/macos/build-release.sh
sh deploy/macos/manage.sh restart
curl -sSI http://127.0.0.1:4173/html-app-host.html
```

The host response must be HTML and its CSP must start with
`sandbox allow-scripts;` and contain `script-src 'unsafe-inline'` without a
hash. It must not include a second, restrictive script policy from another
proxy. Check the public URL's headers too if they differ from loopback.
Never solve this by adding `allow-same-origin` or disabling the main UI's CSP.

Reload the PWA/browser UI and reopen the HTML file. A frontend rebuild without
the Caddy route change is insufficient.

## Visible diagnostic

Copy [html-app-diagnostic.html](examples/html-app-diagnostic.html) into the
kernel's authorized workspace and open it in the file viewer. It makes no
changes to kernel data; the SDK button lists one page of conversations.

| Display | Meaning |
|---|---|
| Missing `diagnostic v2` heading | The displayed file/version is stale or different. |
| `BOOT PENDING` stays visible | The document rendered but the boot script did not execute. Check CSP and the deployed host. |
| `SCRIPT EXECUTED` then `CLICK PENDING` after tapping | JavaScript ran, but the click handler did not receive the tap. Investigate overlays/input handling. |
| `CLICK RECEIVED` | The basic script and input path works. |
| `SDK WAITING` | A call is pending; check host approvals and connectivity. |
| `SDK OK` | The injected helper and frontend relay completed a request. |
| `SDK ERROR` or error text below | The page caught a bridge, request, or script error. |

Error listeners are installed before the diagnostic boot mutation. If scripts
are blocked altogether, those listeners cannot run either; the static
`BOOT PENDING` text is intentional.

## Browser regression check

```sh
npx playwright install chromium webkit
node scripts/check-html-app.mjs --expect-blocked --attempted-fix
node scripts/check-html-app.mjs
node scripts/check-html-app.mjs --webkit
```

The first command after installation reproduces the old `srcdoc` failure with
the ineffective hash-plus-unsafe-inline policy. The other checks bundle the
actual `FileView` inside the UI's modal and serve the policies from the Caddyfile
over HTTP. They exercise script startup, touch input, a mocked SDK round trip,
parent DOM isolation, and closing the modal. No real kernel requests are made.
Playwright WebKit on Windows is useful coverage, not a claim that the running
Mac Mini or a physical iPhone has been tested.
