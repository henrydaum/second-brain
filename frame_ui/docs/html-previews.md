# HTML previews

For the agent authoring workflow and a complete example, read
[Writing HTML Apps with the Second Brain SDK](HTML_APPS.md).

Open an `.html` or `.htm` file in the file viewer to run it in an embedded
iframe. Write self-contained HTML with inline CSS and JavaScript. The viewer
fetches the contents through the existing authenticated `/files` endpoint and
passes the document to `/html-app-host.html`, a static, sandboxed host with its
own CSP. No kernel plugin is required. Deploy both the UI build and the Caddy
configuration: the host route must not inherit the main UI's script policy.

The iframe permits scripts but withholds same-origin access, popups, top-level
navigation, native form submission, and downloads. Inputs and JavaScript click
handlers work. This is browser isolation, not the kernel's Python sandbox or a
CPU/memory quota. In production, the host CSP blocks direct network connections
and external scripts; inline scripts, inline styles, and HTTPS images are allowed.

The viewer supplies `brain.call(type, args = {})` before the page's scripts.
It returns a Promise containing the SDK result (the HTTP response's `data`).
Use the Request names and argument shapes in `SDK.md` and `HTTP_PROTOCOL.md`:

```html
<!doctype html>
<button id="load">Load conversations</button>
<pre id="output"></pre>
<script>
document.querySelector('#load').onclick = async () => {
  const output = document.querySelector('#output');
  try {
    const conversations = await brain.call('conv.list', {});
    output.textContent = JSON.stringify(conversations, null, 2);
  } catch (error) {
    output.textContent = error.message;
  }
};
</script>
```

The parent UI relays these calls through its existing HTTP client, using its
authentication, current session, and frontend authority. No CORS configuration
or credentials in the HTML are needed for `brain.call`. Kernel permission
checks still apply; approvals appear in the host UI. Kernel errors reject the
Promise with `message`, `code`, `status`, and `type`. Transport errors may only
have a message. `frontend.*` control methods are reserved for the host UI.
There is no separate App permission identity; this is a prototype using the
frontend's authority, not a security boundary for kernel operations.

The relay checks the sending window and a per-document token. Closing the
preview disconnects the relay and discards late results; it does not cancel or
undo requests already accepted by the kernel. Up to 64 calls may be pending
per preview, with no timeout while waiting for human approval.

Direct `fetch` is blocked by the production host CSP. Use the relay for SDK
calls. Do not add `allow-same-origin`: combined with
scripts it would expose the surrounding UI's origin.

Relative URLs resolve against the UI page, not the HTML file's disk directory.
Use inline assets or HTTPS images. Files exceeding the text preview
limit are refused rather than executing a partial document. Close and reopen
the viewer after editing; existing file activity invalidates the text cache
when it observes a write.

For a visible boot/click/SDK check and Mac deployment instructions, see
[HTML App troubleshooting](HTML_APP_TROUBLESHOOTING.md).
