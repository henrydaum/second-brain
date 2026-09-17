# Writing HTML Apps with the Second Brain SDK

An agent can create an interactive App by writing one `.html` file and giving
the user its path to open in the Second Brain UI file viewer. Use HTML for the
structure, inline CSS for presentation, and JavaScript for behavior. No plugin
registration, Python entry point, package installation, or frontend rebuild is
required for each App.

The viewer injects `window.brain` before the App's scripts. Its one method is:

```javascript
const result = await brain.call("request.type", { argument: "value" });
```

The call goes through the host UI's HTTP client and uses its current session,
authentication, and frontend authority. The App does not need a server URL,
bearer token, thread ID, or its own connection to the event stream.

## Authoring workflow

1. Read this guide and look up the required requests in
   [HTTP_PROTOCOL.md](HTTP_PROTOCOL.md) and [SDK.md](SDK.md).
2. Write a self-contained HTML file in the agent's authorized workspace. Use
   the workspace path supplied by the running Second Brain environment; do not
   hard-code the UI repository or invent a user's filesystem path.
3. Provide the file path or attachment so the user can open it in the file
   viewer. Merely writing the file does not open the preview automatically.
4. Test rendering, then one read request, then any user-requested mutations.
5. After editing, close and reopen the preview. File activity invalidates the
   cached contents when the UI observes the write; there is no guaranteed hot
   reload. If it still shows old contents, reload the UI and reopen the file.

Python plugin templates and `sdk.plugins.validate` are not the authoring path
for HTML Apps. Opening the file directly in another browser tab or from disk
does not inject `brain`.

## Complete example

Save this as `conversations.html`. It displays a page of conversation records
without assuming which fields each record contains.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conversation browser</title>
  <style>
    body { margin: 0; padding: 24px; font: 16px system-ui; color: #18202b;
           background: #f5f7fa; }
    button { padding: 8px 16px; cursor: pointer; }
    button:disabled { cursor: wait; }
    pre { padding: 16px; background: white; border: 1px solid #ccd3dd;
          border-radius: 8px; white-space: pre-wrap; overflow-wrap: anywhere; }
  </style>
</head>
<body>
  <h1>Conversations</h1>
  <button id="load" type="button">Load conversations</button>
  <p id="status" role="status" aria-live="polite">Ready.</p>
  <pre id="output"></pre>
  <script>
    const button = document.querySelector('#load');
    const status = document.querySelector('#status');
    const output = document.querySelector('#output');

    button.addEventListener('click', async () => {
      button.disabled = true;
      status.textContent = 'Loading…';
      try {
        if (!window.brain) {
          throw new Error('Open this file in the Second Brain UI file viewer.');
        }
        const page = await brain.call('conv.list', {
          details: true, limit: 20, offset: 0
        });
        output.textContent = JSON.stringify(page.items, null, 2);
        status.textContent = page.has_more
          ? 'Showing the first page; more conversations are available.'
          : 'Loaded.';
      } catch (error) {
        status.textContent = error.code === 'approval_declined'
          ? 'The request was not approved.'
          : error.message || 'Could not load conversations.';
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
```

## Calling the SDK

Use a Request name such as `fs.read`, not `sdk.fs.read` or `/sdk/fs.read`.
Pass named arguments in a plain JSON-compatible object; the default is `{}`.
Use `await` inside an async function or a module script.

| Operation | JavaScript |
|---|---|
| List a page of conversations | `await brain.call('conv.list', { details: true, limit: 20, offset: 0 })` |
| Read a conversation | `await brain.call('conv.read', { id: conversationId })` |
| Read a text file | `await brain.call('fs.read', { path: filePath })` |
| Save text to an authorized path | `await brain.call('fs.write', { path: filePath, data: text, mode: 'overwrite' })` |
| Create a conversation without activating it | `await brain.call('conv.create', { title: 'App notes', activate: false })` |

Here `filePath`, `text`, and `conversationId` are values your App supplies.
Host paths refer to the machine running the kernel. JSON serialization handles
path values; in JavaScript literals, escape Windows backslashes or use forward
slashes where supported.

The Promise resolves to the result itself, not a Fetch `Response` or an HTTP
envelope. Do not call `.json()` or unwrap `.data` on it. Request-specific result
shapes still apply: `conv.list` with `details: true` returns
`{ items, has_more, categories }`; `fs.read` returns text. For binary Requests,
follow the HTTP protocol's base64 representation rather than Python `bytes`.

The Python SDK guide describes both Requests and Python-only helpers. This
bridge exposes Requests; it does not provide Python helpers, `sdk.Path`, or
Python objects in JavaScript. Prefer HTTP wire argument names when the Python
convenience API differs. Check the documented operation rather than assuming
every Python method maps directly to identical JavaScript arguments.

## Errors, approvals, and state

Always catch rejected calls and show an understandable status in the App.
Kernel errors include `message`, `code`, `status`, and `type`; transport errors
may provide only a message. Relay errors include `invalid_request`,
`reserved_request`, and `too_many_requests`.

Kernel permission checks still run. An operation may wait for an approval in
the surrounding UI, so keep the page responsive while awaiting it. Do not retry
a mutation automatically: a lost response does not prove the operation failed.
Disable the initiating button while it is pending to avoid accidental repeats.

All `frontend.*` requests are reserved for the host UI, including approval
responses. Other calls use the frontend's authority, not separate App grants.
Make changes only for the user's requested task. Render returned text with
`textContent` or DOM text nodes rather than inserting it as HTML.

The bridge supports at most 64 pending calls per preview and has no approval
timeout. Prefer explicit refresh buttons over rapid polling. There is no
subscription API in `brain`; do not open another `/events` stream for the
host's thread, since that can replace the UI's connection.

Keep temporary state in JavaScript variables. Save persistent state explicitly
through documented SDK Requests. Closing or replacing the preview destroys its
local state and disconnects the relay, but does not undo or cancel kernel work
that was already accepted.

## Browser constraints

The viewer preserves normal browser scrolling and the App's CSS. Let the page
grow with its content; avoid page-wide `overflow: hidden` or gesture handlers
that cancel scrolling. Use `overflow: auto` on constrained containers whose
contents must remain reachable.

Scope swipe controls to their interactive areas. For a horizontal swipe control,
use `touch-action: pan-y pinch-zoom` so vertical page scrolling and pinch zoom
remain available. Use `touch-action: none` only on a control that needs to handle
both gesture directions, such as a drawing canvas. Leave enough surrounding
space for users to start a normal scrolling gesture outside those controls.
Do not put gesture restrictions on `html`, `body`, or a page-sized wrapper:
ancestor restrictions also affect their descendants. A gesture's behavior is
determined where it starts; moving out of a control mid-swipe does not reliably
turn that gesture into page scrolling.

The iframe permits JavaScript but has an opaque origin. It cannot access the
parent DOM or rely on origin-backed storage such as `localStorage`. Popups,
top-level navigation, native form submission, and downloads are withheld.
Use buttons and input handlers; if using a form, prevent its native submission.

Use `brain.call` for SDK access. The production host CSP blocks direct `fetch`.
Do not embed credentials, implement a second
relay, change the iframe sandbox, or overwrite `window.brain`.

Inline CSS, JavaScript, and assets are simplest. Relative resource URLs resolve
against the host page, not the HTML file's directory. The production policy
blocks external scripts; bundle libraries inline if needed. HTTPS images are
allowed. React requires browser-ready JavaScript;
raw JSX and TypeScript are not compiled by the viewer.

Keep Apps small: the viewer uses a roughly 2 MiB text preview limit and refuses
truncated HTML. Browser isolation is not a CPU/memory quota or the Python
sandbox. Avoid blocking loops and unbounded work.

For implementation details, see [html-previews.md](html-previews.md),
[the relay](../src/lib/html-app.ts), and
[the HTTP client](../src/lib/client.ts).
