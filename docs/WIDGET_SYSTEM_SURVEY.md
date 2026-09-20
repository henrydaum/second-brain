# Widget system survey

Survey date: 2026-09-19. Scope: the checked-out implementation, authoring template,
discovery and watcher, conversation persistence, HTTP request path, browser
bridge, panel, deployment policy, and focused tests. This is an architectural
review, not a completed browser/security audit. Proposed APIs below are design
sketches, not existing functionality.

## Assessment

Widgets already have the foundation of a local application platform: executable
HTML, a coherent visual environment, and access to the kernel's request system.
Their ceiling is much higher than a rendered answer. They can become interfaces
to files, databases, local processes, agents, and long-running work.

The limiting factor is the application model around the HTML. Today there is
one selected file and one JSON state slot per conversation, mounted in one
panel. Making the system substantially more capable requires durable instances,
live data, assets, an agent interaction contract, and a reliable development
loop. Simply exposing more request names will not address those gaps.

Keep the simple entry point: writing one HTML file should remain sufficient.
Add optional structure as an app needs it. Keep generic substrate in the kernel;
domain applications and specialist integrations should remain packages.

## What is already good

- One-file authoring needs no build process, registration ceremony, or Python.
- Bundled, installed, and workspace widgets participate in the existing trees.
  Discovery exposes shadowed files rather than silently discarding that fact.
- `brain.call` uses the existing request/policy machinery. Local capabilities
  need not be implemented again in a widget-specific backend.
- The opaque iframe keeps author code out of the authenticated host page.
  `frontend.*` calls are blocked by the relay. Sending-window and token checks
  bind messages to the mounted document.
- Theme and size updates do not normally reload a widget. Semantic HTML gets
  useful styling, and authors can override it deliberately.
- Conversation binding is persisted in the kernel and announced to clients.
  Pending widgets before the first conversation message are supported.
- The panel supports desktop resizing, mobile split view, and fullscreen. It
  stays mounted when hidden, preserving live browser state.
- There are useful tests for mounting, message routing, state delivery,
  conversation switching, and persistence.

## Concrete findings

### 1. State belongs to a conversation slot, not a widget instance

`pipeline/database.py:set_conversation_widget` updates the widget name without
clearing or partitioning the old state when changing from one non-null name to
another. The pending-session handler does the same intentionally. Selecting B
after A therefore hands B the JSON A saved; if B saves, A's previous state is
replaced. Selecting Empty deletes the state.

This is unsuitable for a library of useful apps. The UI's optimistic selection
also retains the previous state's value while changing the name.

Introduce an instance identity distinct from the widget definition. Persist
state against the instance, and let conversations reference instances. Selecting
another app should not delete work. Hiding, detaching, and deleting an instance
should be separate actions. Support state schema versions and migrations.

### 2. Persistence has no acknowledgement or transition barrier

`brain.state.set` delays the request by 500 ms, returns no persistence promise,
and reports failure only to the console. Unmounting before that timer fires can
lose the last edit. Continuous edits keep postponing the write. There is no
revision check for two sessions editing the same state and no update broadcast
to the other mounted document.

The relay does not stamp state writes with a fixed instance or conversation.
`_widget_conversation` resolves the live session's current conversation. That
creates a transition hazard: the old browser document may still be alive after
the server has changed the session binding. The exact race needs an integration
test; the missing instance check is visible in the code.

Make `set` update a host-owned save queue immediately, with batching performed
outside the disposable iframe. Add acknowledged `flush`, visible save status,
maximum batching delay, and conflict detection using revisions. Stamp every
write with the authenticated mounted instance and reject stale generations.
Do not depend on an unload callback to finish asynchronous persistence.

### 3. Editing and showing work are not yet a complete authoring loop

The watcher reports widget creation/edits but does not invalidate the browser's
catalog or mounted source. The picker refreshes at mount and when opened.
`WidgetFrame` fetches source on path changes; the panel key contains conversation
and path, but no source revision.

Consequently a newly authored and bound widget can be absent from the cached
list until the user opens the picker. Editing an already selected file leaves
the old code running. The panel's open state is separate from binding, so binding
does not itself reveal the panel.

Add a definition-change event with a content revision, explicit Reload, and a
development mode that offers or performs a state-preserving reload. Distinguish
an intentional "show this app" action from routine binding synchronization so
conversation events do not keep reopening a panel the user hid. Preserve the
last working revision and offer rollback after a failed update.

### 4. Widget identity is lost before kernel execution

`attachAppRelay` forwards only request type and arguments through the shared
HTTP client. `_frontend_act` constructs a chain rooted at the session with
`frontend:<name>`; it does not identify the widget, source revision, or instance.
The authoring template's claim that the widget appears in the chain is therefore
stronger than the implementation. The older HTML preview documentation correctly
describes the use of frontend authority.

Keep the breadth of local capabilities, but give each mounted instance a
kernel-verifiable identity. Bind it to definition/revision, session, user, and
instance. Widget code must not be allowed to assert its own identity through an
ordinary argument. Use that identity in approvals, the activity ledger, standing
grants, resource ownership, and cancellation.

This is also a product feature: "this organizer can read this folder and write
its index" can be a durable grant, with understandable attribution. Powerful
widgets should not require approving the same operation repeatedly or granting
every widget the same authority.

### 5. The browser capability contract differs across environments

The production host CSP blocks direct connections but permits arbitrary HTTPS
images. Thus "NO NETWORK" in the template is not literally enforced: remote
image requests are network requests and can carry data in their URLs. Vite does
not configure the equivalent host CSP. An opaque origin by itself does not mean
network requests are impossible.

The production policy also prevents ordinary external script loading and does
not enable WebAssembly compilation or blob workers. The iframe explicitly
denies camera, microphone, geolocation, and clipboard permissions. Native
downloads/popups are not enabled by its sandbox flags.

Maintain one tested host policy for development and deployment. Declare which
browser features are supported, and expose capability detection to authors.
Provide controlled resource/network APIs instead of requiring authors to infer
what works from console failures. Test optional worker/Wasm modes separately.

Browser references: [CSP](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy),
[iframe sandbox](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/iframe),
and [script-src / WebAssembly](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy/script-src).

### 6. Large single-file apps can be read only partially

The HTTP file route intentionally caps responses and returns 206 even when the
client did not request a range. Its current window is 11.25 MiB. `readWidget`
accepts any successful response and calls `response.text()` once, without
fetching subsequent ranges. Large HTML with embedded assets can therefore be
parsed and run as a truncated document.

Use the existing complete-file fetching machinery, or reject oversized HTML
explicitly. Optional asset packages would avoid embedding every binary into HTML.

### 7. A ready handshake and diagnostics are missing

The host posts the document, then announces initial values and marks the frame
ready on a zero-delay timer. That is not an acknowledgement that the bridge
initialized or the app rendered successfully. Requests have a pending-count cap
but no cancellation contract. A request waiting for approval legitimately needs
to remain open; a dead bridge or lost transport needs a different status.

Add a versioned ready handshake, initialization failure reporting, request
status/cancellation, and structured console/error/CSP reporting. A widget
inspector should show identity, source revision, persistence status, active
requests, subscriptions, and recent errors. Let the agent inspect the same
diagnostics so it can verify and repair what it authored.

### 8. Lifecycle and authoring documentation need reconciliation

The panel deliberately stays mounted while hidden, but the template says closing
it unmounts the document. Hidden widgets receive no visibility announcement, so
polling, animation, and local work have no standard pause signal. The fullscreen
Escape handler is attached to the host document; keyboard events originating
inside an iframe need explicit forwarding to reach it.

There is also a legacy `widget-host.html`, duplicate widget helper code, and
HTML-preview documentation/browser harness aimed at the earlier file-viewer
surface. Consolidate the active contract and move real-browser coverage onto
the widget surface. Clearly distinguish browser APIs that are unavailable,
policy-blocked, and available but discouraged.

## The application model I would build

Separate four things:

1. **Definition:** HTML or an optional package, with an ID and source revision.
2. **Instance:** persistent user work, data references, state revision, and grants.
3. **View:** a disposable mount in a conversation, tab, split, or separate window.
4. **Job:** kernel-managed work that may continue without a mounted view.

A conversation references an instance. The same instance can have multiple
views when intended; another instance of the same definition has separate data.
Jobs belong to an explicit owner and expose progress and cancellation. A browser
tab should not be the thing that keeps an overnight task alive.

This model enables a dashboard opened from several conversations, two independent
analysis documents using the same app, and returning to work without finding the
original chat. It does not require implementing a desktop window manager first.

## Capability priorities

### Live data and long-running work

Add scoped subscriptions over the host's existing event connection. Deliver
file changes, job progress, relevant database/domain changes, agent output, and
instance-state updates. Include unsubscribe, cleanup, reconnect snapshots or
replay, ordering, batching, and backpressure. Avoid forwarding the entire
internal event bus into every widget.

Build an ergonomic job abstraction over existing script/process/agent handles:
start, inspect, stream progress, cancel, resume viewing, and collect outputs.
Clearly distinguish cancelling a view's observation from stopping the job.

### Local resources and native interaction

Provide file/folder selection, attachment selection, drag-and-drop, resource
handles, save/export, clipboard actions, opening a file in its native app, and
opening external links. Host-mediated browser interactions should preserve user
gesture requirements. OS integrations should use the kernel and platform
adapters rather than pretending every iframe browser API is available.

For media, provide scoped resource URLs with streaming and range support, not
only base64 in JSON. Small images can use blobs; a multi-gigabyte video needs a
different path. Establish resource lifetime and revocation rules.

### Agent collaboration inside applications

Give apps a way to expose a concise semantic snapshot: selected records, current
document, active filters, dirty fields, and what actions are available. Expose
schema-defined actions the agent can invoke and return structured results.
Provide an explicit "ask the agent about this selection" interaction with
attachments/context and streamed progress.

The prompt currently names the bound widget and tells the agent how to read its
source. That is useful but does not tell it what the user selected or what is
currently visible. Treat widget-supplied context as application data, not as
trusted instructions. Include instance/revision checks so a delayed agent action
cannot target a different document.

### Dependencies and computation

Keep standalone HTML as the default. Add an optional manifest and asset folder
for larger apps: title, description, version, entrypoint, state schema version,
dependencies, and requested capabilities. Resolve pinned dependencies through
the existing package system and local cache.

Support common charts, editors, canvas/3D libraries, fonts, workers, and Wasm
through a documented loading mechanism. A worker can keep a view responsive;
expensive or durable native computation belongs in scripts/services. Do not
require a per-widget development server or arbitrary CDN loading as the normal
authoring path.

### Agent development tools

The high-value loop is: create, open, inspect, interact, revise, verify.
Provide screenshots, bounded DOM/accessibility snapshots, console exceptions,
request traces, and source locations to the authoring agent. Add a preview mode
with mock data and effects disabled, plus narrow/wide and light/dark checks.
Keep versions and a restore action. Let users fork an installed app into a
workspace draft with explicit identity rather than fighting discovery precedence.

### Presentation and composition

After durable instances work, add recent apps, pinning, searchable metadata,
tabs/splits, and opening a view in a separate window. Deep links should identify
an instance and optionally a selection. Let views exchange typed resource
references through the host when the user connects them.

Offer two styling modes: integrated application defaults and a creative canvas
with minimal resets. Preserve theme/accessibility helpers without making every
diagram, game, artwork, and presentation adopt the same visual language.

## What this could enable

| Application | Local advantage | Key missing substrate |
|---|---|---|
| Folder organizer | Preview and apply changes to real files; agent classifies ambiguous items | Scoped files, preview/apply operations, undo journal |
| Research workspace | Link evidence in PDFs/notes to claims; agent works on selected material | Resource viewers, semantic selection, agent actions |
| Data notebook | Browser controls and plots over local Python/SQL computation | Jobs, streamed results, durable datasets |
| Media workbench | Browse/transcribe/index local audio and video without buffering everything | Media resources, jobs, agent progress |
| Automation console | Start, monitor, cancel, and revisit scheduled work | Durable job ownership and subscriptions |
| Creative simulation | Interactive canvas/3D/audio with saved projects | Dependencies, optional workers/Wasm, export |
| Personal dashboard | Persistent views over local records and installed integrations | Independent instances, reactive queries |
| Agent operations view | Inspect parallel work, compare outputs, approve proposed effects | Agent handles, progress, provenance, structured actions |

These are target applications, not claims that all supporting APIs exist today.
Native effects should support preview/apply and undo where feasible. External
or irreversible operations need honest status; "cancel" cannot promise rollback.

## Suggested delivery sequence

**First: make current promises dependable.** Fix state partitioning and stale
writes, add acknowledged saves, source revisions/reload, catalog invalidation,
complete HTML reads, browser-policy parity, and an explicit ready/error protocol.
Add widget provenance before making capabilities broader or granting them more
persistently. Validate these with real-browser integration tests.

**Second: make useful local apps economical to author.** Add resource handles and
file/export UI, scoped subscriptions, job conveniences, typed SDK declarations,
capability discovery, and optional dependencies/assets. Build a file organizer
and a data notebook as acceptance applications; both exercise real local value.

**Third: make applications and agents collaborate.** Add semantic context,
structured actions, streamed agent work, and agent-accessible diagnostics.

**Fourth: broaden the workspace.** Add an app library, multiple views, standalone
windows, composition, and richer creative runtimes. Let usage of real apps
determine which of these are worth their complexity.

## Validation performed

- `tests/test_widgets.py`: **28 passed**.
- Focused UI suites for the HTML relay, widget styles, frame, panel, and binding:
  **27 passed across five files**.
- Reviewed request provenance, deployment CSP, state persistence, watcher,
  authoring contract, and source delivery statically.
- Did not run a production browser session, validate installed user-authored
  widgets, or perform a penetration test. Transition races and browser-specific
  feature support require targeted browser/integration reproduction.
- No runtime implementation changes were made as part of this survey.
