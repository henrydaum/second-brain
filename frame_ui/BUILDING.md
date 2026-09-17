# Starting point

The app is copied from `Second Brain UI`. Keep its existing React components
and place iframe slots wherever they are useful. No slots have been inserted
and no replacement layout system has been added.

## Preserved widget machinery

- `src/mount.js`: `mountWidget(element, widget, { html, scheme })` mounts a
  sandboxed iframe into a sized element and returns `setScheme` and `unmount`.
  Its message bridge, source checks, reserved-request checks, ResizeObserver,
  and cleanup are preserved. `mount.d.ts` makes the API available to TypeScript.
- `public/widget-host.html`: the iframe host and its Content Security Policy.
- `src/widgets.js`: `listWidgets()` discovers HTML widgets and `readWidget(widget)`
  fetches one through the authenticated HTTP file route.
- `src/theme.js`: the token sheet and base styles injected into each widget.
  Do not apply its frame-level stylesheet over the React app's styling.
- `src/client.js`: the preserved widget HTTP access. It now imports `THREAD`
  from `src/lib/client.ts`, so a widget uses the same session as the app.
- `server-config.js`: the existing server-side backend and credential lookup.
  `vite.config.ts` uses it for the authenticated proxy and keeps port 5174.

The React runtime already owns the SSE connection and renders approvals.
Do not call the preserved `openStream` helper to create a second connection
on the same thread; a second stream replaces the first.

## Where to start inserting a slot

- `src/App.tsx`: the overall app composition.
- `src/components/thread.tsx`: chat content.
- `src/components/conversation-sidebar.tsx`: conversations.
- `src/components/files-drawer.tsx`: current-chat files in the left sidebar.
  The Chats / Files switch replaces only the list; the right side is free.
- `src/components/file-explorer-dialog.tsx`: explorer.
- `src/components/file-viewer-dialog.tsx`: viewer.
- `src/components/settings-dialog.tsx`: settings.

A slot is simply a container with a defined size. Give `mountWidget` the
container, widget metadata, HTML from `readWidget`, and the resolved app theme
(`useResolvedTheme` in `src/lib/theme.ts`). Forward later theme changes to
`setScheme`, and call `unmount` when deliberately removing the widget.

Keep the slot's DOM element and iframe mounted when hiding, resizing, or
rearranging it. Reparenting an iframe reloads its document. React conditional
rendering and changing keys also destroy widget state, so choose a stable
mount location and use CSS for layout changes. Do not restore the old Panels
registry, layout table, or editor just to insert an iframe.

The previous frame source, including its slot pooling implementation, is
backed up under `../.codex_smoke/frame-ui-backup-20260917-155140/`.
