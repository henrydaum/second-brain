# Second Brain UI

The existing React UI is the starting point. Its interface and modules are
intact; future widget slots can be inserted where needed. The discarded frame
layouts, editor, and built-in Panels abstraction have been removed.

```sh
npm ci
npm run dev
npm run build
npm test
```

Development runs at http://localhost:5174. The existing `server-config.js`
reads the kernel's configuration and the Vite proxy authenticates `/sdk`,
`/events`, and `/files`. No browser token or copied `.env.local` is required.

The iframe machinery remains in `src/mount.js`, `src/widgets.js`,
`src/client.js`, `src/theme.js`, and `public/widget-host.html`.
See [BUILDING.md](BUILDING.md) for insertion points and lifecycle rules,
and [WIDGET_STYLE.md](WIDGET_STYLE.md) for the widget authoring contract.
