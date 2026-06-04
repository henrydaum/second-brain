# Second Brain — Package Store

This branch (`store`) is **not application code**. It's the package registry the
Second Brain *lite* kernel installs optional plugins from. The app reads it
directly over git (`git show origin/store:…`) — there is no server.

## Layout

```
packages/
  index.json                 # catalog: [{id, name, description, tags}, …]
  <package-id>/
    manifest.json            # id, name, description, requires[], files[], entrypoints?
    files/                   # plugin files, laid out exactly as they install
      services/service_*.py
      tools/tool_*.py
      tasks/task_*.py
      commands/command_*.py
      <family>/helpers/*.py
```

A **package** is the unit you install. Its `files/` land under
`DATA_DIR/installed_plugins` (same layout) where the kernel auto-discovers them.
`requires` lists other package ids, installed first. `entrypoints` (optional —
auto-detected from `files` otherwise) are the top-level plugin files to load;
helper files under `<family>/helpers/` are payload, not entrypoints.

## Two kinds of package

- **Atomic** — one capability: a service, tool, task, command, or helper
  (e.g. `parser-pdf`, `tool-web-search`, `service-gmail`).
- **Bundle (meta-package)** — *no files*, just a `requires` list that pulls in a
  coherent group. Install one, get the set; uninstalling prunes the members
  nothing else still needs. Current bundles:
  - `all-parsers` — every file parser + Whisper + OCR
  - `indexing-search` — full ingest + retrieval pipeline
  - `plan-mode`, `scheduling`, `web-search`, `gmail`, `mcp`, `google-drive`

## Installing (from the app)

Use `/packages` in the REPL: browse, `install <id>`, `uninstall <id>`. Install
copies files in, `pip install`s any missing third-party imports, loads
entrypoints, and writes a receipt under `DATA_DIR/packages`. Parser packages
activate immediately (the parser service reloads).

## Publishing

From the `lite` branch:

```
python scripts/package_publisher.py publish <id> --name "…" --description "…" \
    --file SRC=services/service_x.py --require other-id --tag foo
```

Omit `--file` and pass only `--require` to publish a bundle. The script runs in a
throwaway worktree of this branch, validates the whole store, commits, and pushes
— without touching your checkout. `package_publisher.py validate` checks integrity.

## Python dependencies

By default install **auto-detects** a package's third-party imports and
`pip install`s the missing ones (mapping import names to PyPI names where they
differ, e.g. `fitz`→PyMuPDF). **Most packages need to declare nothing.**

Only override this when the scan can't read your real deps — typically
optional, alternative, or platform-specific imports that are lazily imported and
shouldn't all be force-installed:

- `--pip <name>` (repeatable) — an **authoritative** dependency list that
  replaces the scan entirely. Supports PEP 508 markers, e.g.
  `--pip "pyobjc-framework-Vision; sys_platform=='darwin'"`.
- `--no-pip` — declare an empty list: install nothing and skip the scan.

Example: `service-ocr` declares `--pip pillow-heif` because its OCR engines
(`torch`, `winrt`, the macOS `pyobjc` frameworks) are optional per-platform
choices the user installs themselves — the scan would otherwise try to install
all of them and fail.
