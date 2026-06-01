# store/ — plugin staging catalog (lite branch)

This tree holds the plugins that were **moved out of the kernel** when Second
Brain was stripped to its microkernel on the `lite` branch. It mirrors the
`plugins/` layout (`services/`, `services/helpers/`, `tasks/`, `tools/`,
`tools/helpers/`, `commands/`, `frontends/`).

These files are **not on the discovery path**, so the kernel does not load them.
They are preserved here (via `git mv`, so history is intact) to become the seed
content of the future **plugin store** — the registry the kernel will install
from and uninstall to.

Nothing here is wired up yet. The next milestone is a manifest format + a
`/plugin install|uninstall|list` command that copies a plugin from a registry
into `DATA_DIR/sandbox_*` (already a discovery path) and pip-installs its declared
deps. See the "LITE BRANCH — the kernel" section in [../CLAUDE.md](../CLAUDE.md).

## What's here

- `services/` — embed, ocr, whisper, gmail, drive, mcp, web_search, timekeeper
- `services/helpers/` — modality parsers (image/audio/video/tabular/container), mcp_oauth
- `tasks/` — the whole pipeline (extract/chunk/index/embed/textualize), ocr/audio,
  dream_memory, update_titles, spawn_subagent
- `tools/` — edit_file, sql_query, run_command, render_files, lexical/semantic/hybrid
  search, web_search, email_*, schedule_subagent, test_plugin
- `tools/helpers/` — SearchResult, email_context
- `commands/` — mcp, agent, schedule, update
- `frontends/` — telegram

## Likely install bundles (future)

`pipeline-text` (extract→chunk→index), `search-lexical`, `search-semantic`
(+embed, pulls torch), `modality-ocr`, `modality-audio`, `integration-gmail`,
`integration-drive`, `integration-mcp`, `frontend-telegram`, `agent-subagents`,
`scheduling` (timekeeper + schedule command + dream_memory/update_titles).
