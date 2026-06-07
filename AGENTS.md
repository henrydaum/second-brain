# AGENTS.md

Guidance for AI coding agents (Codex, Cursor, Copilot, Claude Code, Gemini CLI, etc.)
working **on** the Second Brain codebase. Humans should read [README.md](README.md);
this file is the agent-facing companion. If anything here conflicts with an
explicit user instruction, the user wins.

## ⚡ This is the `lite` branch (the kernel)

This branch is a **microkernel** strip-down: a minimal core that boots, runs the
agent turn, persists conversations, and loads plugins. Most product capability
arrives through installed packages. **Read the "LITE BRANCH — the kernel"
section at the top of [CLAUDE.md](CLAUDE.md) before making changes.** On `lite`,
search, scheduling, integrations, Telegram, file-editing tools, shell tools,
memory tooling, and heavy parsers are package capabilities unless the runtime
catalog proves they are installed.

## What this project is

Second Brain Lite is the **local-first AI kernel** for Second Brain: a
programmable conversation runtime with SQLite persistence, an agent turn loop,
five plugin families, package install/uninstall, and a live plugin watcher.
Full-product capabilities such as indexing/retrieval, durable memory workflows,
web search, scheduling, integrations, and Telegram belong in the store. Python +
SQLite. Solo-maintained (Henry). There is **no GUI** — a Flet GUI was removed;
do not reintroduce one.

## Setup commands

```bash
pip install -r requirements.txt   # Python 3.11+ required
python main.py                    # run the app (delegates to main.pyw)
python -m pytest -q --basetemp .pytest_tmp_full   # run the test suite
python -m pytest tests/test_service_llm.py -q --basetemp .pytest_tmp_llm
```

There is no build step and no linter config in the repo. Match the surrounding
style rather than reformatting.

## Environment notes

- **Platform is Windows-first** (developed on Win11). Paths in config are
  Windows-style; the code is cross-platform but test on Windows assumptions.
- The default dev shell is **PowerShell**. In PowerShell use `$null`, `$env:VAR`,
  and backtick line-continuation — not bash syntax.
- The app stores state in a per-OS **DATA_DIR** (see [paths.py](paths.py)):
  `%LOCALAPPDATA%/Second Brain/` on Windows. Config lives in `config.json` /
  `plugin_config.json` there; the SQLite DB, package receipts, sandbox plugins,
  installed plugins, and optional `memory.md` live there too.
  Don't hardcode DATA_DIR — import from `paths.py`.
- An LLM profile is required for agent features. Tests stub the LLM, so they
  run without API keys.

## Architecture (where things live)

- `state_machine/` — pure conversation primitives: participants, turns, phases,
  actions, forms, approvals, serializable `PhaseFrame`s. Keep this layer pure
  (no transport, no I/O coupling).
- `runtime/` — sessions, persistence, approvals, dispatch, agent-turn driver
  (`conversation_loop.py`), and `context.py` (`SecondBrainContext`, the bag every
  plugin receives). `conversation_runtime.py` is the single dispatcher and is the
  accepted "ugly duckling" — large on purpose.
- `agent/` — `system_prompt.py` (the only place the system prompt is assembled;
  sections are gated by which tools the active scope exposes), `tool_registry.py`.
- `plugins/` — base contracts plus the small kernel plugin set.
- `pipeline/` — file watcher, SQLite task queue, orchestrator DAG; it idles
  until task packages are installed.
- `events/` — pub/sub bus.
- `templates/` — the **source of truth** for how to author each plugin family.

## The plugin system (read this before adding features)

Everything user-extensible is a plugin in one of five families. Each family has
a built-in dir, a sandbox dir in DATA_DIR, and an installed-package dir in
DATA_DIR. Discovery is by file presence and filename prefix.

| Family   | Base class      | Built-in dir         | Installed dir                  | File prefix   |
|----------|-----------------|----------------------|--------------------------------|---------------|
| Tools    | `BaseTool`      | `plugins/tools/`     | `installed_plugins/tools/`     | `tool_*.py`   |
| Tasks    | `BaseTask`      | `plugins/tasks/`     | `installed_plugins/tasks/`     | `task_*.py`   |
| Services | `BaseService`   | `plugins/services/`  | `installed_plugins/services/`  | `service_*.py`|
| Commands | `BaseCommand`   | `plugins/commands/`  | `installed_plugins/commands/`  | `command_*.py`|
| Frontends| `BaseFrontend`  | `plugins/frontends/` | `installed_plugins/frontends/` | `frontend_*.py`|

Rules of thumb:

- **Tool** = an LLM-callable action returning a `ToolResult`. Put the
  model-facing summary in `llm_summary`; put frontend/debug payload in `data`.
  Tools declare `requires_services` and are gated on those services being loaded.
- **Task** = pipeline/event worker; should be idempotent.
- **Service** = a long-lived shared backend with `_load()`/`unload()` lifecycle
  and a top-level `build_services(config) -> dict` factory. Reach peers via
  `self.services`. A service that needs runtime objects implements
  `bind_runtime(*, tool_registry=None, orchestrator=None, command_registry=None,
  frontend_manager=None, runtime=None)` — see `service_plugin_watcher.py`.
- **Command** = a user-facing slash command; may collect a `FormStep` form.
- **Frontend** = a transport; submits runtime actions and renders runtime output.

When adding a plugin, **read the matching `templates/*_template.py` first**, then
read a similar built-in or installed plugin, then write yours. Names must be
unique across built-in, sandbox, and installed roots.

### Service ↔ startup ordering gotcha

In [main.pyw](main.pyw): managed services in `autoload_services` and installed
`lifecycle = "extension"` services call `svc.load()` **before** the tool
registry exists and **before** `bind_runtime` is wired (that happens in
`_bind_runtime_services` after frontends start). Runtime-extension services that
register hooks/tools should do that work in `bind_runtime()` and make it
idempotent; do not assume the registry/runtime is available at load time.

## Conventions

- OpenAI message format (`[{"role","content"}, ...]`) is the lingua franca for
  LLM calls; services with `is_llm_backend = True` translate per provider. Tool schemas
  are OpenAI function-calling shape (`to_schema()` on `BaseTool`).
- Keep the system prompt cache-friendly: static → semi-stable → dynamic, with
  `system` only at message index 0 (some providers, e.g. MiniMax, reject it
  elsewhere). See `agent/system_prompt.py`.
- Don't add heavy imports at module top level in plugins — import inside `_load`
  / `run` so discovery stays cheap and optional deps stay optional (see how
  `service_litellm.py` lazy-imports `litellm`).
- Prefer extending via a plugin over editing the core runtime.
- Keep optional capabilities optional. If a feature needs new dependencies,
  package them with the plugin instead of growing the kernel.

## Testing

- Tests live in `tests/` and use `pytest` + `monkeypatch`. They stub external
  services (LLM, litellm, etc.) — see `tests/test_litellm_service.py` for the
  fake-module pattern. New plugins with non-trivial logic should get a test that
  fakes their dependencies rather than hitting the network.
- Run `python -m pytest -q` before declaring a change done, and report real
  output. Optional native deps (`py7zr`, `rarfile`, OCR/whisper) may be absent in
  a given environment — that's expected and not a regression.

## What not to do

- Don't reintroduce a GUI.
- Don't hardcode paths or secrets; use `paths.py` and config/env.
- Don't bypass the state machine for conversation flow, or the tool registry for
  tool dispatch.
- Don't commit or push unless the user asks.
