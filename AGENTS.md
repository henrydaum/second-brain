# AGENTS.md

Guidance for AI coding agents (Codex, Cursor, Copilot, Claude Code, Gemini CLI, etc.)
working **on** the Second Brain codebase. Humans should read [README.md](README.md);
this file is the agent-facing companion. If anything here conflicts with an
explicit user instruction, the user wins.

## What this project is

Second Brain is a **local-first AI runtime**: a programmable conversation runtime
with file indexing/retrieval, durable memory, web search, scheduling, a Telegram
+ REPL frontend, and a live plugin system the agent can extend at runtime.
Python + SQLite. Solo-maintained (Henry). There is **no GUI** — a Flet GUI was
removed; do not reintroduce one.

## Setup commands

```bash
pip install -r requirements.txt   # Python 3.11+ required
python main.py                    # run the app (delegates to main.pyw)
python -m pytest -q               # run the test suite
python -m pytest tests/test_litellm_service.py -q   # run one test file
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
  `plugin_config.json` there; the SQLite DB and `memory.md` live there too.
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
- `plugins/` — the five extension families (see below).
- `pipeline/` — file watcher, SQLite task queue, orchestrator DAG.
- `events/` — pub/sub bus.
- `templates/` — the **source of truth** for how to author each plugin family.

## The plugin system (read this before adding features)

Everything user-extensible is a plugin in one of five families. Each has a
baked-in dir (source-controlled) and a sandbox dir (in DATA_DIR, mutable at
runtime). Auto-discovered on startup by filename prefix.

| Family   | Base class      | Dir                  | File prefix   |
|----------|-----------------|----------------------|---------------|
| Tools    | `BaseTool`      | `plugins/tools/`     | `tool_*.py`     |
| Tasks    | `BaseTask`      | `plugins/tasks/`     | `task_*.py`     |
| Services | `BaseService`   | `plugins/services/`  | `service_*.py`  |
| Commands | `BaseCommand`   | `plugins/commands/`  | `command_*.py`  |
| Frontends| `BaseFrontend`  | `plugins/frontends/` | `frontend_*.py` |

Rules of thumb:

- **Tool** = an LLM-callable action returning a `ToolResult`. Put the
  model-facing summary in `llm_summary`; put frontend/debug payload in `data`.
  Tools declare `requires_services` and are gated on those services being loaded.
- **Task** = pipeline/event worker; should be idempotent.
- **Service** = a long-lived shared backend with `_load()`/`unload()` lifecycle
  and a top-level `build_services(config) -> dict` factory. Reach peers via
  `self.services`. A service that needs the tool registry / orchestrator
  implements `bind_runtime(*, tool_registry=None, orchestrator=None,
  command_registry=None, frontend_manager=None)` — see
  `service_plugin_watcher.py` and `service_mcp.py` for the pattern.
- **Command** = a user-facing slash command; may collect a `FormStep` form.
- **Frontend** = a transport; submits runtime actions and renders runtime output.

When adding a plugin, **read the matching `templates/*_template.py` first**, then
read a similar existing plugin, then write yours. Names must be unique across
baked-in + sandbox.

### Service ↔ startup ordering gotcha

In [main.pyw](main.pyw): `autoload_services` calls `svc.load()` **before** the
tool registry exists and **before** `bind_runtime` is wired (that happens in
`_bind_runtime_services` after frontends start). So a service that registers
tools must do so idempotently from **both** `_load()` and `bind_runtime()` — do
not assume the registry is available at load time. `service_mcp.py` shows this.

## Conventions

- OpenAI message format (`[{"role","content"}, ...]`) is the lingua franca for
  LLM calls; `LiteLLMService` / `OpenAILLM` translate per provider. Tool schemas
  are OpenAI function-calling shape (`to_schema()` on `BaseTool`).
- Keep the system prompt cache-friendly: static → semi-stable → dynamic, with
  `system` only at message index 0 (some providers, e.g. MiniMax, reject it
  elsewhere). See `agent/system_prompt.py`.
- Don't add heavy imports at module top level in plugins — import inside `_load`
  / `run` so discovery stays cheap and optional deps stay optional (see how
  `service_llm.py` lazy-imports `litellm`/`openai`/`lmstudio`).
- Prefer extending via a plugin over editing the core runtime.

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
