# Second Brain — Architecture Notes

Local-first file intelligence pipeline with REPL + Telegram frontends. Python /
SQLite. Solo dev (Henry). The Flet GUI was removed; do not reintroduce.

---

# ⚡ LITE BRANCH — the kernel (READ FIRST)

**You are on the `lite` branch.** This branch is a deliberate strip-down of
Second Brain into a **microkernel**: a minimal, reliable core that boots, runs
the conversation loop + agent turn, and loads/unloads plugins — with *everything
else* destined to be installed from a **plugin store** (the agentskills.io model:
a registry you browse, install, and uninstall from). `main` remains the full
product. Do not port heavy features back into the kernel; they belong in the store.

> Goal in priority order: (1) the kernel works **flawlessly and reliably**, then
> (2) build install/uninstall against a cloud store, then (3) versioning and
> possibly containerization. We are at the end of step (1).

## What ships in the kernel (`plugins/`)

Plugins are discovered purely by file presence (`plugins/plugin_discovery.py`).
The kernel was produced by **moving** non-essential plugins into `store/` (a
staging catalog that mirrors `plugins/`, preserved via `git mv` to seed the
future store) — *not* by deleting them. What remains:

- **Services:** `service_llm`, `service_parser` (text-only — see below),
  `service_plugin_watcher` (hot-reload = the install/uninstall substrate).
- **Tasks:** `task_compact_chat` (context-safety; nothing else).
- **Tools:** `tool_read_file`, `tool_ask_user_question`, and `tool_propose_plan`
  (the last is injected only in plan mode by `runtime/runtime_config.py:~95`, but
  the file must stay — see hard deps below).
- **Frontend:** `frontend_repl` only. Telegram moved to `store/frontends/`.
- **Commands:** REPL UX + introspection only — `config`, `setup` (LLM onboarding
  wizard), `llm`, `conversations`, `clear`, `cancel`, `plan`, `doctor`,
  `locations`, `commands`, `tools`, `services`, `tasks`. Moved out: `mcp`,
  `agent`, `schedule`, `update`.

The pipeline substrate (`pipeline/` — orchestrator, watcher, event_trigger) still
boots, but ships **zero pipeline tasks**: it idles until a pipeline plugin
(extract/chunk/index/embed) is installed. `parse_text` is kept and registers
text/code extensions plus PDF/DOCX/PPTX (all heavy libs are lazy
`try/except ImportError`, so they degrade gracefully). The richer modality
parsers (image/audio/video/tabular/container) moved to `store/services/helpers/`.

## The kernel boundary (the one rule)

Core code (`pipeline/`, `runtime/`, `state_machine/`, `agent/`, `events/`,
`config/`, `main.pyw`) hard-imports **exactly three** plugin modules. Keep these
three resolvable in any kernel:
1. `service_llm` — `runtime/agent_scope.py` + `runtime/conversation_loop.py`.
2. `tool_propose_plan` — `runtime/runtime_config.py` (plan mode).
3. `parser_registry` — `pipeline/orchestrator.py`, `pipeline/watcher.py`,
   `agent/system_prompt.py`.

Everything else is discovery-based. The agent system prompt gates every optional
section behind `_has_tool(...)` in `agent/system_prompt.py`, so missing tools
degrade silently and correctly.

## Hardening applied for kernel reliability

These edits exist so the kernel degrades cleanly when a stdlib plugin is absent —
the difference between a microkernel and a pile of assumptions:
- **`runtime/conversation_runtime.py` `request_compaction`** — guards on
  `bus.has_subscribers(COMPACT_CHAT)` and skips compaction instead of blocking the
  full 120s timeout when `task_compact_chat` isn't installed.
- **`runtime/runtime_config.py` `build_loop`** — the "no LLM" path now raises a
  friendly message pointing at `/setup` instead of an opaque error.
- **`config/config_data.py`** — `autoload_services` trimmed to
  `["llm", "parser", "plugin_watcher"]`; `enabled_frontends` → `["repl"]`;
  `DEFAULT_SCHEDULED_JOBS` → `{}` (jobs/timekeeper are store plugins now).
- **`requirements.txt`** — kernel-minimal (`litellm`, `watchdog`, `Pillow`;
  optional doc-parsing deps commented). The full per-plugin
  dependency map is documented in that file's footer.

## Next steps (not yet built)

- **Plugin store**: a manifest per plugin (deps, default config, default scheduled
  jobs, version), a registry (start GitHub-backed like skills), and a `/plugin`
  command (`search`/`install`/`uninstall`/`list`). The install *substrate already
  exists*: `plugin_discovery.load_single_plugin`/`unload_plugin`,
  `service_plugin_watcher` hot-reload, the `DATA_DIR/sandbox_*` discovery dirs, and
  the pip-install gate. Seed catalog = the `store/` tree on this branch.
- **Versioning + containerization** (Henry's follow-on; design later).

## Verifying the kernel

Discovery/boot smoke (no frontend, no config writes):
```bash
python -c "from pathlib import Path; _R=Path.cwd(); \
from config import config_manager; from pipeline.database import Database; \
from pipeline.orchestrator import Orchestrator; from agent.tool_registry import ToolRegistry; \
from plugins.plugin_discovery import discover_services, discover_tasks, discover_tools; \
c=config_manager.load(); db=Database(c['db_path']); s=discover_services(_R,c); \
o=Orchestrator(db,c,s); discover_tasks(_R,o,c); t=ToolRegistry(db,c,s); t.orchestrator=o; \
discover_tools(_R,t,c); print(sorted(s), sorted(o.tasks), sorted(t.tools))"
```
Expect services `[llm, parser, plugin_watcher]`, tasks `[compact_chat]`, tools
`[ask_user_question, read_file]`. Then `python main.py`, run `/setup` to configure
an LLM, and confirm a REPL round-trip + clean compaction on a long conversation.

---

## Recent work — state machine unification

The conversation layer was unified around a single state machine
(`ConversationState` in [state_machine/conversationClass.py](state_machine/conversationClass.py))
driven by [state_machine/runtime.py](state_machine/runtime.py)
(`ConversationRuntime`). Every frontend action — REPL, Telegram, future
background drivers — flows through one labeled `cs.enact(...)` site in
`_dispatch`, mirroring PokerMonster's `run_game`. Agent turns hand off to
`ConversationLoop.drive()`, which has its own labeled enact site for the
agent's moves.

The same primitives now back commands and tools: a `CallableSpec` has a
handler, an optional form (list of `FormStep`), and an optional
`form_factory(args, cs)` for dynamic forms. Forms suspend into a `PhaseFrame` on
the cache stack, surviving restarts via the persistence layer
([state_machine/persistence.py](state_machine/persistence.py)).

The runtime exposes `runtime.active_session_key` / `active_conversation_id`
so background drivers can identify themselves: anything with a session key
that doesn't match the active one is, by definition, running unattended.
The tool registry uses this to refuse `background_safe=False` tools from
non-active sessions. The scheduled-subagent layer was deleted and is
slated for a clean rebuild on top of these primitives — there is no
`is_subagent` flag in the runtime.

## Command lifecycle (current)

A command emits two events: `COMMAND_CALL_STARTED` (first invocation, even if
a form will be filled afterward) and `COMMAND_CALL_FINISHED` (after the
handler runs, or on cancel during a form). Same `call_id` across the
lifecycle — pinned to the form's `PhaseFrame.data["call_id"]` so STARTED
and FINISHED match up. See
[state_machine/actionClass.py](state_machine/actionClass.py)
`_CallableAction.execute` and `_run`.

`BaseFrontend` ([plugins/BaseFrontend.py](plugins/BaseFrontend.py)) subscribes
both events and routes them through `render_tool_status(session_key,
payload)`. Telegram edits a single message in place: `⋯ /name` →
`✓ /name` or `✕ /name` with error. REPL prints the same shapes to stdout.

## Where to plug in

- **Add a slash command**: drop a `BaseCommand` subclass into
  [plugins/commands/](plugins/commands/) as `command_*.py`, or into the sandbox
  command directory via `build_plugin`. Commands receive `SecondBrainContext`
  in both `form(args, context)` and `run(args, context)`.
- **Add a tool**: drop a `BaseTool` subclass into [plugins/tools/](plugins/tools/);
  it's discovered automatically. Tools receive `SecondBrainContext` from
  [runtime/context.py](runtime/context.py).
- **Drive an agent from a task**: call `context.runtime.iterate_agent_turn(...)`
  on a session key. The runtime persists history and markers atomically
  for you. Background drivers should keep their session key distinct from
  the active one so the registry's `background_safe` gate kicks in.
- **Let an agent run a slash command**: the `slash_command` tool
  ([plugins/tools/tool_slash_command.py](plugins/tools/tool_slash_command.py))
  dispatches with a structured dict, skipping the form. Same
  COMMAND_CALL_STARTED/FINISHED events as a human run.

## Command plugins

Slash commands now mirror the rest of the plugin system. The repo starts with a
clean command slate: add built-ins as `command_*.py` files under
[plugins/commands/](plugins/commands/), or create sandbox commands under
`DATA_DIR/sandbox_commands`. The registry in
[plugins/frontends/helpers/command_registry.py](plugins/frontends/helpers/command_registry.py)
is only the adapter: it builds context-aware forms, parses one-shot `/cmd ...`
input mechanically, and dispatches structured dict args.

## Sandbox plugin system

The agent can author its own tools/tasks/services/commands into sandbox folders
via the `build_plugin` tool. The `run_command` tool gates pip install/uninstall
behind user approval. Sandbox plugins are auto-discovered alongside
first-party ones in [plugins/](plugins/).

## Files that matter most

- [runtime/context.py](runtime/context.py) — `SecondBrainContext`, the
  shared bag tools/tasks receive.
- [state_machine/runtime.py](state_machine/runtime.py) —
  `ConversationRuntime`, the single dispatcher. ~940 lines and growing; this
  is the accepted "ugly duckling" of the codebase.
- [state_machine/actionClass.py](state_machine/actionClass.py) — every
  user/agent action type lives here; one class per action.
- [pipeline/orchestrator.py](pipeline/orchestrator.py) — task scheduling and
  the dependency-pipeline DAG. `runtime` is wired in
  [plugins/frontends/bootstrap.py](plugins/frontends/bootstrap.py).
- [agent/system_prompt.py](agent/system_prompt.py) — single entry point for
  building the agent system prompt; gates sections by which tools the
  current scope exposes.
