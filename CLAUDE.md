# Second Brain — Architecture Notes

Local-first file intelligence pipeline with REPL + Telegram frontends. Python /
SQLite. Solo dev (Henry). The Flet GUI was removed; do not reintroduce.

## Recent work — state machine unification

The conversation layer was unified around a single state machine
(`ConversationState` in [state_machine/conversationClass.py](state_machine/conversationClass.py))
driven by [state_machine/runtime.py](state_machine/runtime.py)
(`ConversationRuntime`). Every frontend action — REPL, Telegram, scheduled
subagent — flows through one labeled `cs.enact(...)` site in `_dispatch`,
mirroring PokerMonster's `run_game`. Agent turns hand off to
`ConversationLoop.drive()`, which has its own labeled enact site for the
agent's moves.

The same primitives now back commands and tools: a `CallableSpec` has a
handler, an optional form (list of `FormStep`), and an optional
`form_factory(args, cs)` for dynamic forms. Forms suspend into a `PhaseFrame` on
the cache stack, surviving restarts via the persistence layer
([state_machine/persistence.py](state_machine/persistence.py)).

Subagent cron jobs were rewired onto the same runtime:
[plugins/tasks/task_run_subagent.py](plugins/tasks/task_run_subagent.py) now
loads a session keyed by `subagent:<job_name>` and calls
`runtime.iterate_agent_turn(...)`. Approvals, forms, cancel semantics — all
the same path as user turns.

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
payload)`. Telegram edits a single message in place: `⏳ /name` →
`✓ /name` or `✗ /name` with error. REPL prints the same shapes to stdout.

## Where to plug in

- **Add a slash command**: drop a `BaseCommand` subclass into
  [plugins/commands/](plugins/commands/) as `command_*.py`, or into the sandbox
  command directory via `build_plugin`. Commands receive `SecondBrainContext`
  in both `form(args, context)` and `run(args, context)`.
- **Add a tool**: drop a `BaseTool` subclass into [plugins/tools/](plugins/tools/);
  it's discovered automatically. Tools receive `SecondBrainContext` from
  [runtime/context.py](runtime/context.py).
- **Drive an agent from a task**: call `context.runtime.iterate_agent_turn(...)`
  on a session key. See `task_run_subagent.py` for the canonical example.
- **Let an agent run a slash command**: the `slash_command` tool
  ([plugins/tools/tool_slash_command.py](plugins/tools/tool_slash_command.py))
  dispatches with a structured dict, skipping the form. Same
  COMMAND_CALL_STARTED/FINISHED events as a human run.

## Command plugins

Slash commands now mirror the rest of the plugin system. Built-ins live in
[plugins/commands/command_core.py](plugins/commands/command_core.py), sandbox
commands live under `DATA_DIR/sandbox_commands`, and the registry in
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
