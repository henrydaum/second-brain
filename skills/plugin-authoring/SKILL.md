---
name: plugin-authoring
description: How to write Second Brain sandbox code — a script, or a plugin (tool, task, service, command, frontend) — use when asked to build, fix, or extend either.
dependencies_files: tools/tool_use_skill.py
---

# Authoring Second Brain plugins

Second Brain is a microkernel: capabilities are plugins discovered by file
presence. You can author plugins yourself into the sandbox tree.

## Where plugins live

- `<DATA_DIR>/sandbox_plugins/<family>/` — your drafts (write here).
- `<DATA_DIR>/installed_plugins/<family>/` — store-installed (don't edit).
- `plugins/<family>/` in the repo — kernel built-ins (don't edit).

Families: `tools/tool_*.py`, `tasks/task_*.py`, `services/service_*.py`,
`commands/command_*.py`, `frontends/frontend_*.py`. Shared helper code goes
in `<family>/helpers/` next to the plugin, imported with RELATIVE imports
(`from .helpers import x`) so files work in any tree.

## First: does this need to be a plugin at all?

A plugin is a capability the *kernel registers* — it has a name, it stays
loaded, other code calls it. If you just need to do a piece of work, write a
**script** instead: `sandbox_plugins/scripts/<name>.py`, a file with a
`main(sdk)` function, no base class and no declarations, run with the
`run_script` tool. Scripts are far cheaper to write and are not asked about,
so they are the right answer for one-off computation, data reshaping, bulk
file work — anything you would otherwise have used the shell for.

Write a plugin when the thing should still exist tomorrow and be reachable by
name. Write a script when you just need it done.

## The flow

1. Read the matching template in `templates/` (`tool_template.py`,
   `task_template.py`, `service_template.py`, `command_template.py`,
   `frontend_template.py`) — each is a complete annotated reference.
2. Read one existing plugin of the same family for current style.
3. Write `sandbox_plugins/<family>/<prefix>_<name>.py` with a file tool.
4. Check it: if the `validate` tool is installed, call
   `validate(path="sandbox_plugins/tools/tool_<name>.py")`. It reads the file
   without importing it and reports every contract violation with a line
   number and a fix — imports, inheritance, naming, collisions, declarations.
5. On failure: read the error, edit the same file, retry. The plugin
   watcher live-loads valid edits; deleting the file unloads it.

## Contracts in one breath

- **Tool**: subclass `BaseTool`; set `name`, `description`, `parameters`
  (JSON schema), `requires_services`, `background_safe` (False if it needs
  a human present), `max_calls`; implement `run(self, context, **kwargs) ->
  ToolResult`. Optional `agent_prompt` / `agent_prompt_for(ctx)` adds
  system-prompt guidance — ALL guidance about your plugin belongs there,
  never in kernel files.
- **Task**: subclass `BaseTask`; `trigger` = "path" (per-file pipeline) or
  "event" (bus channel you own as a module constant); declare
  `event_payload_schema`, `requires_services`, optional `output_schema`
  for an SQL output table.
- **Service**: subclass `BaseService`; module-level
  `build_services(config) -> {"name": Instance}` is REQUIRED (discovery
  calls it); `_load()` must set `self.loaded = True` and return True;
  `unload()` sets it False. `lifecycle = EXTENSION` for hook carriers that
  should auto-load.
- **Command**: subclass `BaseCommand`; `form(args, context)` returns
  `FormStep` lists for interactive flows; `run(args, context)` returns the
  reply string.

## Declaring dependencies

Module-level literals (read by AST, never imported — keep them literal):
```python
dependencies_files = ['services/service_x.py']   # store-relative paths
dependencies_pip = ['some-package']
```

## Context (what `run` receives)

`context.db` (SQLite, parameterize everything), `context.config`,
`context.services` (dict, check `.loaded`), `context.runtime` (sessions,
`iterate_agent_turn`), `context.user_id`, `context.call_tool`,
`context.approve_command` (gate destructive ops), `context.request_user_input`.

## Pitfalls already paid for

- Forgetting `build_services()` → service silently never registers.
- `_load()` not setting `self.loaded = True` → `requires_services` gating
  rejects every dependent tool.
- Absolute imports of helper files → breaks when the file moves between
  sandbox and installed trees.
- Non-literal `dependencies_*` (f-strings, concatenation) → package manager
  rejects the file.
- `background_safe=True` on a tool that prompts the user → hangs unattended
  sessions. If it asks a human anything, it's `background_safe = False`.
