---
name: plugin-authoring
description: How to write Second Brain sandbox code — a script, or a plugin (tool, task, service, command, frontend) — use when asked to build, fix, or extend either.
dependencies_files: tools/tool_use_skill.py
---

# Authoring Second Brain plugins

Second Brain is a microkernel: capabilities are plugins discovered by file
presence. You can author plugins yourself into the sandbox tree.

## Where plugins live

- `<DATA_DIR>/workspace/<root>/` — your drafts (write here).
- `<DATA_DIR>/installed/<root>/` — store-installed (don't edit).
- `bundled/<root>/` in the repo — ships with the app (don't edit).

Families: `tools/tool_*.py`, `tasks/task_*.py`, `services/service_*.py`,
`commands/command_*.py`, `frontends/frontend_*.py`. Shared helper code goes
in `<family>/helpers/` next to the plugin, imported with RELATIVE imports
(`from .helpers import x`) so files work in any tree.

## First: does this need to be a plugin at all?

A plugin is a capability the *kernel registers* — it has a name, it stays
loaded, other code calls it. If you just need to do a piece of work, write a
**script** instead: `workspace/scripts/<name>.py`, a file with a
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
3. Write `workspace/<root>/<prefix>_<name>.py` with a file tool.
4. Check it: if the `validate` tool is installed, call
   `validate(path="workspace/tools/tool_<name>.py")`. It reads the file
   without importing it and reports every contract violation with a line
   number and a fix — imports, inheritance, naming, collisions, declarations.
5. On failure: read the error, edit the same file, retry. The plugin
   watcher live-loads valid edits; deleting the file unloads it.

## The one rule

Your code cannot act, it can only ask. Every entry point receives `sdk`, and
anything touching disk, network, clock or process goes through it —
`sdk.fs.read(path)` not `open(path)`, `sdk.db.query(...)` not a cursor,
`sdk.log(...)` not `logging`. Requests return their value and raise on failure,
so the code reads as straight-line Python:

```python
from guest.bases import BaseTool


class WordCount(BaseTool):
    name = "word_count"
    description = "Count the words in a file."
    parameters = {"type": "object",
                  "properties": {"path": {"type": "string"}},
                  "required": ["path"]}

    def run(self, sdk, path):
        return len(sdk.fs.read(path).split())
```

`docs/SDK.md` is the reference for what `sdk` can do; its examples are executed
by the test suite, so they are correct.

## Contracts in one breath

Import the base from `guest.bases` — never from `plugins.*`, which a box
cannot see. Every entry point takes `sdk` where the old native contract took
`context`.

- **Tool**: `BaseTool`; set `name`, `description`, `parameters` (JSON schema),
  `requires_services`, `background_safe` (False if it needs a human present),
  `max_calls`; implement `run(self, sdk, **kwargs)`. Return any value.
- **Task**: `BaseTask`; `trigger` = "path" (per-file pipeline) or "event" (a
  bus channel you own as a module constant); implement `run(self, sdk, paths)`
  for "path" and `run_event(self, sdk, payload)` for "event" — the wrong one
  is silently never called. Optional `output_schema` for an SQL output table.
- **Service**: `BaseService`; implement `start(self, sdk)`, optionally
  `stop`/`poll`. Name callable methods in `exports`, hook doorways in
  `hooks = {moment: method_name}`. No `build_services`, no `self.loaded` — the
  kernel owns the lifecycle, and a live box *is* the loaded state.
- **Command**: `BaseCommand`; `run(self, sdk, args)` returns the reply
  markdown; optional `form(self, sdk, args)` returns form steps as plain
  dicts. Declare `require_approval` or `approval_actions` up front rather than
  asking mid-run.
- **Frontend**: `BaseFrontend`; `start` sets up and *returns* — there is no
  main loop. The kernel calls `poll` repeatedly and `render(kind, payload)`
  between polls.

Optional `agent_prompt` on any of them adds system-prompt guidance — a plain
string, or `def agent_prompt(self, sdk)` when it depends on live state. ALL
guidance about your plugin belongs there, never in kernel files.

## Declaring dependencies

Module-level literals (read by AST, never imported — keep them literal):
```python
dependencies_files = ['services/service_x.py']   # store-relative paths
dependencies_pip = ['some-package']
```

Declarations on the class (`name`, `requests`, `exports`, `hooks`, ...) are
read the same way, so they must be plain literals too — `tuple(ACTIONS)` reads
as nothing at all.

## Pitfalls already paid for

- Importing a kernel module (`runtime`, `config`, `plugins`, `state_machine`,
  `agent`, `pipeline`, `events`, `paths`) → loads in-process, fails in a
  subprocess. `validate` catches it; the runner would not.
- `os`, `sys`, `pathlib`, `subprocess`, `requests`, `open()`, `logging` → all
  refused, each with an sdk equivalent. `sdk.path.*` covers path arithmetic.
- Absolute imports of helper files → breaks when the file moves between
  sandbox and installed trees. Use relative imports.
- Non-literal `dependencies_*` (f-strings, concatenation) → package manager
  rejects the file.
- `background_safe=True` on a tool that prompts the user → hangs unattended
  sessions. If it asks a human anything, it's `background_safe = False`.
- More than one plugin class in a file, or a filename whose prefix does not
  match the base class → refused. One class, one file, matching name.
