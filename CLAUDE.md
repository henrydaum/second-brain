# Second Brain — Architecture Notes

Local-first AI kernel with SQLite persistence, a REPL frontend, package
install/uninstall, and live plugin loading. Python / SQLite. Solo dev (Henry).
The Flet GUI was removed; do not reintroduce.

---

# ⚡ LITE BRANCH — the kernel (READ FIRST)

**You are on the `lite` branch.** This branch is a deliberate strip-down of
Second Brain into a **microkernel**: a minimal, reliable core that boots, runs
the conversation loop + agent turn, persists conversations, and loads/unloads
plugins. Product capabilities arrive through a **package store** (the
agentskills.io model: a registry you browse, install, and uninstall from).
`main` remains the full product. Do not port heavy features back into the
kernel; they belong in packages.

> Goal in priority order: (1) the kernel works **flawlessly and reliably**, then
> (2) build install/uninstall against a cloud store, then (3) versioning and
> possibly containerization. We are at the end of step (1).

## What ships in the kernel (`plugins/`)

Plugins are discovered purely by file presence (`plugins/plugin_discovery.py`).
The kernel was produced by **moving** non-essential plugins into `store/` (a
staging catalog that mirrors `plugins/`, preserved via `git mv` to seed the
future store) — *not* by deleting them. What remains:

- **Services:** `service_llm`, `service_compactor` (context-safety),
  `service_parser` (text + image helper discovery), and
  `service_plugin_watcher` (hot-reload = the install/uninstall substrate).
  If another tracked service remains, treat it as kernel-boundary debt unless
  the user explicitly keeps it.
- **Tasks:** none.
- **Tools:** none in the tracked kernel tree. `tool_read_file`,
  `tool_ask_user_question`, shell/file-editing tools, SQL tools, and plugin
  authoring tools are package capabilities unless discovery shows they are
  installed.
- **Frontend:** `frontend_repl` only. Telegram moved to `store/frontends/`.
- **Commands:** REPL UX + introspection only — `config`, `setup` (LLM onboarding
  wizard), `llm`, `conversations`, `clear`, `cancel`, `debug`, `frontends`,
  `locations`, `commands`, `tools`, `services`, `tasks`, `packages`.
  Profile/scheduling/MCP/update commands are package capabilities unless the
  tracked tree still carries a transitional command.

The pipeline substrate (`pipeline/` — orchestrator, watcher, event_trigger) still
boots, but ships **zero pipeline tasks**: it idles until a pipeline plugin
(extract/chunk/index/embed) is installed.

**Parsers.** The kernel keeps only the two dependency-light parsers:
`parse_text` (UTF-8 / code / CSV / TSV, stdlib) and `parse_image` (standard
rasters via Pillow, already a kernel dep; HEIC lazy). Shared text helpers live
in `parsing_utils.py`. The registry (`parser_registry.py`) carries a static
native-modality default map so `get_modality` resolves image/audio/video even
with no parser installed (attachment routing relies on this). Every heavier
parser is an installable store package (`parser-pdf`, `parser-office`,
`parser-tabular`, `parser-audio`, `parser-video`, `parser-gdoc`,
`parser-container`) that ships a `services/helpers/parse_*.py` file —
**not** a plugin entrypoint. `ParserService._load()` rebuilds the registry by
discovery-scanning `services/helpers/parse_*.py` across the built-in, sandbox,
and installed roots, so installed parsers light up on load; `package_manager`
reloads the parser service on install/uninstall of any such file so it takes
effect live. The attachment system is unified onto this one registry:
`attachments/parse.py` builds an `Attachment` via `parser.get_modality` +
`parser.parse(path, "text")` (no separate attachment-parser registry).

## The kernel boundary (the one rule)

Core code (`pipeline/`, `runtime/`, `state_machine/`, `agent/`, `events/`,
`config/`, `main.pyw`) hard-imports **exactly two** plugin modules. Keep these
two resolvable in any kernel:
1. `service_llm` — `runtime/agent_scope.py` + `runtime/conversation_loop.py`.
2. `parser_registry` — `pipeline/orchestrator.py`, `pipeline/watcher.py`,
   `agent/system_prompt.py`.

Everything else is discovery-based. The agent system prompt gates every optional
section behind `_has_tool(...)` in `agent/system_prompt.py`, so missing tools
degrade silently and correctly.

## Hardening applied for kernel reliability

These edits exist so the kernel degrades cleanly when a stdlib plugin is absent —
the difference between a microkernel and a pile of assumptions:
- **`plugins/services/service_compactor.py`** — context compaction is a
  synchronous service call from the conversation loop, so the kernel does not
  route a blocking request through the event task queue.
- **`runtime/runtime_config.py` `build_loop`** — the "no LLM" path now raises a
  friendly message pointing at `/setup` instead of an opaque error.
- **`config/config_data.py`** — `autoload_services` trimmed to
  `["llm", "compactor", "parser", "plugin_watcher"]`; `enabled_frontends` → `["repl"]`;
  `DEFAULT_SCHEDULED_JOBS` → `{}` (jobs/timekeeper are store plugins now).
- **`requirements.txt`** — kernel-minimal. Optional parser, scheduler,
  frontend, LLM backend, search, and integration dependencies belong to package
  manifests. If `requirements.txt` grows, check whether the dependency is truly
  kernel infrastructure.

## Package store V1

- **Local package store**: `/packages` reads explicit package manifests from the
  `origin/store` git ref with `git show`, copies package files into
  `DATA_DIR/installed_plugins`, loads recorded entrypoints, and writes receipts
  under `DATA_DIR/packages`. Uninstall mirrors install, refuses live dependents,
  and prunes only unneeded auto-installed dependencies.
- **Publishing**: use `python scripts/package_publisher.py publish <id> --name ...`
  with repeatable `--file SOURCE=DEST`, `--require`, and `--update`.
  The script uses a temporary worktree for `origin/store`, validates the whole
  store, commits, and pushes without switching or dirtying the current branch.
- **pip dependencies**: install auto-detects third-party imports in a package's
  `.py` files and `pip install`s the missing ones (mapping import roots to PyPI
  names via `PIP_NAMES`, e.g. `fitz`→PyMuPDF). A manifest may instead declare an
  authoritative `pip` list (publisher `--pip`/`--no-pip`), which overrides the
  scan — the escape hatch for optional/alternative/platform deps the scan can't
  read (e.g. `service-ocr`'s per-OS engines). Uninstall does not pip-uninstall.
- **Bundles (meta-packages)**: a package may ship **no files** and only a
  `requires` list — installing it pulls its members, uninstalling prunes the
  auto-installed ones. Used for the curated bundles (`all-parsers`, `plan-mode`,
  `scheduling`, `web-search`, `gmail`, `mcp`, `google-drive`, `indexing-search`).
- **Deferred**: remote/GitHub fetch, versioning, package config cleanup,
  scheduled-job installs, and containerization.

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
For a hermetic smoke, point DATA_DIR at an empty temporary location first;
otherwise local installed packages will appear in discovery and hide kernel
boundary drift. Expect kernel services, no tasks, and no built-in tools. Then
`python main.py`, run `/setup` to install/configure starter capability, and
confirm a REPL round-trip + clean compaction on a long conversation.

---

## Recent work — state machine unification

The conversation layer was unified around a single state machine
(`ConversationState` in [state_machine/conversation.py](state_machine/conversation.py))
driven by [runtime/conversation_runtime.py](runtime/conversation_runtime.py)
(`ConversationRuntime`). Every frontend action — REPL, installed Telegram, future
background drivers — flows through one labeled `cs.enact(...)` site in
`_dispatch`, mirroring PokerMonster's `run_game`. Agent turns hand off to
`ConversationLoop.drive()`, which has its own labeled enact site for the
agent's moves.

The same primitives now back commands and tools: a `CallableSpec` has a
handler, an optional form (list of `FormStep`), and an optional
`form_factory(args, cs)` for dynamic forms. Forms suspend into a `PhaseFrame` on
the cache stack, surviving restarts via the persistence layer
([runtime/persistence.py](runtime/persistence.py)).

The runtime exposes `runtime.active_session_key` / `active_conversation_id`
so background drivers can identify themselves: anything with a session key
that doesn't match the active one is, by definition, running unattended.
The tool registry uses this to refuse `background_safe=False` tools from
non-active sessions. The scheduled-subagent layer was deleted and is
slated for a clean rebuild on top of these primitives — there is no
`is_subagent` flag in the runtime.

"Is a human present at this session right now?" is asked in exactly three
places (interactive-tool gating, the notify-prompt block, background
notification push) via one reader: `runtime.is_attended(session_key)`. By
default this is just `session_key == active_session_key` (the single-active
rule), but a frontend can override it per session — `RuntimeSession.attended`
(`bool | None`, ephemeral, not persisted), set through
`runtime.set_session_attended` or the `BaseFrontend.mark_attended` /
`mark_unattended` helpers. This is the kernel's hook for **concurrent
multi-user frontends** (e.g. a website marking a session attended on socket
connect, unattended on disconnect): the kernel only *reads* attendance, the
frontend *owns* the policy. Single-user frontends (REPL, installed Telegram) set
nothing and keep `attended=None`, inheriting the global behavior unchanged.

### The user dimension

Sessions also carry an **ephemeral, frontend-bound `user_id`** ("whose data is
this?"), seeded fallback `DEFAULT_USER_ID = 1` (the base user). **Identity
(`user_id`) and authorization (`frontend_profile`) are separate axes** — there is
no privileged "admin" user; the REPL is powerful because its *frontend_profile* is
unrestricted, not because of its user. A frontend **declares** how sessions map to
users via `BaseFrontend.user_binding` (`"single"` ⇒ every session is
`default_user_id`; `"per_user"` ⇒ each identity its own user) + `default_user_id`;
the base auto-binds unbound sessions to that default, and `per_user` frontends call
`bind_session(key, external_id)` / `identify(...)` to upgrade on login. Login itself
is a frontend concern (the kernel ships no crypto — it stores `password_hash`
opaquely). `session.user_id` is **not** persisted
in the marker: ownership lives on `conversations.user_id` (the source of truth), so
identity can never leak in by loading a conversation. Per-user data is the `users`
table (`user_type` label + `config` JSON blob + `username`/`password_hash` columns),
reached anywhere via `context.user_id` / `context.current_user()` / `context.db`.
`user_type` is frontend-defined metadata (guest/admin/paid/creator/etc.), not a
kernel admin bypass; frontends and policy plugins decide what it means. Plugins declare
**user-scoped settings** with `{"scope": "user"}` in a setting's `type_info`; `/config`
reads/writes those against the current user's `config` blob instead of the global
config. The remembered `last_active_conversation_id` also lives in the current
user's config blob, so startup restore is per-user rather than one public/global
pointer. `active_agent_profile` and `skip_permissions` are user-scoped too:
profile definitions remain global, but the user's selected profile and trusted
tool list live with that user. **Conversation ownership is enforced** by `runtime.assert_conversation_access`
on every load/mutate-by-id path (`load_history`, `load_conversation`, `open_session`,
`inject_user_message(..., conversation_id=...)`, `delete_conversation`, `set_conversation_category`,
`set_conversation_notification_mode`) — listing filters are convenience only;
`override=True` (or using the raw `db.*` methods) is the system path.

## Command lifecycle (current)

A command emits two events: `COMMAND_CALL_STARTED` (first invocation, even if
a form will be filled afterward) and `COMMAND_CALL_FINISHED` (after the
handler runs, or on cancel during a form). Same `call_id` across the
lifecycle — pinned to the form's `PhaseFrame.data["call_id"]` so STARTED
and FINISHED match up. See
[state_machine/action.py](state_machine/action.py)
`_CallableAction.execute` and `_run`.

`BaseFrontend` ([plugins/BaseFrontend.py](plugins/BaseFrontend.py)) subscribes
both events and routes them through `render_tool_status(session_key,
payload)`. Rich frontends such as installed Telegram can edit a single status
message in place; the REPL prints the same shapes to stdout.

## Where to plug in

- **Add a slash command**: write a `BaseCommand` subclass as `command_*.py` in
  the sandbox, installed package tree, or deliberately in [plugins/commands/](plugins/commands/)
  when it is true kernel behavior. Commands receive `SecondBrainContext` in both
  `form(args, context)` and `run(args, context)`.
- **Add a tool**: write a `BaseTool` subclass as `tool_*.py` in the sandbox,
  installed package tree, or deliberately in [plugins/tools/](plugins/tools/)
  when it is true kernel behavior. Tools receive `SecondBrainContext` from
  [runtime/context.py](runtime/context.py).
- **Drive an agent from a task**: call `context.runtime.iterate_agent_turn(...)`
  on a session key. The runtime persists history and markers atomically
  for you. Background drivers should keep their session key distinct from
  the active one so the registry's `background_safe` gate kicks in.
- **Let an agent run a slash command**: use an installed command/tool bridge if
  one is present in the current tool catalog. The kernel should not hardcode
  command-running tools for packages it may not ship.

## Command plugins

Slash commands now mirror the rest of the plugin system. The repo starts with a
clean command slate: add built-ins as `command_*.py` files under
[plugins/commands/](plugins/commands/), or create sandbox commands under
`DATA_DIR/sandbox_plugins/commands`. The registry in
[plugins/frontends/helpers/command_registry.py](plugins/frontends/helpers/command_registry.py)
is only the adapter: it builds context-aware forms, parses one-shot `/cmd ...`
input mechanically, and dispatches structured dict args.

## Sandbox plugin system

The agent can author tools/tasks/services/commands/frontends into
`DATA_DIR/sandbox_plugins/<family>/` when an editing/package-authoring tool is
installed and in scope. Shell and file-editing tools are not kernel guarantees.
Sandbox and installed plugins are auto-discovered alongside first-party ones in
[plugins/](plugins/). Plugin helpers should use relative imports so files can
move between built-in, sandbox, and installed trees.

## Files that matter most

- [runtime/context.py](runtime/context.py) — `SecondBrainContext`, the
  shared bag tools/tasks receive.
- [runtime/conversation_runtime.py](runtime/conversation_runtime.py) —
  `ConversationRuntime`, the single dispatcher. This is the accepted "ugly
  duckling" of the codebase.
- [state_machine/action.py](state_machine/action.py) — every
  user/agent action type lives here; one class per action.
- [pipeline/orchestrator.py](pipeline/orchestrator.py) — task scheduling and
  the dependency-pipeline DAG. `runtime` is wired in
  [runtime/bootstrap.py](runtime/bootstrap.py).
- [agent/system_prompt.py](agent/system_prompt.py) — single entry point for
  building the agent system prompt; gates sections by which tools the
  current scope exposes.
