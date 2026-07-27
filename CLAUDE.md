# Second Brain — Architecture Notes

Local-first AI kernel with SQLite persistence, a REPL frontend, package
install/uninstall, and live plugin loading. Python / SQLite. Solo dev (Henry).
The Flet GUI was removed; do not reintroduce.

---

# ⚡ THE KERNEL (READ FIRST)

Second Brain is a **microkernel**: a minimal, reliable core that boots, runs
the conversation loop + agent turn, persists conversations, and loads/unloads
plugins. Product capabilities arrive through a **package store** (the
agentskills.io model: a registry you browse, install, and uninstall from).
Do not bake heavy features into the kernel; they belong in packages.

> Goal in priority order: (1) the kernel works **flawlessly and reliably**, then
> (2) build install/uninstall against a cloud store, then (3) versioning and
> possibly containerization. We are at the end of step (1).

## What ships in the kernel (`plugins/`)

Plugins are discovered purely by file presence (`plugins/plugin_discovery.py`).
The kernel was produced by **moving** non-essential plugins into `store/` (a
staging catalog that mirrors `plugins/`, preserved via `git mv` to seed the
future store) — *not* by deleting them. What remains:

- **Services:** `service_compactor` (context-safety),
  `service_timekeeper` (lightweight event clock), and
  `service_plugin_watcher` (hot-reload = the install/uninstall substrate).
  Parsing was here and deliberately is not any more — it is kernel routing,
  see **Parsers** below. If another tracked service remains, treat it as
  kernel-boundary debt unless the user explicitly keeps it.
- **Tasks:** none.
- **Tools:** none in the tracked kernel tree. `tool_read_file`,
  `tool_ask_user_question`, shell/file-editing tools, SQL tools, and plugin
  authoring tools are package capabilities unless discovery shows they are
  installed.
- **Frontend:** `frontend_repl` only. Telegram and the MCP server
  (`frontend_mcp_server` — exposes Second Brain to external MCP clients over
  streamable HTTP; tested from main via `tests/test_frontend_mcp.py`, which
  loads it off the store ref) live on the store branch. `enabled_frontends`
  is deliberately not whitelisted by the kernel: config normalization keeps
  unknown names so installed store frontends survive load, and bootstrap
  warns/skips what discovery can't resolve.
- **Commands:** REPL UX + introspection only — `config`, `setup` (LLM onboarding
  wizard), `llm`, `conversations`, `clear`, `cancel`, `debug`, `frontends`,
  `locations`, `commands`, `tools`, `services`, `tasks`, `packages`.
  Profile/scheduling/MCP/update commands are package capabilities unless the
  tracked tree still carries a transitional command.

The pipeline substrate (`pipeline/` — orchestrator, watcher, event_trigger) still
boots, but ships **zero pipeline tasks**: it idles until a pipeline plugin
(extract/chunk/index/embed) is installed.

**Parsers.** Parsing is **kernel routing plus importable functions, not a
service** — `parsing/` (`registry.py`, `result.py`). It was a service, and that
forced every result to travel as a live object: a PIL image, a numpy array, an
open `av.Container`. Those cannot cross a process boundary, so nothing that
parsed could be sandboxed, and the least trustworthy code in the system —
foreign C libraries chewing arbitrary files — was the one part that had to run
unmediated in the kernel's process.

The split that fixes it: **text and `container` (child paths) are parse
*results*; image/audio/video/tabular are *intermediates*.** Every intermediate
is on the way to text or to a file, consumed by exactly one specialist that
immediately transforms it. So `ParseResult.crossable` is the line, and code
needing a heavy modality calls `parsing.parser_for(ext, modality)` to pull the
parser into *its own box* alongside the thing that consumes it — the waveform
never crosses anything, the transcript does. `sdk.parse.file` refuses
non-crossable modalities with a message pointing at that route.

**A parser is `parse_x(sdk, path, config) -> ParseResult`** — one signature,
two callers. Inside a box it gets the real SDK and every effect is a Request;
when the *kernel* calls it (`parsing.parse`) it gets `parsing.kernel_sdk.
KERNEL_SDK`, a deliberate in-process stand-in whose `fs.read` reads directly,
because the kernel is already inside the boundary and mediating it against
itself would be theatre. The parser cannot tell which it has, and that is what
makes the same file importable into a box without a second contract. It also
retired the `services` dict that used to be threaded through every call:
delegating parsers now use `sdk.services.call("google_drive", ...)`, and
`parsing.bind_services()` just points the stand-in at the live registry.
`plugins/helpers/parse_text.py` is the migrated reference — it validates
clean and loads in a subprocess box, both pinned by tests.

**The contract lives in the guest** (`sandbox/guest/parsing.py`): `ParseResult`,
`CROSSABLE`, `clean_text`/`max_chars`, and `register`. A parser is guest code,
and the child process runs with `sandbox/` as its cwd and *cannot see the
kernel at all* — so a parser importing a kernel module loads in-process and
fails in a subprocess, which is the case the heavy parsers most need. Kernel
code reaches the same objects through `parsing`, which re-exports them; that
is the one place kernel code imports `sandbox.guest.*`, and it is the guest
*contract*, never host machinery. `register` is a **collector**, not a
registry: a parser calls it at import time, `discover()` drains what
accumulated per module, and a box running the same line simply collects
nothing — which is what keeps one file loadable in both worlds. Parsers must
also avoid `pathlib` (the validator refuses it); match suffixes with
`str.endswith`.

**Parsers live in `helpers/` at a plugin tree's root**, not under a family —
`plugins/helpers/parse_text.py`, `DATA_DIR/installed_plugins/helpers/
parse_pdf.py`. A parser belongs to no family because it is not a plugin: no
base class, no entry point, nothing discovery registers. `helpers/` is a
fourth tree root beside the five families (`plugin_paths.helper_dirs()`,
`package_manager.TREE_ROOTS`), and files there carry no name prefix.

The kernel keeps only the dependency-light `parse_text` parser (UTF-8 / code /
CSV / TSV, stdlib); shared text helpers are `parsing/utils.py`, re-exported
from `parsing` so every parser reaches them by one absolute path regardless of
which tree it landed in. The registry carries a static native-modality default
map so `get_modality` resolves image/audio/video with **no parser installed**
(attachment routing relies on this). Every heavier parser is an installable
store package (`parser-pdf`, `parser-office`, `parser-tabular`,
`parser-image`, `parser-audio`, `parser-video`, `parser-gdoc`,
`parser-container`) shipping a `helpers/parse_*.py` file. `parsing.discover()`
rebuilds the registry by scanning those across the built-in, sandbox, and
installed roots; `package_manager` rescans on install/uninstall so it takes
effect live. `parsing.bind_services()` supplies peers for delegating parsers
(`parse_gdoc` → `google_drive`) — a reference, not a lifecycle.
`attachments/parse.py` builds an `Attachment` via `parsing.get_modality` +
`parsing.parse(path, "text")` (no separate attachment-parser registry).

**The LLM.** Talking to a model is **kernel routing plus installable
backends** — `llm/` — and it got there the same way parsing did, for the same
reason. `service_llm.py` was a central service that *other services imported*:
the store's `service_litellm.py` opened with `from plugins.services.service_llm
import BaseLLM, LLMResponse, ...`, and a file importing a kernel module can
never load in a subprocess. So the least trustworthy code in the system — a
volatile third-party SDK, a network socket and an API key — was the one part
that had to run unmediated in the kernel's process.

Of `service_llm.py`'s 576 lines, roughly 250 were bookkeeping that existed
*because backends were services*: one registered service per model profile,
a resync when a backend file changed, `_mirror_active` copying five attributes
off the default so the router could impersonate it, `/llm` mutating the live
registry as a side effect of editing config. None of it was about talking to a
model, and none of it survives.

What replaces it: a **`Brain`** per configured profile (`llm/registry.py`),
holding settings and a **pool of boxes**. `loaded` means a live process, not a
flag. The pool exists because `PersistentBox.call` serializes under one lock —
one box per profile would queue a scheduled subagent behind a foreground turn —
and it *can* exist because a backend is stateless with respect to the profile:
every model name, key and endpoint arrives on the `LLMRequest`, so any box
serves any call. Its ceiling is `max_workers + 1`, derived rather than chosen,
because a subagent runs on an orchestrator worker and that plus the foreground
turn is the most concurrent calls that can exist.

**A backend is `helpers/llm_*.py`, and belongs to no family** — same as a
parser: no base class the kernel registers, no entry point, nothing discovery
finds. `supports_streaming`, `supports_tool_choice`, `native_modalities` and
`display_name` are **module-level declarations read by AST**, so asking what a
backend can do never costs a provider-library import. The contract lives in the
guest (`sandbox/guest/llm.py`: `LLMRequest`, `LLMResponse`, `BaseLLMBackend`,
`is_context_limit_error`) and `llm/` re-exports it — the classifier is needed on
both sides, since the backend recognises its own provider exception and the
compaction layer classifies what it was handed.

**`ModelRequest.llm` is a model *name*, not an object.** An escort swaps
brains by naming one; the kernel resolves it when the call is placed. Same
handle-not-the-thing move as `<secret:…>`, and it makes native and sandboxed
hooks identical — a sandboxed one could never hold a live model anyway.

**Streaming inverted, and lost a feature on purpose.** `on_delta` was a live
callback whose *return value* aborted the stream; neither half crosses a
boundary. Text now goes out through `sdk.model.delta` — token-scoped like
`model.proceed`, and sent as a one-way `notice` on the wire (still classified
and recorded, just not awaited), because a reply per token would make streaming
from a subprocess slower than not streaming. The abort boolean is simply gone:
stopping is *cancellation*, which the kernel already owns, and a cancelled
guest's next Request raises `Terminated`. A native backend still gets the old
boolean, since it runs in-process and can be told.

**Attachment routing moved kernel-side.** It used to be `BaseLLM.
_prepare_attachments`, which meant every backend inherited a method reaching
into `attachments.*` — a kernel import in the file that most needs isolating.
The loop now splits the bundle against the model's capabilities and the
backend's `native_modalities`, appends the text fallback itself, and the box
receives plain dicts whose bytes the backend reads with `sdk.fs.read_bytes`.

**Backend discovery is sandbox-only.** Every installed provider is a
`helpers/llm_*.py` backend and runs through a `Brain` pool. The deprecated
`plugins/services/service_llm.py` compatibility shim is gone.
`llm.registry.as_brain` still adapts directly injected objects exposing
`chat_with_tools` — test doubles and the stress harness use that seam — but
plugin discovery never imports a native provider into the kernel.

`/llm` gained explicit `load` / `unload` actions. Loading used to be a side
effect of editing a profile; now a brain holds real processes, so opening one
is something the user asks for. Only the default profile loads at boot.

**The store branch needs the matching migration**, five mechanical changes per
parser:

1. `services/helpers/parse_*.py` → `helpers/parse_*.py`
2. imports from `plugins.services.helpers.{ParseResult,parser_registry,
   parsing_utils}` → `from guest.parsing import ParseResult, clean_text,
   max_chars, register` (the **guest**, not `parsing` — a kernel import breaks
   the subprocess path)
3. signature `(path, config, services)` → `(sdk, path, config)`
4. `open(path)` → `sdk.fs.read(path)`; `services["x"].m()` →
   `sdk.services.call("x", "m")`; `logger.*` → `sdk.log`
5. drop `pathlib`; declare `dependencies_pip` for the heavy library and
   `isolation = "subprocess"`

Run `sandbox.validator.validate_file` on each: `conforms.` means it will load
in a box. Nothing shims the old paths — the files have to move anyway.

## The kernel boundary (the one rule)

Core code (`pipeline/`, `runtime/`, `state_machine/`, `agent/`, `events/`,
`config/`, `attachments/`, `main.pyw`) hard-imports **zero** plugin modules.
Every capability, without exception, arrives by discovery.

It was two, then one, now none, and each step down worked the same way: the
*routing* moved into the kernel and the *implementations* became installable
helpers. Parsing went first (`parsing/`, see **Parsers**), the LLM followed
(`llm/`, see **The LLM**). Both times the boundary got narrower **by adding
kernel code** — worth remembering the next time this rule looks like it needs
widening. The question to ask is not "may core import this plugin?" but "is
the part core actually needs standing knowledge, and is the rest a helper?"

This rule is executable: `tests/test_kernel_boundary.py` AST-walks every core
module and pins the complete set of `plugins.*` import edges (the plugin
substrate only, including lazy function-local imports), and asserts the
sanctioned-implementation list is empty. Widening the boundary fails the suite
until the test's allowlist — and this section — are updated deliberately.

Everything else is discovery-based. The agent system prompt collects optional
guidance from each in-scope plugin's `agent_prompt_for(ctx)` (see `_collect` in
`agent/system_prompt.py`), so missing plugins degrade silently and correctly —
uninstalling a package removes its prompt text with it.

## Hardening applied for kernel reliability

These edits exist so the kernel degrades cleanly when a stdlib plugin is absent —
the difference between a microkernel and a pile of assumptions:
- **`plugins/services/service_compactor.py`** — context compaction is a
  synchronous service call from the conversation loop, so the kernel does not
  route a blocking request through the event task queue. Its trigger lives in
  the loop's own **compaction layer** (`_compaction_layer`), a kernel escort
  always stacked inside registered `model_call` escorts (onion: registered
  escorts → context guard → empty-response nudge → backend) — context safety
  is hook-shaped but never registry-dependent. Reactive overflow retries
  rebuild the prompt from compacted history and re-enter the inner onion, so
  they keep the post-escort brain, bus events, and provider params.
- **`runtime/runtime_config.py` `build_loop`** — the "no LLM" path now raises a
  friendly message pointing at `/setup` instead of an opaque error.
- **`config/config_data.py`** — `autoload_services` trimmed to
  `["llm", "timekeeper"]` (extension services auto-load when installed);
  `enabled_frontends` → `["repl"]`;
  `DEFAULT_SCHEDULED_JOBS` → `{}` (no default jobs; `service_timekeeper` ships
  in the kernel tree, and scheduled-job *consumers* are store plugins).
- **`requirements.txt`** — kernel-minimal. Optional parser, scheduling-consumer,
  frontend, LLM backend, search, and integration dependencies belong to package
  metadata. If `requirements.txt` grows, check whether the dependency is truly
  kernel infrastructure.

## The action ledger

The kernel's flight recorder: an append-only `action_ledger` table
(`pipeline/database.py`) recording **every action the system takes**, so
unattended operation is auditable and anything is reconstructable after the
fact. Three origins:

- `user_enact` — written at the labeled enact site in
  `ConversationRuntime._dispatch`.
- `agent_enact` — written by `ConversationLoop._enact_logged`, the gateway
  all agent-side enacts flow through (tool calls, send_text, end_turn,
  over-budget summaries).
- `system` — acts outside the state machine: package install/uninstall
  (with provenance: store commit + per-file SHA-256, recorded by
  `package_manager` — the seed of future versioning), `config_save`
  (changed key **names** only, never values), and conversation lifecycle
  ops including **refused** cross-user attempts (`ok=0, access_denied`).

Failure policy: ledger writes are best-effort at every layer
(`db.record_action` swallows + logs; `runtime/ledger.py` helpers tolerate
missing/stubbed dbs) — the ledger observes the system and must never break
it. Rows are capped (`LEDGER_JSON_CAP`, truncation wrapper stays valid
JSON); no FKs on purpose so audit rows outlive what they describe.

Retention is the **single** `data_retention_days` setting (0 = keep
forever): `Database.prune_expired` deletes everything that accumulates
without bound — ledger rows, idle conversations (messages cascade; any new
message resets a conversation's clock), finished `task_runs` — once at
bootstrap plus a cheap ledger-only sweep on writes. The prune itself is
ledger-recorded. Don't add per-table retention knobs; fold new unbounded
tables into this one.

The ledger is write-optimized filler by volume — read it *targeted*
(by conversation_id / session_key / origin), never linearly; agent-facing
guidance lives in the store `sb-troubleshooting` skill, not the kernel
prompt. The stress oracle checks recent-row well-formedness
(`_check_ledger` in `stress/invariants.py`); tests in
`tests/test_ledger.py`. Query/inspection UX (`/ledger`) is deliberately a
future store package, not kernel.

## Package store V1

- **Tree mirror, not package archives.** The `origin/store` branch mirrors what
  `DATA_DIR/installed_plugins` would look like if every optional plugin/helper
  were installed: `tools/tool_*.py`, `services/service_*.py`,
  `frontends/frontend_*.py`, `commands/command_*.py`, `tasks/task_*.py`, plus
  family-local `helpers/` files. `/packages install <stem>` and
  `/packages uninstall <stem>` target the file stem (`frontend_telegram`,
  `parse_pdf`, `bundle_starter`, etc.).
- **Dependency metadata lives in code.** Plugin base classes expose
  `dependencies_files` and `dependencies_pip`; helpers use the same names as
  module-level literal lists. The package manager reads these fields with AST
  parsing, never by importing store files.
- **Install is a tree copy.** `/packages` reads the target file from
  `origin/store`, recursively follows `dependencies_files`, runs `pip install`
  for collected `dependencies_pip`, and copies the same relative paths into
  `DATA_DIR/installed_plugins`. The store copy always wins: a differing
  existing file is overwritten in place (no versioning yet — the store branch
  is assumed to hold the newest version); byte-identical files are skipped.
- **Uninstall scans live trees.** Uninstall follows the installed target's
  dependency metadata, scans built-in, sandbox, and installed plugin trees, and
  removes only candidate files/pip packages no remaining file still declares.
  Kernel requirements are never pip-uninstalled. Bundles are cloud-only
  manifests in `origin/store` that list store-relative files and feed the same
  resolver. Config cleanup, SQL table cleanup, and versioning are deferred.

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
Plus the two kernel authorities, which discover independently of plugins:
```bash
python -c "import parsing, llm; from config import config_manager; c=config_manager.load(); print('parsers:', parsing.discover(), 'backends:', llm.discover(), llm.backend_names()); llm.refresh(c); print('brains:', llm.describe())"
```
For a hermetic smoke, point DATA_DIR at an empty temporary location first;
otherwise local installed packages will appear in discovery and hide kernel
boundary drift. Expect kernel services, no tasks, and no built-in tools. Then
`python main.py`, run `/setup` to install/configure starter capability, and
confirm a REPL round-trip + clean compaction on a long conversation.

---

# 🧪 THE SANDBOX (`sandbox/`)

**Naming warning, read this first.** Two unrelated things are called "sandbox"
here. `DATA_DIR/sandbox_plugins/` is the *agent-authored plugin tree* (see
"Sandbox plugin system" below). `sandbox/` is the **security boundary** — the
subject of this section. They have nothing to do with each other.

Plugins are arbitrary code on the other side of a boundary. The sandbox
mediates it: sandboxed code **cannot act, it can only ask**. Anything touching
disk, network, clock, or process is a typed **Request** the kernel classifies,
executes, and answers. The threat model is *carelessness, not malice* — an
agent with good intentions and no judgement — which is why the validator is a
linter rather than a proof, and why the subprocess (not the linter) is the
actual boundary. Design rationale: `The Second Brain Security Contract` +
`SECURITY_CONTRACT_APPENDIX.md`.

**Two halves, and the one rule.** `sandbox/guest/` runs *inside*: the Request
vocabulary, the wire protocol, the SDK, the plugin base classes, the child
entry point. Stdlib-only and self-contained — it is the shippable unit a
container image would copy. Everything else is *host*: `policy` (the single
`classify()` that decides safe/unsafe), `handlers` (the only code that touches
the world), `interpreter` (serial gate, parallel execution), the two runners,
`boxes`, `facade`, `bridge`, `validator`, `parity`, `migrate`. **The guest never
imports the host** — pinned by `tests/test_sandbox_guest_boundary.py`, the
sandbox's counterpart to the kernel boundary test.

`sandbox/__init__.py` aliases the guest package under the bare name `guest` in
`sys.modules` (every submodule, derived from the directory). Plugin source
therefore says `from guest.bases import BaseTool` and resolves identically
in-process and in a subprocess, where the child runs `python -m guest.child`
with `sandbox/` as its cwd.

**Boxes.** A box is one execution context: one process, one memory space, one
lifetime. Files in the same box import each other; files in different boxes
cannot reach each other at all — the only way across is a Request. Declaring
nothing gets an ephemeral in-process box of your own. Services and frontends
are persistent (loaded once, called into, serialized one call at a time).
Persistence is resolved *before* code runs, so nothing can drift into it by
refusing to finish — that is a hang, and it times out.

**Ending code is the kernel's decision**, escalating ask → starve → kill.
Starvation only reaches code that propagates failures; killing reaches the
rest, and in-process there is no kill. A cancelled execution raises
`Terminated` (a `BaseException`, so a bare `except Exception` cannot swallow
it) rather than returning a failure — denial and cancellation are different
things.

**Provenance.** Every Request carries a chain rooted in what *caused* the work
(`user`, `cron:nightly_index`, a subagent). The kernel owns it as its own call
stack, so plugins can neither read nor misstate it; it is what makes an
approval dialog answerable, and it doubles as the cycle detector. Approval
reuses the kernel's existing `vet_permission` doorway (enriched with
`origin="request"` plus the typed `request`/`chain`/`decision`), then
`skip_permissions`, then a dialog; unattended sessions refuse rather than block.

**An approval is scoped to what the command declared.** `Chain.approved` is a
frozenset of Request types — the command's own `requests` list, read by AST —
never a boolean. It was a boolean, and that made one "yes" to `/update`
authorize all 87 Requests including egress and plugin installation. The grant
is the *declaration* rather than an argument allowlist because that is the
only decidable question: predicting what `git pull` does is Rice's theorem,
while asking whether the command said it runs a shell is set membership. It is
also the honest reading of what the user answered — they approved a command,
and a command's scope is the capability classes it declared. `push` copies the
grant down unchanged, so a callee can never widen it. This made `requests`
load-bearing after a long career as documentation, so the validator now checks
every name against the closed Request vocabulary (`_check_requests`) — the
audit that motivated it found `/setup` declaring `path.get`, which is not a
Request type and never was. The dialog states the grant rather than the
command name (`approval.describe_grant`, rendered by the bridge from the same
declaration) — a scope nobody is shown is not consent.

**Services are resident boxes.** A sandboxed `BaseService` bridges to a native
one whose `_load()` opens a persistent box and whose `unload()` closes it.
Methods named in `exports` become real attributes on the adapter, because
native callers reach a service by attribute access (`services.get("x").m()`),
not through `service.call`. The synthetic module supplies `build_services`,
since that is how discovery finds services. The box owns the start deadline,
so the adapter sets `load_timeout = 0` rather than race two timers.

**Resident polling is shared infrastructure.** Services and frontends may
define `poll(self, sdk)` and a positive `poll_interval`; the kernel owns the
thread and drives the serialized box call. Truthy drains immediately, falsy
waits the interval, and `max_poll_failures` (default five) bounds repeated
errors. Services default to polling disabled; frontends retain their 50 ms
default. This is an inbound lifecycle contract, not an `sdk.poll()` request.

**Frontends are resident boxes the kernel *drives*.** All five families are now
bridged. A frontend is a residency like a service, but with the loop inverted:
a native frontend blocks in `start()` forever, and a box takes one call at a
time, so a guest that never returned from `start` would hold its box and no
render could get in — the frontend would go deaf the moment it started
listening. So the guest's `start` sets up and returns, and `_adapt_frontend`
runs the loop on the daemon thread `FrontendManager` already gives it, calling
`poll` repeatedly (truthy = "did work, call me straight back"; falsy = pause
`poll_interval`). Between polls is when a render lands. Five consecutive poll
failures stop the frontend rather than spinning on a dead box.

`BaseFrontend` itself is **not** migrated and should not be: its 880 lines are
host-side routing — fourteen bus subscriptions funnelling into nine `render_*`
methods, and `submit_*` funnelling into `runtime.handle_action`. The base owns
*when*, the guest owns *how*, so the base becomes the adapter. The nine
`render_*` collapse to one `render(kind, payload)` box call (`sandbox/
frontends.py` holds the `KINDS` both sides must agree on); `capabilities`
crosses as a literal dict and is rebuilt into `FrontendCapabilities`.

The inbound half — `sdk.frontend.submit_text/submit_attachment/submit_action/
cancel/bind/attended/resolve` — is five Requests scoped the same way
`model.proceed` is: **by reachability, not by a verdict.** The adapter is
parked at a *desk* under a token when its box opens, every Request carries it
back, and it resolves to that adapter and no other. A tool importing the same
namespace holds no token and is refused; the desk is cleared at stop, so a
leaked token reaches nothing. `frontend.bind` sits here rather than with
`user.write` deliberately — approving your own login would make a `per_user`
frontend unusable, and which native path runs is decided by whether an
`external_id` was named rather than by the plugin picking a method.

**The console is the kernel's, lent to one frontend.** `input()` is refused and
stays refused, for three compounding reasons: it blocks (holding the box, so
the frontend cannot render until the next keypress), a subprocess box's stdin
*is* the wire protocol (reading it eats the frames the box talks over), and a
rule that works in-process and corrupts the transport under isolation is the
worst kind. `sandbox/console.py` inverts it the same way the poll loop
inverted — a kernel thread reads stdin into a bounded buffer and
`sdk.console.read_line()` drains it without blocking, so a console frontend can
be **subprocess-isolated**, which `input()` could never allow. The reader takes
any iterator of lines, so tests drive a console without a terminal.

Exclusive by declaration: `uses_console = True`, first claimant wins, second is
refused — two readers would split a person's keystrokes non-deterministically,
presenting as dropped characters. Release names the token so a frontend that
already lost the claim cannot revoke its successor's. `sdk.md.plain()` is the
guest's copy of the kernel's `render_plain` (pure; byte-identical, pinned by a
test) for monospace rendering.

**Hooks are declared, not registered.** A service names doorways in
`hooks = {moment: method}`, read by AST like `exports`. The bridge stands a
shim at each and removes it on unload — a hook cannot leak, because the plugin
never registered one. Payloads are **projections**, not encoded kernel
objects: `guest/hooks.py` holds the guest vocabulary and `sandbox/hooks.py` is
the only place that knows both. Every failure mode (unloaded service, dead
box, raising method, unrecognised verdict) collapses to `None` — abstention —
so a sandboxed hook can never break a turn. Two consequences are load-bearing:
a scope shaper sees tool *names* and can only narrow (widening is
`sdk.session.add_tool`), and `ModelRequest.llm` is a model *name* the kernel
resolves, the same handle-not-the-thing move as `<secret:…>`.

**The bus, inbound, is declared the same way.** `sdk.events.emit` always
worked — publishing is an ordinary Request — but sandboxed code could not
*hear*. A plugin now names `subscribed_channels = [...]` and writes one
`on_event(sdk, channel, payload)`; the bridge subscribes at load and drops
every listener at unload, so a subscription cannot outlive its plugin (a leak
with no symptom: deliveries land on a dead box and are swallowed forever).
`sandbox/events.py` is the host half, `project()` the counterpart to hook
projection — it strips `bus.request`'s live `threading.Event` and result list,
so a sandboxed subscriber sees a round-trip event as fire-and-forget and cannot
stall a publisher it was never trusted to answer. Two rules are enforced
rather than documented: **only services and frontends may subscribe** (a tool
is a call that ends), and channel names are **not** validated against
`events/event_channels.py` — that file is explicit that plugins own their own
channels, so an allowlist would refuse one plugin listening to another.

`sdk.model.proceed` is the sole Request whose handler is a **per-call closure**
rather than a static table entry: an escort's `proceed` is parked host-side
under a one-shot token for exactly the duration of one doorway visit. Code
holding no token reaches no call, which is why the Request is refused outside
a `model_call` hook rather than being ambient authority.

**The dual-mode loader is how migration works.** `plugins/plugin_discovery.py`
→ `_load_plugin_module` asks `sandbox.bridge.adapt()` first. A file importing
`guest.bases` gets wrapped in a *native-looking adapter* subclassing the real
`BaseTool`/`BaseTask`/`BaseCommand`; everything downstream registers and calls
it unchanged. Unmigrated plugins load exactly as before. **Migrated and native
plugins coexist, so the app works at every point in the migration** — one file,
one commit, `git checkout` to revert. Detection is by AST, never by importing.

**The kernel boundary is unchanged.** Core hard-imports no plugin module at
all. `sandbox/` is reached only from `plugin_discovery`, and nothing
in `sandbox/` imports `plugins.*` except the bridge (which needs the native
base classes to subclass).

**The SDK idiom** — Requests return their value and raise on failure; a bare
return is wrapped:

```python
def run(self, sdk, path):
    return len(sdk.fs.read(path).split())     # no Result to unwrap
```

`sdk.Denied` (refused) subclasses `sdk.Failed` (anything). `sdk.ok(x,
llm_summary=...)` only when attaching extras. `sdk.log(...)`, never `logging`.

**Secrets.** A config setting holding a credential is *named* `secret_*` —
that prefix is the declaration, matching how the rest of the system declares
things by name. Those read back as `<secret:name>` handles which the kernel
substitutes inside `net.http`, so code uses a credential it never held.
Environment variables are guessed by name instead, because nothing declares
them. `sdk.secrets.reveal(name)` gets plaintext for foreign libraries that do
their own I/O; a plugin reading its *own* declared setting is not asked
(configuring it was the consent), anyone else is. A credential inside a foreign
library is past the kernel's reach — accepted, documented, would need real OS
containment to fix.

**Docs:** `SDK.md` (hand this to an agent writing sandbox code — its examples
are executed by `tests/test_sdk_docs.py`), `MIGRATING_PLUGINS.md` (the
per-plugin procedure), `SECURITY_CONTRACT_APPENDIX.md` (the ~87-Request
catalogue with policy inputs).

**Migration tooling:** `sandbox.migrate.plan(path)` reports what converting a
plugin involves, line by line, with the Request each effect becomes.
`sandbox.parity.compare(path, entry, ...)` runs the working tree against
`git show HEAD:<path>` with the *same context object* and diffs the return
values — so **commit before migrating**, since HEAD is the baseline. Templates
in `templates/` still teach the old contract and should be migrated before any
plugin, since they are what gets copied.

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
non-active sessions. The scheduled-subagent layer was rebuilt on these
primitives and ships as the store Scheduling bundle (`command_schedule`,
`task_spawn_subagent`, `tool_schedule_subagent`): timekeeper jobs emit a
bus event, the spawn task opens its own `spawn_subagent:<cid>` session and
drives `runtime.iterate_agent_turn(...)`. There is no `is_subagent` flag in
the runtime — a subagent is just a session whose key isn't the active one.

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

## Presentation convention: markdown on the wire

Command/tool output is a **string of GitHub-flavored markdown**, built with
the primitives in
[plugins/frontends/helpers/formatters.py](plugins/frontends/helpers/formatters.py):
`md_table` for data tables, `detail_card(title, pairs)` for describe-style
key/value cards, `quote_block` for prose under a card (descriptions,
previews, payloads), and fenced code blocks for multi-line technical dumps
(/debug, /locations — rich renderers collapse single newlines in prose).
Tables must start their own block (blank line before), or GFM parsers fold
them into the preceding paragraph. Each frontend then renders by policy, not
by sender: the REPL runs `render_plain` (aligns tables, strips fence
markers); Telegram's rich path renders markdown natively but compacts
detail-card-shaped tables into code blocks, and its HTML fallback renders
tables/quotes as `<pre>`/`<blockquote>`. Don't invent a structured message
type for this — markdown is deliberately the interchange format (it is also
what the LLM emits, so frontends need exactly one rendering path).

`BaseFrontend` also exposes optional per-frontend polish hooks:
`render_queued_ack` (suppress the textual mid-turn ack in favor of e.g. a
message reaction) and `render_conversation_banner` (mirror the session's
conversation title on a persistent surface; fed by the
`SESSION_CONVERSATION_CHANGED` bus channel).

## Where to plug in

- **Add a slash command**: write a `BaseCommand` subclass as `command_*.py` in
  the sandbox, installed package tree, or deliberately in [plugins/commands/](plugins/commands/)
  when it is true kernel behavior. Commands receive `SecondBrainContext` in both
  `form(args, context)` and `run(args, context)`.
- **Add a *sandboxed* tool** (the direction of travel): write a `BaseTool`
  subclass from `guest.bases` as `tool_*.py`. It receives `sdk`, not
  `context`, and the bridge registers it like any other tool. See `SDK.md`.
- **Add a tool** (the pre-migration contract): write a `BaseTool` subclass as `tool_*.py` in the sandbox,
  installed package tree, or deliberately in [plugins/tools/](plugins/tools/)
  when it is true kernel behavior. Tools receive `SecondBrainContext` from
  [runtime/context.py](runtime/context.py).
- **Bend a per-turn kernel decision**: register a hook from a service's
  `bind_runtime`/`_load` via `runtime.hooks.add(moment, fn)`
  ([runtime/hooks.py](runtime/hooks.py), worked examples in
  [templates/hook_template.py](templates/hook_template.py)). The agent turn
  is a fixed ritual with a doorway at every moment, and nothing influences a
  turn except through a doorway. Six moments, one contract (`fn(ctx,
  payload)`; return `None` to abstain; raising hooks are logged and skipped):
  `turn_start` (adjuster — pre-drive injection: prompt extras, staged
  attachments, queued actions; skipped on restart re-drives; keep fast),
  `shape_scope` (adjuster — inject/hide tools per session), `vet_permission`
  (verdict — allow/deny sensitive calls; asked at two stages, `"approval"`
  for sensitive commands and `"unattended_call"` for interactive tools in
  unattended sessions, where the kernel's default on abstain is to refuse), `model_call` (**escort** —
  `fn(ctx, request, proceed)` owns the round trip to the model: rewrite the
  `ModelRequest` (swap `request.llm`, edit messages, set `tool_choice` on
  backends with `supports_tool_choice`), place the call, inspect the
  response, retry — subsumes the old LLM-selector mechanism and the old
  restart-to-swap-brains dance), `end_turn` (verdict — the doorman at the
  exit: `Allow` / `SendBack(note)` / `RequireTool(name)` / `Redrive()`,
  hard-capped at `DOORMAN_FIRE_LIMIT` interventions per turn so a doorman
  can never trap the agent; the kernel's over-budget wrap-up is itself the
  default doorman at `reason == "budget_exhausted"`), and `turn_finish`
  (observer — fires once per logical turn with a `TurnOutcome`). Hooks can
  also queue tool calls onto `session.pending_agent_actions` (drained at
  loop boundaries through the normal enact/ledger path).
  `session.restart_turn = True` remains the mid-turn spelling of `Redrive()`
  for tools. Hooks run in registration order (= plugin load order); a
  priority knob is deliberately deferred until two plugins actually conflict. Every agent enact ledger row records the driving model in
  `data_json.llm` (post-escort) and doorway-forced acts carry
  `data_json.hook`.
- **Ship a task with a schedule**: declare `default_jobs` on the task
  (`{job_name: {"channel", "cron", "payload"}}`). The orchestrator seeds the
  Timekeeper job at registration if absent (disabled jobs count as existing)
  and removes it at unregistration, so default jobs live exactly as long as
  their task and a reinstall picks up an updated declaration. Disabling —
  not deleting — is the durable way to silence a default job.
- **Observe finished turns** (learn-from-outcome loops, memory writers):
  subscribe to `SESSION_TURN_COMPLETED` — emitted once per logical turn from
  the drive site, foreground and background alike, with `ok`/`cancelled`/
  `user_id`/`final_text`/`new_messages` (restart re-drives don't emit; crashes
  emit with `ok: False`). Bus handlers run on the drive thread, so heavy
  consumers should be pipeline tasks with `trigger="event"` on the channel —
  the event trigger queues a `task_runs` row and the orchestrator does the
  work off-thread.
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

## Sandbox plugin system (the *plugin tree*, not `sandbox/`)

Unrelated to the security sandbox above — this is where agent-authored plugins
live on disk.

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
- [parsing/registry.py](parsing/registry.py) — the file-type authority:
  routing, discovery, and `parser_for` (the importable half). Not a service,
  on purpose.
- [llm/registry.py](llm/registry.py) — the model authority: profiles to
  `Brain`s, the box pools, load/unload, and the dual-mode adapter that keeps
  unmigrated backends working. Not a service, for the same reason.
- [agent/system_prompt.py](agent/system_prompt.py) — single entry point for
  building the agent system prompt; gates sections by which tools the
  current scope exposes.
- [sandbox/policy.py](sandbox/policy.py) — `classify()`: the entire
  authorization surface for sandboxed code, plus `Chain` (provenance).
- [sandbox/interpreter.py](sandbox/interpreter.py) — the drive loop the whole
  sandbox hangs off: serial gate, parallel execution.
- [sandbox/facade.py](sandbox/facade.py) — `Sandbox`: the one API.
  `run()` blocks, `start()` returns a `Run` to wait on or cancel, `open()`
  loads a resident box.
- [sandbox/bridge.py](sandbox/bridge.py) — the dual-mode loader that lets
  migrated and unmigrated plugins coexist.
- [sandbox/guest/sdk.py](sandbox/guest/sdk.py) — what plugin authors actually
  type. Each namespace is exactly one Request family.
- [sandbox/hooks.py](sandbox/hooks.py) — the two-way translation between the
  kernel's doorways (live objects) and the guest's (plain data), plus the
  escort's parked-closure mechanism.
