# Second Brain — Architecture Notes

Local-first AI kernel with SQLite persistence, a REPL frontend, package
install/uninstall, and live plugin loading. Python / SQLite. Solo dev (Henry).
The Flet GUI was removed; do not reintroduce.

This file is the deep architecture map for changing the kernel. It is not the
authoring contract for sandbox code. Before writing a script or extension,
read `docs/SDK.md` and the matching executable template in `templates/`; use
the code pointers there for the specific subsystem. For permission questions,
start with `docs/PERMISSIONS_MAP.md`. Current code wins over historical notes
in this file when they disagree.

---

# ⚡ THE KERNEL (READ FIRST)

Second Brain is a **microkernel**: a minimal, reliable core that boots, runs
the conversation loop + agent turn, persists conversations, and loads/unloads
plugins. Product capabilities arrive through a **package store**: a registry
you browse, install, and uninstall from.
Do not bake heavy features into the kernel; they belong in packages.

> Goal in priority order: (1) the kernel works **flawlessly and reliably**, then
> (2) build install/uninstall against a cloud store, then (3) versioning and
> possibly containerization. We are at the end of step (1).

## The layout (`trees.py`)

Two facts decide where any piece of extension code lives, and both are declared
in **one table** at the repo root, beside `paths.py`.

**Who finds it.** A root whose files carry a *prefix* is scanned — something
globs `f"{prefix}*.py"` and indexes what it finds. A root with no prefix is only
ever reached by something naming the file. Eight roots:

| Root | Prefix | Found by |
|---|---|---|
| `tools/` `tasks/` `services/` `commands/` `frontends/` | `tool_` … | `plugin_discovery` |
| `parsers/` | `parse_` | `parsing.discover()` |
| `llm/` | `llm_` | `llm.discover()` |
| `scripts/` | — | `script.run` / `isolation.is_script` |

**Who put it here.** Four trees holding the same eight roots: `bundled/` (ships
with the app), `DATA_DIR/installed` (the store's), `DATA_DIR/workspace` (the
agent's), and `origin/store` itself, which is the same shape reached over git.
Discovery precedence is bundled → installed → workspace and **first match
wins**; resolution *by filename* (`isolation.resolve_script`) deliberately runs
the other way, because there the agent means the file it wrote.

**A root is declared only when the kernel itself routes it.** That is the test
to apply before adding a ninth: it is a claim that core code needs standing
knowledge of the folder, the same question the kernel boundary asks.
`bundles/` fails it — it exists because of store packages, the kernel does not
name it, and `package_manager` keeps handling it on its own. `workspace/memory/`
fails it differently: the kernel *does* name it (`agent.system_prompt._agent_
memory` inlines `MEMORY.md` into the prompt), but only ever in the one tree the
agent writes. A root is a shape every tree holds, and a bundled or installed
`memory/` would mean nothing. It lives inside `workspace/` so the standing
free-write grant covers the notes the prompt asks the agent to maintain —
outside it, every save raised a dialog for a file the system itself asked for;
`migrations._move_memory` moves an older top-level `memory/` in at boot.
`skills/` was a third, and is gone entirely: it was the only store package that
was a *folder* rather than a file, which cost the package manager ~50 lines of
bespoke handling — folder-as-package install, SKILL.md frontmatter dependency
parsing, folder-prefix uninstall — for a concept nothing in the kernel routed.
Removing the packages removed the plumbing with them.

**There is no top-level `helpers/`.** A helper exists to help a plugin, so it
lives inside the family it helps (`<tree>/tools/helpers/x.py`) — the one nested
folder the layout allows. The root existed because parsers and backends had
nowhere else to go, which made "not a plugin" the definition of a folder two
kernel registries were scanning. A helper shared by *two* families now has no
home: promote it to a service, or let the owning family hold it. Do not
resurrect the root.

`plugins/` is **exclusively substrate** — `native/` (the five adapter base
classes), discovery, the watcher, `plugin_paths`, `command_registry`. It is
not a tree. That is what lets
`tests/test_kernel_boundary.py` treat any `plugins.*` import as allowed
instead of maintaining a nine-entry allowlist by hand; what keeps it honest is
`test_the_plugins_package_holds_no_implementations`, which fails on a
tree-root-named folder or a class subclassing one of the five bases. `native/`
passes both — it is not a root name, and the bases do not subclass each other.

`migrations.py` moves an older DATA_DIR to this layout at boot — idempotent,
stdlib-only, and it refuses to guess: anything in an old `helpers/` that is not
a parser or a backend is left in place and logged.

**`trees.materialize()` then makes the layout real**, every root in every local
tree, at boot right after the migration. The table is a *claim about where
things go*, and a folder that only appears once something lands in it does not
make that claim to anybody. The only code that had ever created a directory was
the watcher, which iterates `watched_only=True` — so `scripts/`, the safe
alternative to `proc.run` we most want reached for, existed in no tree at all;
`bundled/` was skipped entirely on the reasoning that the source tree is the
developer's; and `/locations` showed three trees with three different folder
lists and no way to tell which difference meant anything. Uninstalling then
deleted whatever was left empty, so which folders `installed/` had depended on
what you happened to have installed and in what order you removed things.
`package_manager._remove_empty_dirs` now stops at `trees.is_root_dir`, and
`plugins.list(source="families")` derives the store's category menu from
`ROOTS` plus `EXTRA_FAMILIES` rather than a hardcoded six that silently
discarded `llm` and `parsers`.

## What ships in the app's tree (`bundled/`)

Plugins are discovered purely by file presence (`plugins/plugin_discovery.py`).
The kernel was produced by **moving** non-essential plugins into `store/` (a
staging catalog that mirrors `plugins/`, preserved via `git mv` to seed the
future store) — *not* by deleting them. What remains:

- **Services:** `service_compactor` (context-safety),
  `service_timekeeper` (lightweight event clock), and
  the kernel-owned `plugins.plugin_watcher` (hot-reload = the
  install/uninstall substrate).
  Parsing was here and deliberately is not any more — it is kernel routing,
  see **Parsers** below. If another tracked service remains, treat it as
  kernel-boundary debt unless the user explicitly keeps it.
- **Tasks:** none.
- **Tools:** none in the tracked kernel tree. `tool_read_file`,
  `tool_ask_user_question`, shell/file-editing tools, SQL tools, and plugin
  authoring tools are package capabilities unless discovery shows they are
  installed.
- **Frontend:** `frontend_repl` only. Telegram (`frontend_telegram`, migrated
  to the SDK) and the MCP server (`frontend_mcp_server` — exposes Second Brain
  to external MCP clients over streamable HTTP, and **not** yet migrated: it
  still imports `logging`, `pipeline.database` and
  `state_machine.conversation_phases`) live on the store branch. Testing them
  is split by whose behaviour is under test. What the *kernel* claims about
  them — the validator's verdict, the declarations the bridge reads, the
  isolation the tree resolves — is `tests/test_store_frontend_contracts.py`
  and runs by default. Their own behaviour (markdown rendering, chunking, the
  streamed-reply tracker, media planning, MCP session identity) is marked
  `store` in `pytest.ini` and deselected, since a kernel change cannot break
  it; run it with `pytest -m store`. Both reach the store branch through
  `tests/support.store_source`, which prefers a store *worktree* when the
  clone has one, so it checks the file being edited rather than the last
  commit. `enabled_frontends`
  is deliberately not whitelisted by the kernel: config normalization keeps
  unknown names so installed store frontends survive load, and bootstrap
  *prunes* what discovery can't resolve — a name it cannot match is a store
  frontend that is no longer installed, and warning about it on every boot
  forever taught the user to ignore boot warnings. Normalization still keeps
  unknown names, because at that point the answer isn't known yet.
- **Commands:** REPL UX + introspection only — `config`, `setup` (LLM onboarding
  wizard), `llm`, `conversations`, `clear`, `cancel`, `debug`, `frontends`,
  `locations`, `commands`, `tools`, `services`, `tasks`, `packages`,
  `permissions`, `mode`, `schedule`, `quit`, `restart`. The last two used to be
  native `_HostCommand` instances built in the composition root, holding
  `shutdown_fn` and the scaffold directly; they are ordinary sandboxed
  commands over the `app.stop` Request now, which is what parity with the
  other eighteen means.
  `schedule` is kernel because it manages *any* Timekeeper job and the
  Timekeeper is a kernel service — a store command was the only way to reach
  two things the kernel already owned. `permissions` and `mode` are kernel for
  the same shape of reason: one lists and revokes the three standing-grant
  settings the policy reads, the other sets the standing answer the approver
  gives, and a safety surface that stops working when a package is uninstalled
  is worse than none. Between them they are the two commands that answer "what
  is allowed here" — one scoped to destinations, one to time.
  Profile/MCP/update commands are package capabilities unless the
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
needing a heavy modality pulls the parser into *its own box* alongside the
thing that consumes it — the waveform never crosses anything, the transcript
does.

**Getting it there is a declaration, not an import** (`parse_modalities`).
Kernel-side that route is `parsing.parser_for(ext, modality)`; a *box* cannot
use it, because the child runs with `sandbox/` as its cwd and `import parsing`
is a `ModuleNotFoundError` — so for a long while sandboxed code had no route to
a heavy modality at all. A plugin now declares `parse_modalities = ["image"]`,
`parsing.sources_for` resolves that against the live registry, and the resolved
*files* are imported into the plugin's box ahead of its entry
(`guest.loader.install_parsers` → `guest.parsing.adopt_registrations`).
`sdk.parse.file` then finds a local route and calls it; finding none it falls
through to the Request, which still refuses a non-crossable modality — but now
names the declaration that fixes it.

Three properties are load-bearing. **The kernel resolves**, because which files
provide `"image"` is a fact about what is installed and a box cannot know it —
so naming a capability can never reach a file the kernel would not have
offered. **Declaring tightens isolation**: provisioned parsers are foreign by
construction, so `required_isolation` reads the declaration and subprocesses
the plugin. And **that is why it is a declaration rather than the plugin
importing the parser itself** — a relative import of a declared helper is
invisible to the isolation decision (the entry file's AST shows a sibling, not
the C library behind it), which had installed plugins resolving IN_PROCESS
while PyMuPDF loaded into the kernel's own process. `_imports_foreign_code`
closes both halves: the declared modality, and the declared helper's own
imports.

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
`bundled/parsers/parse_text.py` is the migrated reference — it validates
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

**Parsers live in `parsers/`, a tree root of their own** — `bundled/parsers/
parse_text.py`, `DATA_DIR/installed/parsers/parse_pdf.py`. A parser belongs to
no plugin *family* (no base class, no entry point, nothing `plugin_discovery`
registers) but it is very much registered: `parsing.discover()` globs
`parse_*.py` and routes to it by extension and modality. See **The layout**
below for why that makes it a root and where the prefix rule comes from.

The **watcher** classifies a changed file by the root it sits in
(`PluginWatcher._root_of` → `trees.locate`): `parsers` rescans
`parsing.discover()`, `llm` rescans the backends, anything else falls through
to `plugin_info`. It used to sniff the *filename stem*, because everything that
was not one of the five families lived in a single `helpers/` folder and only
the prefix could tell a backend from a parser from an ordinary library — so a
plain library file came back as "not in a known plugin folder", which put a red
✕ in the user's chat for saving one. That whole classifier is gone; the
directory is the answer. Family-local helpers (`bundled/frontends/helpers/…`)
are still not watched at all: observers are scheduled non-recursively, so
editing one silently requires a restart.

The kernel keeps only the dependency-light `parse_text` parser (UTF-8 / code /
CSV / TSV, stdlib); the parser contract and shared text helpers live in
`sandbox/guest/parsing.py`, which every parser imports as `guest.parsing` so
the same file works in-process and in a subprocess. The registry carries a static native-modality default
map so `get_modality` resolves image/audio/video with **no parser installed**
(attachment routing relies on this). Every heavier parser is an installable
store package (`parser-pdf`, `parser-office`, `parser-tabular`,
`parser-image`, `parser-audio`, `parser-video`, `parser-gdoc`,
`parser-container`) shipping a `parsers/parse_*.py` file. `parsing.discover()`
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
serves any call. Its ceiling is `max_concurrent_subagents + 1`, derived rather
than chosen, because a subagent is the only thing that places a model call
concurrently with the foreground turn. It read `max_workers` back when a
subagent ran on an orchestrator worker; it runs on its own pool now, and the
two must be the *same* setting rather than two that happen to agree — a
fan-out wider than the pool serializes its calls behind one box lock and
presents as merely slow.

**A backend is `llm/llm_*.py`, and belongs to no family** — same as a
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
boundary. Text now goes out through `sdk.llm.delta` — token-scoped like
`llm.proceed`, and sent as a one-way `notice` on the wire (still classified
and recorded, just not awaited), because a reply per token would make streaming
from a subprocess slower than not streaming. The abort boolean is simply gone:
stopping is *cancellation*, which the kernel already owns. A native backend
still gets the old boolean, since it runs in-process and can be told.

**And giving up the answer gave up the unwind, which took a while to notice.**
The rule everywhere else is that a cancelled guest learns at its *next*
Request, which raises `Terminated`. A streaming loop has no next Request — the
only one it makes is `llm.delta`, and a notice is never answered, so
`channel.notify` can only raise on a *closed pipe*. So there was nothing to
refuse and nothing to raise, and the loop thread sat on the pipe until the
provider was done. A model that degenerated into repeating one token ran to
the provider's own repetition guard, unstoppably, while `/cancel` did nothing
anybody could see. The kernel ends such a call by ending the box
(`PersistentBox.interrupt`, and `Brain._interrupt` which must evict it from
`_boxes` as well as `_idle`, or the pool leases a corpse forever after). That
is why cancelling a model call costs a subprocess start — see **Cancelling
is immediate** below.

**Attachment routing moved kernel-side.** It used to be `BaseLLM.
_prepare_attachments`, which meant every backend inherited a method reaching
into `attachments.*` — a kernel import in the file that most needs isolating.
The loop now splits the bundle against the model's capabilities and the
backend's `native_modalities`, appends the text fallback itself, and the box
receives plain dicts whose bytes the backend reads with `sdk.fs.read_bytes`.

**Backend discovery is sandbox-only, and there is now no other kind.** Every
installed provider is a `llm/llm_*.py` backend running through a `Brain` pool;
plugin discovery never imports a provider into the kernel. `service_llm.py`
went first, then `NativeBrain`/`as_brain` — the ~180-line adapter that let an
object exposing the old `chat_with_tools` be driven as a brain. Its last
holdouts were test doubles, so `tests/support.FakeLLM` is Brain-shaped now
(`chat(request, on_delta=None) -> LLMResponse`), and `as_brain` went with it
rather than surviving as an identity function.

One consequence worth knowing when reading an older test: a double's recorded
`attachments` are the routed `{path, modality, file_name}` **dicts** an
`LLMRequest` carries. They used to be `Attachment` objects, because
`NativeBrain` rebuilt an `AttachmentBundle` for the old contract.
`describe()`'s `sandboxed` key survives as a literal `True` — it is part of
what the `llm.list` Request answers with, and a field that quietly disappears
is worse for a plugin than a true constant.

`DEFAULT_BACKEND = "LiteLLMService"` is **not** a leftover: the store's
backend calls itself `LiteLLMBackend` and claims the old name with
`replaces = ["LiteLLMService"]`, so stored configs keep resolving through
`backend_aliases`.

`/llm` gained explicit `load` / `unload` actions. Loading used to be a side
effect of editing a profile; now a brain holds real processes, so opening one
is something the user asks for. Only the default profile loads at boot.

**And it reaches the registry through `llm.list` / `llm.load` / `llm.unload`,
because nothing else could.** Deleting `service_llm.py` left `/llm` asking the
*service* registry about profiles — which had never held one and now never
would. Nothing raised: `ctx.services` simply had no key for any profile, so
every lookup answered `{}` and the command reported each model "not installed"
and "Unloaded" while conversations drove those same models perfectly well
through `usable_brain`. Two registries, one question, and the UI was reading
the wrong one; the tell was `load` reporting *"No backend is installed for
minimax/MiniMax-M3"*, which named a profile where the missing thing would have
been a backend. `llm.list` answers from `describe()`, which had existed since
the migration with exactly the right row shape and zero callers.

This is also where `display_name` finally gets read. Backends have declared it
all along and `backend_display_names()` had no consumers outside its own test,
so every picker in the app — `/llm`, `/setup` — offered raw class names, and
the profile card printed the *configured* string verbatim: "LiteLLMService",
the retired name, for a backend calling itself "LiteLLM (any provider)".
Displaying one takes two hops, and the missing one was the first:
`backend_aliases()` maps the stored name to the class that `replaces` it,
*then* the display map applies.

**When a capability is absorbed into the kernel, find the commands that
managed it.** The settings half of this is already documented above
(`llm_profiles` losing its owner); this is the same lesson one layer up, and it
failed the same silent way — a registry lookup that returns empty rather than
raising. `/packages update` had a third instance: it called `llm.refresh`
without `force`, which short-circuits on an unchanged *profile dict*, so
updating a backend's **source** left every open brain running the old code and
reported success.

**The store branch needs the matching migration**, five mechanical changes per
parser:

1. `services/helpers/parse_*.py` → `parsers/parse_*.py`
2. imports from `plugins.services.helpers.{ParseResult,parser_registry,
   parsing_utils}` → `from guest.parsing import ParseResult, clean_text,
   max_chars, register` (the **guest**, not `parsing` — a kernel import breaks
   the subprocess path)
3. signature `(path, config, services)` → `(sdk, path, config)`
4. `open(path)` → `sdk.fs.read(path)`; `services["x"].m()` →
   `sdk.services.call("x", "m")`; `logger.*` → `sdk.log`
5. drop `pathlib`; declare `dependencies_pip` for the heavy library (isolation
   is not declared — the kernel subprocesses it because it sees the import)

Run `sandbox.validator.validate_file` on each: `conforms.` means it will load
in a box. Nothing shims the old paths — the files have to move anyway.

## The kernel boundary (the one rule)

Core code (`pipeline/`, `runtime/`, `state_machine/`, `agent/`, `events/`,
`config/`, `attachments/`, `main.pyw`) hard-imports **zero** plugin modules.
Every capability, without exception, arrives by discovery.

The kernel may import the plugin *substrate*: base contracts, discovery and
path metadata, registry adapters, and `plugins.plugin_watcher`. The watcher is
part of discovery itself—it observes plugin trees and applies live registry
changes—rather than a discoverable capability.

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
guidance from each in-scope plugin's **`agent_prompt`** (see `_collect` in
`agent/system_prompt.py`), so missing plugins degrade silently and correctly —
uninstalling a package removes its prompt text with it.

**`agent_prompt` is one name with two shapes.** A plugin with nothing
conditional to say declares a plain string; one whose text depends on live
state defines `def agent_prompt(self, ctx)` (`sdk` in the guest). It was two
names — a static `agent_prompt` attribute plus an `agent_prompt_for(ctx)`
method whose base implementation returned it — duplicated byte-identically
across all five native base classes and the guest base, while the store grew
three spellings for one contract because neither name appeared in any template
or in `docs/SDK.md`.

The two cannot simply become a method, because the string form is not a
convenience: it is *read by AST at load* and copied onto the adapter, so a
static contribution costs no box call, while a dynamic one costs a real call
into the guest. So `_collect` accepts either, and that tolerance is
load-bearing rather than tidy: a string shadowing a method raises `TypeError`
into `_collect`'s `except`, and the guidance would vanish with **no symptom at
all**.

**The shape is the declaration, and it decides two things.** Nothing needs a
`dynamic_agent_prompt` flag: the bridge reads the AST (`_prompt_method`) and the
native bases declare `agent_prompt: str = ""`, so `callable()` is already an
exact answer. Writing a method *is* the statement that the text moves.

*Where it lands.* A string is settled at load, so it belongs in the semi-stable
block inside the cacheable position-0 message. A method exists because its
answer changes, so it goes in the dynamic `[SYSTEM CONTEXT UPDATE]` block with
the kernel's own live state — exactly the argument `_mode_suffix` already makes
for itself. Left in the prefix, every refresh would rewrite the one message
providers cache across a conversation, so fixing staleness would have cost a
cache miss on every later call of the turn. `_collect` takes a `live` flag and
`_in_scope` enumerates the populations once for both passes.

*How often it is recomputed.* `_collect` runs on every **LLM call**, not once
per turn, and for an ephemeral family every call into the guest is a fresh box —
so `_cached_prompt` caches. It used to cache *forever*, which made the method
shape a lie: a tool listing the scripts directory went on describing it as it
stood when the adapter was built, including for the file the agent had just
written. `sandbox/epoch.py` resolves both halves — one counter, bumped in
`Interpreter._settle` when a Request that *changed* something succeeds, and the
cache is stamped with it. A read-only stretch (read, search, think, call the
model again — most of a turn) costs zero recomputes; one `fs.write` costs
exactly one.

Two things about that counter are load-bearing. **`llm.delta` is excluded**: it
is a write and is *not* in `READ_ONLY`, but a streaming backend sends one per
token, so counting them would invalidate everything on every call and silently
undo the caching — the ledger's sandbox sink excludes it from recording for the
same shape of reason and draws the line the same way. And **refusals do not
count**, or `lockdown` would recompute every live prompt on every denial. It is
global rather than per-`Request.family` on purpose: nothing declares which
family a prompt method reads, so scoping would mean inferring the dependency,
and over-invalidating costs one box call while under-invalidating is silently
wrong. The lifetime resets in `residency.py` sit on top (`forget_prompt`, which
clears text and stamp together — clearing only the text leaves the stamp
matching): a residency's prompt is only knowable while its box is open, which
is a question about the box rather than about the world.

`_collect` and `bridge._prompt_method` answered to the old `agent_prompt_for`
for as long as any *loadable* plugin still wrote it. That is now nothing — the
kernel tree never did, and the one migrated store file that did
(`tool_sql_query`) was renamed in the same commit — so the fallback is gone.
It is pinned as a **negative** in `test_the_old_prompt_spelling_contributes_
nothing` rather than simply deleted, because the failure it guarded against is
silence in either direction: a plugin whose prompt never arrives looks
entirely healthy, so the rename has to be something the suite states out loud.
Unmigrated store plugins still spelling it (`service_location`) do not load
at all, so they cannot be the silent case.

## Hardening applied for kernel reliability

These edits exist so the kernel degrades cleanly when a stdlib plugin is absent —
the difference between a microkernel and a pile of assumptions:
- **`bundled/services/service_compactor.py`** — context compaction is a
  synchronous service call from the conversation loop, so the kernel does not
  route a blocking request through the event task queue. Its trigger lives in
  the loop's own **compaction layer** (`_compaction_layer`), a kernel escort
  always stacked inside registered `llm_call` escorts (onion: registered
  escorts → context guard → empty-response nudge → backend) — context safety
  is hook-shaped but never registry-dependent. Reactive overflow retries
  rebuild the prompt from compacted history and re-enter the inner onion, so
  they keep the post-escort brain, bus events, and provider params.
- **`runtime/runtime_config.py` `build_loop`** — the "no LLM" path now raises a
  friendly message pointing at `/setup` instead of an opaque error.
- **A capability absorbed into the kernel must take its settings with it.**
  `service_llm.py` declared `llm_profiles` and `default_llm_profile`; moving
  the code into `llm/` left the declarations behind, owned by nobody. That was
  not cosmetic — `config_manager.save` treated "already in plugin_config.json"
  as ownership, so an undeclared key's home was an accident of history, and any
  write that did not carry it sent it to config.json while every reader looked
  in plugin_config.json. Users lost their model configuration on plugin
  install. Both keys are kernel settings now; a kernel declaration always beats
  an existing plugin_config entry; and `config_manager.rehome_kernel_keys`
  moves stragglers at boot, new home first so a crash costs a duplicate rather
  than the value. Check this whenever something graduates into the kernel.
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
fact. Four origins:

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
- `sandbox` — every effect a plugin performs, written by the sink
  `runtime/ledger.sandbox_sink` builds and `main.pyw` hands to
  `Sandbox.bind_ledger`. Nothing wired that sink until recently, so the
  flight recorder was blind to exactly the code it most needed to watch.
  Rows carry the provenance chain in `data_json.chain`, which is the reason
  it is worth reading: `cron:nightly -> task_index -> service_web` says what
  a bare Request type cannot. **Reads are dropped** at the sink
  (`requests.READ_ONLY`) or a polling frontend would write twenty
  `console.read` rows a second forever; anything refused is kept regardless
  of type, since a denied read is a real event. `error_code` on a sandbox row
  is now `Result.code` (see **Error codes** below), falling back to `"failed"`
  for an uncoded failure — it used to be a two-value vocabulary
  reverse-engineered from whether the *message* started with "denied".

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
guidance belongs in a store package, not the kernel
prompt. Row well-formedness is pinned by `tests/test_ledger.py`.
Query/inspection UX (`/ledger`) is deliberately a
future store package, not kernel.

## Package store V1

- **Tree mirror, not package archives.** The `origin/store` branch mirrors what
  `DATA_DIR/installed` would look like if every optional plugin/helper
  were installed: `tools/tool_*.py`, `services/service_*.py`,
  `frontends/frontend_*.py`, `commands/command_*.py`, `tasks/task_*.py`, plus
  family-local `helpers/` files. `/packages install <stem>` and
  `/packages uninstall <stem>` target the file stem (`frontend_telegram`,
  `parse_pdf`, `bundle_essentials`, etc.).
- **Dependency metadata lives in code.** Plugin base classes expose
  `dependencies_files` and `dependencies_pip`; helpers use the same names as
  module-level literal lists. The package manager reads these fields with AST
  parsing, never by importing store files.
- **Install is a tree copy.** `/packages` reads the target file from
  `origin/store`, recursively follows `dependencies_files`, runs `pip install`
  for collected `dependencies_pip`, and copies the same relative paths into
  `DATA_DIR/installed`. The store copy always wins: a differing
  existing file is overwritten in place (no versioning yet — the store branch
  is assumed to hold the newest version); byte-identical files are skipped.
- **Uninstall follows the dependency edge *backwards*.** Removing a file takes
  everything installed that declares it in `dependencies_files`, transitively,
  and leaves what the target itself depended on alone
  (`_dependents_closure_from_installed`). It walked forwards for a long time,
  which is the relation the file states but the opposite of the one the
  question asks: uninstalling `tool_hybrid_search` took `tool_lexical_search`
  and `tool_semantic_search` — two tools that work perfectly well alone — and
  left `service_memory_retrieve`, which cannot run without it, installed,
  autoloaded and failing every turn. A dependency is a claim about what I
  need, never a claim of ownership, and nothing in the tree records which
  files were installed for their own sake, so the forward question is not
  answerable and is not attempted. A file the target needed stays on disk,
  visible in `/packages list`; that is the cheap failure, and removing
  something that still works is not. **Pip is the exception**, because a
  library is shared by whoever names it with no edge between them: the three
  trees are scanned and anything another file still declares is kept, kernel
  requirements always. Bundles are cloud-only
  manifests in `origin/store` that list store-relative files and feed the same
  resolver. Versioning is deferred.

**`on_install` / `on_uninstall` are what a manifest could not be.** A package
needs things *arranged* — a value contributed to a kernel setting, a folder
made, a table defined — and leaves things behind when it goes. Both were
deferred for a long time as "config cleanup, SQL table cleanup", and the reason
they stayed deferred is that a declaration cannot describe what an arbitrary
plugin did. A list of tables and settings is guesswork about somebody else's
code; the plugin's own code is not. So they are two optional methods on
`BasePlugin`, found by AST (`bridge.lifecycle_entries` — *defining* one counts,
inheriting the base no-op does not, or every installed file would cost a box)
and run by `package_manager._run_lifecycle` in an ordinary ephemeral box.

**The timing is the authorization, and nothing about policy changed.** Both run
inside `/packages`, so the chain is `user:command -> packages -> <plugin>`.
Depth 2, so it inherits neither `Chain.typed_command` nor the install's
`approved` grant — a `config.write` is UNSAFE. But the root is `user:command`,
so the chain is **attended** and the write raises a real dialog naming the
setting and the value, at the one moment somebody deliberately asked for this
package. `db.define` stays SAFE, `DROP TABLE` included, which is what makes
teardown free.

That is the whole of why the two earlier attempts at install-time seeding
failed, both silently. A service seeding from `start()` roots at
`service:<name>` — unattended, so the unsafe write is *refused rather than
asked*, with no dialog to notice. Moving it to a `turn_start` hook and
manufacturing a caller (reverted, `735638f`) made it work and asked at the
moment furthest from anything the user chose to do. Typing a message is consent
to a reply; installing a package is consent to setting that package up.

Three details are load-bearing. **`on_install` fires on a fresh install and on
an update whose bytes changed**, decided by the condition the copy loop already
evaluates to print "Already installed" — so the contract is *idempotent*, and
there is no marker file that can disagree with the tree. **`on_uninstall` runs
first**, before the config-list edits, the unlink and the pip removal, because
a hook cannot load from a deleted file or import a library that has been
uninstalled; and only for `plan.remove_files`, since a kept dependency is one
somebody else still needs. **Neither can veto its operation**: a package whose
setup was declined is still installed, a package whose cleanup failed is still
removed. The run is named with the plugin's *declared* `name` rather than the
file stem, because that link is what `policy._owns_setting` matches against the
setting registry — the same identity mismatch `PersistentBox._identity` fixes
one layer over.

## Verifying the kernel

Discovery/boot smoke (no frontend, no config writes):
```bash
python -c "\
from config import config_manager; from pipeline.database import Database; \
from pipeline.orchestrator import Orchestrator; from agent.tool_registry import ToolRegistry; \
from plugins.plugin_discovery import discover_services, discover_tasks, discover_tools; \
c=config_manager.load(); db=Database(c['db_path']); s=discover_services(c); \
o=Orchestrator(db,c,s); discover_tasks(o); t=ToolRegistry(db,c,s); t.orchestrator=o; \
discover_tools(t); print(sorted(s), sorted(o.tasks), sorted(t.tools))"
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

**A naming warning that no longer applies, kept because the fix is the point.**
Two unrelated things used to be called "sandbox": `DATA_DIR/sandbox_plugins/`,
the agent-authored code tree, and `sandbox/`, the security boundary — this
section. The tree is `DATA_DIR/workspace/` now, so `sandbox/` means exactly one
thing. If a variable, docstring or store plugin still says "sandbox" about a
*tree*, it is stale.

Plugins are arbitrary code on the other side of a boundary. The sandbox
mediates it: sandboxed code **cannot act, it can only ask**. Anything touching
disk, network, clock, or process is a typed **Request** the kernel classifies,
executes, and answers. The threat model is *carelessness, not malice* — an
agent with good intentions and no judgement — which is why the validator is a
linter rather than a proof, and why the subprocess (not the linter) is the
actual boundary. Design rationale: `The Second Brain Security Contract` +
`docs/SECURITY_CONTRACT_APPENDIX.md`.

**Two halves, and the one rule.** `sandbox/guest/` runs *inside*: the Request
vocabulary, the wire protocol, the SDK, the plugin base classes, the child
entry point. Stdlib-only and self-contained — it is the shippable unit a
container image would copy. Everything else is *host*: `policy` (the single
`classify()` that decides safe/unsafe), `handlers` (the only code that touches
the world), `interpreter` (serial gate, parallel execution), the two runners,
`boxes`, `facade`, `bridge`, `validator`. **The guest never
imports the host** — pinned by `tests/test_sandbox_guest_boundary.py`, the
sandbox's counterpart to the kernel boundary test.

`sandbox/__init__.py` aliases the guest package under the bare name `guest` in
`sys.modules` (every submodule, derived from the directory). Plugin source
therefore says `from guest.bases import BaseTool` and resolves identically
in-process and in a subprocess, where the child runs `python -m guest.child`
with `sandbox/` as its cwd.

**Boxes.** A box is one execution context: one process, one memory space, one
lifetime. Files in the same box import each other; files in different boxes
cannot reach each other at all — the only way across is a Request. Services and
frontends are persistent (loaded once, called into, serialized one call at a
time). Persistence is resolved *before* code runs, so nothing can drift into it
by refusing to finish — that is a hang, and it times out.

**Isolation is provenance, not declaration** (`sandbox/isolation.py`). It was
`isolation = "subprocess"`, an AST-read declaration — which made the code being
contained the authority on its own containment, so an agent authoring a plugin
could author its own escape by leaving a line out. The tree decides instead,
because a file cannot assert which tree it is in: `workspace/` is always
subprocessed, `bundled/` is always in-process, and `installed/` is
subprocessed exactly when the validator sees an import it cannot mediate —
computed from the AST (`report.unmediated`), never from `dependencies_pip`,
which would be the same bug one level down. Unknown paths fail closed. A file
still declaring `isolation` gets an advisory note and the value is dropped.
Box grouping cannot be used to escape it: isolation is resolved per file before
grouping, and tightest-wins only ever tightens. A user-facing override
(config allowlist) is planned and is a different thing — a person may decide
what the code may not.

**`report.unmediated` is the entry file's AST, and two things got past it**
(`_imports_foreign_code`). An *imported helper* reads as an ordinary sibling —
`from . import parse_pdf` shows nothing foreign — so an installed plugin whose
own source was pure stdlib resolved IN_PROCESS while PyMuPDF loaded into the
kernel's own process, precisely what the parser migration existed to prevent.
A *declared modality* is the same shape one level up: the kernel loads parser
files into the box and the plugin's source says nothing about them at all.
Both are **declarations**, which is why they can be answered before anything
runs — and the check can only ever tighten. Note this is not a file asserting
its own containment: it asks for a *capability* and pays for it in isolation,
whereas the retired `isolation = "subprocess"` let a file assert the
containment itself and leaving it out was the escape.

The helper half follows the **import**, not the declaration, and that
distinction is the whole of whether the rule is usable.
`dependencies_files` does two jobs: it tells the package manager what else to
install, and it puts a file on the box's import path. Only the second loads
code, and the loader is explicit that it is merely permission — *"Declaring is
what makes a file importable; the plugin still writes the import."* Reading the
declaration alone subprocesses about fifteen store plugins for a packaging
relationship (`tool_web_search` declares `service_web_search`, every email tool
declares `service_gmail`, `task_embed_text` declares `service_embed`) — none of
which ever imports the file, so none of which ever loads its torch. So
`_relative_imports` walks the AST for `from . import x` / `from .x import y`,
intersects that with what was declared, and follows the chain transitively;
an imported-but-unresolvable helper fails closed.

**That boundary is what buys free authorship.** The agent reads, writes, edits
and deletes anywhere under `workspace/` with no approval, because
everything there is contained before it runs. This is the LibOS invariant
rather than an exception to it: writing a file changes what the system can
*ask*, not what it may *affect* — the new plugin's Requests are classified like
anybody else's, and it inherits nothing from having been written without a
dialog. The grant stops at that tree.

**A deadline measures guest execution, not wall clock** (`sandbox/watchdog.py`).
A box that has made a Request is not stuck — it is waiting for something the
kernel itself started, and the kernel may take as long as that takes. Charging
it anyway killed every legitimate slow call at thirty seconds: an escort
placing a model call, a service inside `sdk.ui.ask`, anything reaching
`proc.run`. So `Execution.running_for` is elapsed time minus time blocked on
the kernel, and a box is overdue only on *that*.

The subtraction is of **blocked time**, not "time since the guest last did
something" — the two look equivalent and are not. A runaway spinning on
`while True: sdk.fs.list(".")` is never idle for more than a millisecond, so
the second reading would make it immortal, which is the exact case a deadline
exists for. Subtracting blocked time takes one ninety-second wait off in full
and a thousand short ones off to nearly nothing. A `HARD_CEILING` on wall clock
backs it up, since a runaway can otherwise hide inside long Requests. One
shared watchdog thread enforces this for every box, which also retired the
per-call `threading.Timer` — at a 50 ms frontend poll that was twenty timer
threads a second for the life of the process.

**Ending code is the kernel's decision**, escalating ask → starve → kill.
Starvation only reaches code that propagates failures; killing reaches the
rest, and in-process there is no kill. A cancelled execution raises
`Terminated` (a `BaseException`, so a bare `except Exception` cannot swallow
it) rather than returning a failure — denial and cancellation are different
things. **Starving a *resident* box ends it**: cancellation is per-execution
and a resident box has one `Execution` for its whole life, so there is no way
back — the starved worker is still alive, and reusing the box would put two
threads on one execution. `PersistentBox` therefore marks itself dead on a
call timeout, and a cancelled `Terminated` surfaces as a failure rather than
`ok` with no data. It used to surface as success, which is how a starved REPL
kept polling a dead box forever: `_drive_polls` read the success, reset its
failure count, and the terminal accepted keystrokes and did nothing.

**Cancelling is immediate, because a flag is not a signal.**
`session.cancel_event` is read *between* actions — the top of
`ConversationLoop.drive`'s iteration — and everything slow lives *inside* one.
So `/cancel` was only ever as immediate as the current model or tool call, and
for a streaming model it was not immediate at all. Meanwhile the turn kept
producing: the narration pushed alongside a tool call, the ✕ status, the final
reply, the tail message, and — worst — the subagent barrier, which ran on a
cancelled turn and could force a whole fresh re-drive that was not even
cancelled, since `_drive_agent_turn`'s `finally` clears the flag on the way
past.

The flag now has subscribers. `RuntimeSession.interruptible()` arms a stopper
for the duration of one blocking call and `interrupt()` fires them;
`ConversationRuntime._interrupt_work` is the **one** place that does it, so a
third kind of blocking call cannot reintroduce the freeze by forgetting to be
stopped — the same argument `sandbox/events.publish` and
`handlers/kernel._drive` make for theirs. Two stoppers, because the two things
a turn blocks on are answered differently: the model call is one named box the
pool leased (handed back through `Brain.chat(on_call=...)`, since a caller
cannot know *which* box until the lease happens), and tool calls are whatever
ephemeral runs the sandbox is already tracking in `Sandbox._runs` —
`interrupt_session` filters them with `policy.chain_session`, which is exact
rather than a guess because `bridge._root_for` already roots an agent-caused
call at its session key. **Resident boxes are deliberately out of scope**: a
cancel that took the transport down with it would be worse than the freeze.

Two properties are load-bearing and both are about *silence*. Arming is
**refused once the flag is set** (`_InterruptSlot.arm`), or a cancel landing
between the loop's last check and the next call parks a stopper nobody is left
to fire. And an interrupted turn reports as **cancelled rather than as an
error**: killing the box makes the call fail with `box '…' died during
'__chat__'`, which is the mechanism working, and rendering it puts an `Error:`
on screen one line after `Cancelled.` The turn also says nothing at all in its
own result — the `/cancel` action already answered, and the `new_messages`
branch would otherwise surface the last assistant content, which is agent
output arriving after the person stopped the agent. The interrupted tool still
gets a history row (`Interrupted by user`), because an assistant `tool_calls`
row with no matching tool row is an invalid transcript next turn; only the
status is dropped.

**Provenance.** Every Request carries a chain rooted in what *caused* the work
(`user`, `cron:nightly_index`, a subagent). The kernel owns it as its own call
stack, so plugins can neither read nor misstate it; it is what makes an
approval dialog answerable, and it doubles as the cycle detector. Approval
reuses the kernel's existing `vet_permission` doorway (enriched with
`origin="request"` plus the typed `request`/`chain`/`decision`), then
then a dialog whose options can keep the answer (`sandbox/options.py`);
unattended sessions refuse rather than block.

**The chain only became a stack once it could survive re-entry**
(`sandbox/provenance.py`). `Chain.push` was called at the outermost run and
nowhere else, so every chain was one link deep: a tool reached through
`tool.call` started a *fresh* chain rooted at whatever caused the outer call,
with no memory of its caller. Three things rested on that and none of them
worked — `MAX_DEPTH` and the cycle detector were unreachable, the dialog had
one link to show, and the SAFE classification of `tool.call`/`service.call`
rests explicitly on "the callee's Requests are classified with the caller
still in the chain", which was not happening.

The re-entry happens inside a *handler*, whose signature is `(ctx, args)` —
about a hundred of them, three of which care — so the caller is ambient rather
than a parameter: `Interpreter._execute` marks the thread for the duration of
the handler, and `bridge._forward` and `PersistentBox.call` adopt it. That
works because the whole nested call is synchronous on one thread, which is to
say the thread *is* the call stack. The `ContextVar` is set and reset with a
token in a `finally`, since a pool worker does not reset its context between
tasks and a leaked value would be believed by the next Request to land on it.

A callee therefore **spends its caller's grant** and never re-derives one from
its own `requests` declaration — that re-derivation was the widening
`Chain.push` exists to prevent. A command's own manifest is read only when the
command is the root of the call.

**A callee still keeps its own identity, and that is what it is *called* that
matters** (`PersistentBox._identity`). A box is named for its file, so
adopting a caller's chain pushed `service_timekeeper`; every registry that
reasons about plugin identity — services, settings, and therefore policy's
ownership exemption — knows `timekeeper`. So a resident service became a
stranger to its own bookkeeping the moment somebody else called it: the
timekeeper writing `scheduled_jobs` from its own poll was SAFE ("timekeeper
persists its own scheduled_jobs"), and the identical write reached through
`agent.schedule` was UNSAFE. Approving one dialog therefore raised a second
one, mid tool call, for the callee's own persistence — and the session froze
around it. The pushed link is the registered name now: `target` when a shared
box says which occupant, the box's own root otherwise. Neither comes from the
guest, which is the same reason `policy._callers` trusts a resident root.

**A handler answers from a context, and resident boxes had none.** `ctx` is the
host-side `SecondBrainContext` — it never crosses into the guest; it is what
*answers*. Tools and commands are handed one per call and frontends when their
box opens, but a service is loaded at boot with no session to build one from,
so every config, database and runtime Request a service made was answered from
nothing. `config.read` returned `None` for every key — indistinguishable from
unset — and `config.write` failed outright, which is how the timekeeper came to
hold jobs it could not persist. The composition root now installs a factory
(`runtime.context.kernel_context`, fed by `set_kernel_parts` as each piece is
built) via `Sandbox.bind_context`, and a box called *from* a session adopts
that caller's context so it reads the right user's rows.

**Asking never happens on the gate thread, and never under the session lock.**
Both are the same lesson learned twice, from one freeze. The gate is the single
ordering point for every Request in the process — *including* the ones the
frontend makes to draw the dialog (`console.write`) and to read the answer
(`console.read`) — so an approver called inline made its own question
unanswerable. Unsafe Requests now leave the gate for a small dedicated approval
pool (`Interpreter._ask_then_execute`), separate from the execution pool so a
dialog waiting on a human cannot occupy a worker running plugins. Symmetrically,
a command *body* runs outside `session.lock`: `handle_action` holds that lock
across `_dispatch`, so a command that asked mid-run waited for an answer that
could only arrive as a second action needing the same lock. `_CallableAction.
_run` wraps only the handler in `cs.unlocked()` (`RuntimeSession.unlocked`,
which unwinds and restores the RLock's full depth), exactly as the agent turn
has always run outside it. What keeps a second action out meanwhile is the busy
guard, not the lock — `calling_command`/`calling_tool` are in `BUSY_PHASES` —
and `pop_phase` restores `previous_phase` so answering a mid-run approval does
not drop the session back to the base phase while the call is still running.

**Commands should still ask up front.** The mid-run path works now, but the
*right* path for a command is the state machine's: declare `require_approval`
or the per-action `approval_actions`/`approval_action_prefixes`, and the grant
is stated and answered before the body runs — non-blocking, and it states the
whole scope instead of interrupting half-done work with one Request in
isolation. `/packages` declared `plugin.install` and no gate, which is what
made it take the mid-run path and freeze;
`tests/test_command_approval_declarations.py` now pins the invariant across
every command tree, deriving the consequential set from `policy.ALWAYS_UNSAFE`
rather than restating it. Note declarations are read by **AST**, so they must
be literals — `approval_actions = tuple(ACTIONS)` or `(_DELETE,)` reads as
nothing at all.

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
declaration) — a scope nobody is shown is not consent. Which commands *must*
declare a gate is `policy.CONSEQUENTIAL`, read by
`tests/test_command_approval_declarations.py`. It lives in policy because only
policy can keep it true: the test used to assemble it from `ALWAYS_UNSAFE`
plus three hand-listed branches, so `task.reset` — unsafe for every argument
but *spelled* as a branch — was invisible to a set-membership derivation, and
`/tasks` shipped with no gate at all.

**And a `Decision` carries two strings, because `reason` had three readers.**
The ledger wants it stable and greppable, `interpreter._settle` hands it to a
*model* as the refusal, and the dialog showed it to a *person* — who got the
worst of the three. Worse, the dialog's action line is built from the same
arguments by different code, so it said everything twice: "Run shell commands:
`git pull`" over "run shell command: git pull (in Z:\…)". `reason` keeps the
first two readers; `say` is the human half, and it is deliberately empty on
most branches, because most have nothing to add beyond the arguments the
action line already renders. The title is the phrase and the body is only
what the phrase cannot carry — the arguments, then *who asked*
(`approval.describe_asker`), then `say` when there is one. Both frontends
print the title above the body, so a body repeating it is the same bug in a
new place; there is no constant title any more either.

`describe_asker` names the **leaf** and stops, and the reason is worth
knowing before adding to it: *no root that reaches a dialog is worth naming*.
Only an attended chain is asked about — step 3 refuses the rest — and the
attended roots are exactly `user`, `user:command`, and a session key. The
first two mean "you did this", which is what being asked already means; the
third names the session the dialog is *delivered to*, so printing it told
somebody reading Telegram that they were in Telegram, as a frontend-built
identifier (`telegram:7912761600:7912761600:0`) that put a ten-digit number
twice on screen. Everything a root could interestingly say — a cron schedule,
a background agent, a service acting on its own — belongs to work nobody is
watching, which is refused rather than asked. Those clauses were written and
deleted; `test_only_attended_roots_reach_a_dialog_so_no_root_is_worth_naming`
drives the approver to say so, since the fact is about the order of its steps
rather than about the renderer.

**Services are resident boxes**, and with frontends they are the half of the
bridge that lives in `sandbox/residency.py` — a residency is not a call, so
it is a different file rather than a branch. A sandboxed `BaseService` bridges
to a native one whose `_load()` opens a persistent box and whose `unload()`
closes it.
Methods named in `exports` become real attributes on the adapter, because
native callers reach a service by attribute access (`services.get("x").m()`),
not through `service.call`. The synthetic module supplies `build_services`,
since that is how discovery finds services. The box owns the start deadline —
`BaseService.load` used to wrap `_load` in a second wall-clock timer, and the
adapter had to set `load_timeout = 0` to stop the two racing; both are gone.

**Services are also the one family that may share a file, and they share a
box when they do.** Every other family is registered *as* its file — discovery
finds `tool_x.py` and expects the tool it is named after — so the validator's
one-class rule stands for four of the five. Services are reached through
`build_services`, which has always returned a *dict*, and the reason to put
two in one file is that they share something expensive: `service_embed.py`
holds a text and an image embedder, and two files would mean two torch
imports and two CUDA contexts for models that a serialized box could never
run simultaneously anyway. So `validator` collects `declarations["classes"]`,
one entry per class; `bridge._adapt_service` builds an adapter each over a
single `_Residency`; `open` carries `entries` and `call` carries a `target`.
Two details are load-bearing. The box closes on a **refcount**, not on the
first `unload` — the kernel loads and unloads each service by name with no
idea they are neighbours, and the naive mapping kills a live model with no
symptom beyond the survivor's calls failing. And the box takes the **maximum**
declared `timeout`/`memory_mb` across its occupants (`facade.inspect`), since
one shared ceiling has to fit the slowest one. A lone occupant sends no
`entries` and no `target`, so a one-service file's wire is byte-identical to
what it always was.

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

**A transport library that owns an event loop lives inside that inversion**,
which is the whole of the Telegram migration. python-telegram-bot is
asyncio-only and expects the process; the guest cannot give it a thread
(`threading` is an ERROR — the kernel schedules). What resolves it is that a
subprocess box serves *every* call on one thread (`_serve_persistent` is a
single read loop), so the loop is created in `start` and driven in slices:
`poll` does `run_until_complete(asyncio.sleep(0.08))`, which is the library's
turn, then drains a plain list its handlers appended to. `render` arrives
between polls with the loop idle, so a send is just awaited; anything
longer-lived (a streaming pump, a typing pulse) is a `create_task` that
progresses during later slices. Everything cross-thread — `run_in_executor`
around a blocking submit, `run_coroutine_threadsafe` bridges, locks in the
stream tracker — deleted rather than ported.

This shape **requires subprocess isolation** and cannot ask for it: an
in-process resident box runs each call on a *fresh worker thread*
(`boxes.PersistentBox._invoke`), where a loop bound in `start` belongs to
somebody else by the time `poll` returns. It holds structurally for Telegram
(installed tree + foreign import) rather than by declaration, which is the
right way round — but it is a real constraint on where such a frontend may
live, and `tests/test_store_frontend_contracts.py` pins that the reading comes
out right.

Two `BaseFrontend` hooks are **not** on the wire and a migrated frontend
therefore loses them: `render_queued_ack` (return True to replace the textual
mid-turn ack, e.g. with a message reaction) and `render_conversation_banner`
(mirror the conversation title on a persistent surface). Telegram used both and
gave them up. Carrying `queued_ack` means a render call whose *return value*
matters, which the one-way `_render` deliberately is not; `conversation_banner`
would fit the existing shape and is the cheaper of the two if either comes
back. Adding either means growing `KINDS` in `sandbox/frontends.py`, the
`native_names` map in `_adapt_frontend`, and the test that pins the set.

`BaseFrontend` itself is **not** migrated and should not be: its 880 lines are
host-side routing — fourteen bus subscriptions funnelling into nine `render_*`
methods, and `submit_*` funnelling into `runtime.handle_action`. The base owns
*when*, the guest owns *how*, so the base becomes the adapter. The nine
`render_*` collapse to one `render(kind, payload)` box call (`sandbox/
frontends.py` holds the `KINDS` both sides must agree on); `capabilities`
crosses as a literal dict and is rebuilt into `FrontendCapabilities`.

**Anything that drives the state machine leaves the caller's thread**
(`_drive` in `sandbox/handlers/kernel.py`). A resident frontend calls in from
`poll`, which holds its box's single call lock; `runtime.handle_action` runs
the turn *synchronously*, and a turn renders — straight back into the box that
is still waiting for the Request to answer. The render blocks on the lock and
the frontend is frozen for good. `submit` was detached for this reason;
`resolve` and `cancel` were not, and both reach `handle_action` by the same
path (`resolve_approval` and `cancel` are `submit` with a different action
type), so answering an approval from an inline button froze Telegram every
time. The REPL escaped by luck: in `approving_request` it answers through
`submit_text`, which was already detached. One helper now, so a sixth entry
point cannot be added without inheriting the answer. Detaching costs the
caller a real answer, so **existence is settled synchronously and only the
driving is handed to a thread** — `frontend.resolve` still returns False for
"there was nothing to answer", which is what a frontend branches on to decide
whether a line was a yes/no or an ordinary message.

The inbound half — `sdk.frontend.submit_text/submit_attachment/submit_action/
cancel/bind/attended/resolve` — is five Requests scoped the same way
`llm.proceed` is: **by reachability, not by a verdict.** The adapter is
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
guest's counterpart to the kernel's `render_plain` for monospace rendering —
same *output*, pinned by a test, but not the same code, and the difference
matters if anyone goes looking for duplication to delete. Nothing in
`sdk._Markdown` is a copy of `formatters.py`: measured line-for-line, the six
name-alike pairs (`table`/`md_table`, `card`/`detail_card`, `quote`/
`quote_block`, `plain`/`render_plain`, `truncate`/`truncate_cell`) share **0%**
of their bodies, and `align_tables`/`align_md_tables` shares 43%. The host side
delegates and the guest side is stdlib-only; they agree on results because a
test says so, which is the right coupling. Same finding holds for the hook
types — `runtime/hooks.py` and `guest/hooks.py` look duplicated by name, but
`HookContext` and `ModelRequest` differ in their *fields* (live `session`/
`runtime`/`attachments` against `session_key`/`user_id`/`conversation_id`).
They are projections, which is exactly why `sandbox/hooks.py` exists to
translate. Only the four `end_turn` verdicts are genuinely identical, and
`rebuild` — which coerces a wire dict defensively — survives sharing them, so
folding the two modules together would save about thirty lines and cost a
kernel-boundary widening. Not worth it.

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

**And outbound, a guest never publishes on the thread that delivers**
(`sandbox/events.publish`). `EventBus.emit` runs handlers on the caller's
thread, and when the caller is a guest that thread is holding its box's single
call lock — `poll` holds it for the tick's whole duration. So any subscriber
that calls back into a service blocks on a lock the publisher cannot release
until the emit is answered. That is not hypothetical: the timekeeper fired
`subagent.spawn` from its tick, `SubagentRegistry` answered by asking the
timekeeper to pin the job's conversation, and the box wedged until
`HARD_CEILING` killed it permanently. Worse, each later `cron.*` call parked
one of the sixteen sandbox workers on the dead lock *forever*, so the whole
process went deaf in about sixteen slash commands — a frozen REPL and a
frozen transport from one recurring job.

A guest can never observe a subscriber, so answering immediately and
delivering on a kernel-owned thread costs nothing anybody can see. One queue
and one thread, because thread-per-emit reorders a burst. This is the outbound
twin of `_drive`, and it is the *one place*, so a seventh subscriber cannot
reintroduce the bug by forgetting to detach.

Two backstops keep the same mistake from being fatal again, and both are the
same rule: **nothing waits forever on a box.** `PersistentBox._acquire` bounds
the lock wait by the ceiling the holder is already under and then fails
`ERROR_TIMEOUT` instead of parking a worker for the life of the process. And
`run_in_process` now measures *running* time through `watchdog.overdue`, the
same helper the resident path uses and whose docstring already claimed the two
could not drift — they had, so every ephemeral command was charged for time
the kernel itself spent answering it, and a slow-but-honest command died at
thirty seconds with the report blaming the plugin.

`sdk.llm.proceed` is the sole Request whose handler is a **per-call closure**
rather than a static table entry: an escort's `proceed` is parked host-side
under a one-shot token for exactly the duration of one doorway visit. Code
holding no token reaches no call, which is why the Request is refused outside
an `llm_call` hook rather than being ambient authority.

**The bridge is the only way in.** `plugins/plugin_discovery.py` →
`_load_plugin_module` calls `sandbox.bridge.adapt()`, which reads the file and
answers with a *native-looking adapter* subclassing the real
`BaseTool`/`BaseTask`/`BaseCommand`/`BaseService`/`BaseFrontend`; everything
downstream registers and calls it unchanged. Nothing is imported to decide
this — the AST pass that reads declarations is the same one that answers
whether the file can load. `_load_plugin_module` takes **only a path**: a box
is loaded by file rather than imported, so the module name, tree and reload
flag that five discovery loops used to pass said nothing the bridge could use.

**The native fallthrough is gone**, and with it the detection half that chose
between two loaders (`is_sandboxed`, `imports_sdk`, `SANDBOX_MODULES`). The
loader used to fall back to an ordinary import when `adapt` declined, so the
two contracts coexisted and the app worked at every point in the migration —
one file, one commit, `git checkout` to revert. That was the whole value of
it, and it expired with the migration: past that point coexistence is only a
way for unmediated code to keep running in the kernel's own process.

Deleting the detection pass **improved the error**, which is the tell that it
was the right thing to remove. A non-SDK file now reaches `validate_file` like
any other, and `plugins.Base*` is no longer in `CONTRACT_MODULES` — so instead
of a generic "did not load", the author is told which line imports a native
base class and given `from guest.bases import BaseTool` as the fix
(`validator.RETIRED_BASES`). Keeping `plugins.Base*` admissible there would
have been the real hazard: the file would pass the import check and the bridge
would build an adapter over a `run` that wants a context.

Refusal is **reported, never raised**. Every discovery loop reads `None` as
"skip this file" with no `try` around it, so raising would let one bad plugin
abort the discovery of every other one. The warning names the file and points
at the validator, because a plugin that vanishes silently is indistinguishable
from one that was never installed.

**`plugins/native/` is not the shim and did not go with it.** The five classes
are what the adapter *is* — `bridge.NATIVE_BASES` maps each family to one,
`_find_subclasses` uses them as discovery's predicate, `ToolResult` and
`TaskResult` are the kernel's own result types, and `frontend.py`'s 940 lines
are the host-side routing the guest half deliberately does not own (see
**Frontends are resident boxes the kernel *drives***). Deleting them deletes
the frontend layer, not a compatibility path.

They were `plugins/BaseTool.py` and friends, and the filename went on saying
"this is how you write a tool" long after that stopped being possible — so
they moved, and lost every member only a hand-written native plugin could
use: `BaseService.get_client`/`shared` (a live client is precisely what cannot
cross a boundary), `set_peer_services` and the whole `wire_peer_services`
chain (it handed a service a dict of native adapters no guest can see; peers
are reached with `sdk.services.call`), `BaseService.load`'s timeout thread
(the box owns the start deadline), and `BaseCommand.arg_completions`.

**Import the submodule, never the package.** `plugins/native/__init__.py`
re-exports nothing on purpose. The five differ enormously in what they drag
in — `tool` and `task` are standalone, `command` needs
`state_machine.conversation` for `FormStep`, `frontend` reaches the bus and
the runtime — so a convenience re-export is not merely wasteful, it is a
cycle: `agent.tool_registry` wants `BaseTool`, and pulling `command` alongside
it routes back through `state_machine` into `agent.tool_registry` before it
has finished defining `ToolRegistry`.

**One caller still imports plainly, and it is not a plugin.**
`parsing.discover` reaches `plugin_discovery.import_tree_module` to import
`parsers/parse_*.py` and fire their module-level `register(...)` calls. A
parser belongs to no family and nothing bridges it; it is guest code *by
construction*, and this is the kernel-side half of the dual callability
described under **Parsers** — the same file a box loads through
`guest.loader.install_parsers`.

**The kernel boundary is unchanged.** Core hard-imports no plugin module at
all. `sandbox/` is reached only from `plugin_discovery`, and nothing
in `sandbox/` imports `plugins.*` except the bridge (which needs the native
base classes to subclass).

**A Request may grow arguments; growing the vocabulary is the last resort.**
`fs.search` was a substring scan and `fs.list` a flat `Path.glob`, so `grep`
and `glob` had no way across — a guest-side search costs one round trip per
file, and the walking, pruning, regex and ripgrep they need are exactly the
unmediated reach the sandbox removes. The engine moved host-side
(`sandbox/walk.py`) and the two Requests grew *arguments*: pass none and the
original bare-list answer comes back byte-identical, pass any and the answer
is a dict with `truncated`/`scan_truncated`. Which types exist and what
`classify` says about each did not move. Same shape as the parsing and LLM
migrations — the boundary got narrower by adding kernel code. Ask "is the part
core actually needs standing knowledge?" before adding a type.

**And sometimes not even an argument** (`pipeline/sql_functions.py`). Semantic
search ranks a hundred thousand vectors and wants five rows, but cosine
similarity over a BLOB column is not expressible in SQLite — so the only way
to rank was to read every vector into the asking box: measured at 214 MB of
JSON across ~200 gate-serialized round trips, *per query*. `DB_MAX_ROWS` is
the statement that **the answer crosses, not the data**, and raising it would
have been fixing the symptom.

FTS5 is the precedent sitting in the same schema: `lexical_search` scans the
whole corpus and five rows cross, because the index is in the database and
`ORDER BY rank LIMIT ?` expresses the reduction. The vector case is the same
shape and was only missing the **operator**, so the kernel registers
`vec_cosine` as a scalar function on its connection and the plugin writes
ordinary `db.query` SQL. No Request, no SDK change, no policy change — and
the cap it was fighting simply stops applying, which is the tell that this was
the right level to fix it at.

What makes it kernel-general is that it is not a search: it knows nothing
about embeddings, models, streams or top-k, and it composes with `WHERE`,
`JOIN`, `ORDER BY` and `LIMIT` because it is an *expression*. A
`vector.search` Request would have needed a table, a column, a filter and a
limit — reimplementing SQL, badly, which is how you know the Request was the
wrong shape. numpy accelerates it when a package has installed one and is
never required; the two implementations are pinned equal by test. It returns
NULL rather than raising for anything it cannot compare, because a raising
scalar function fails the whole statement, which would turn one stale row into
a search that never works again.

Two types *were* added, both because they had no honest home. `plugin.validate`
runs the loader's own validator over a source file, so its verdict is the real
one, and it is read-only in the strongest sense — a pure AST walk that never
imports what it reads. It sits in `ALWAYS_SAFE` with the listings rather than
with `plugin.register`, since a dialog in the authoring loop would only teach an
agent to stop checking its work. `tool_test_plugin` is now a thin translation
layer over it (the store's, not the kernel's).

`app.stop(restart=bool)` is what let `/quit` and `/restart` stop being native:
ending the process is not reachable through any other Request, and the two
callables exist only in the composition root, so it travels as
`context.app_control` like every other host resource. One type with an argument
rather than two types — stopping and stopping-then-starting are the same act
with a different tail. `ALWAYS_UNSAFE` in both forms, since coming back up is
not a mitigation for everything in flight dying. The kernel owns the 0.75s
deferral, because a process that ends before its answer is delivered tells the
user nothing about why.

**The shell family is where a type earned its place** — `proc.start` /
`proc.status` / `proc.stop` / `proc.list` beside `proc.run`. Running a command
and *keeping* one are different acts: a live process outlives the Request that
made it, so there is a handle to hand back, poll and eventually kill, and no
return value expresses that. `proc.run`/`proc.start` grew a `shell` argument
instead of the guest wrapping its own command line, because `cmd.exe` does not
understand the escaping `subprocess` produces from a list — a guest passing
`["cmd", "/c", line]` loses every embedded quote, so building the invocation
is host work.

**And it is where the classifier died.** `tool_run_command` carried ~500 lines
deciding whether a command was read-only: decompose at unquoted `&&`/`||`/
`;`/`|`, match each segment against a whitelist, send redirection and
substitution to approval. It is undecidable in principle and it fails in the
invisible direction — a wrong "unsafe" gets reported, a wrong "safe" does not
— and it lived inside the plugin it authorized. So the whole family is
`UNSAFE` and every command is asked about; the migrated tool contains no
classifier and must not regrow one. Where it gets less onerous is
`sandbox/shell.py` — its own module, because working out *what commands a
line runs* stopped being a branch and became a subject: a lexer, a
decomposition and two recognizers, with `classify_shell` as the one entry
point `policy` calls. A recognizer returns a reason to allow a command or
`None` to abstain, so it can only ever widen and a bug costs a dialog. Two
ship. `_read_only_command` is the structural one, and it works only because it
refuses to be complete — the dead classifier tried to decide *every* command,
which is Rice's theorem, while deciding a few and abstaining on the rest is
trivial. Its unit is `(program, subcommand)`, because `git` is not read-only
and `git status` is; it abstains on any shell, any metacharacter, and any
program named by path. `_remembered_prefix` is the other, reading
`shell_allowed_prefixes` — what a person answered "always" to in an approval
dialog. Both ask about **coverage**, never safety — *is every segment already
granted* — which is decidable where safety never was, and both derive their
unit through `shell.command_prefix`, so a grant is stored and matched in one
vocabulary. A raw string prefix would be unsound, since `git push` also
prefixes `git push && rm -rf /`; the line is decomposed with a real lexer
(`shlex`, POSIX shells only, since `cmd` and PowerShell quote differently)
that knows the `&&` in `git commit -m "fix && ship"` is inside a quote — the
thing the dead classifier's regex got wrong. Redirects, substitution and
subshells are refused outright, because there the effect is not in any command
name: granting `echo` must not license `echo x > ~/.bashrc`.
`shell.render_command` is the one renderer the dialog and the ledger row
share, so what a person approves is what gets recorded. `status`/`stop`/`list` are `ALWAYS_SAFE`: they speak about
processes already approved at `start`, and stopping narrows — a dev server the
agent cannot kill without a dialog is one it will not start.

**Scripts are what the classifier's death left missing.** Every command being
asked about left the agent's *cheapest* capability also its most dangerous one,
with nothing safe to reach for instead — so under pressure everything routes
through the one door meant to be hardest. `script.run` is the alternative:
a file of SDK code under `<tree>/scripts/`, run in a subprocess, `SAFE`. It can
be safe precisely because a command line cannot — Python over the SDK has
nothing to interpret, so every effect inside arrives at the gate individually
with the script still in the chain. Same argument as `tool.call`.

The directory is the whole declaration (`isolation.is_script`), because a
script has no prefix, no base class and no entry point to say what it is. Two
things are then read off the destination, the same shape as the `fs.write`
branch: **where** the file is (anywhere but `scripts/` is refused, not asked),
and **what it imports** — a foreign library makes it `UNSAFE` and the dialog
names the library. That last rule is deliberately stricter than the plugin
equivalent: an installed package importing one is subprocessed and not asked
because somebody approved it at `plugin.install`, whereas a script was never
approved by anybody. Scripts are subprocessed in *every* tree, the one place
`required_isolation` skips the per-tree answer. The verdict is re-derived by
the kernel from the path and never supplied on the Request — a caller passing
its own report would be the contained code judging its own containment.

Ephemeral only, on purpose. The resident half already works
(`Sandbox.open` on a module, pinned by `test_a_bare_script_opens_as_a_resident_
server`) and is deliberately not exposed: a script that wants to stay resident
is a *service* and one that wants an hour is a *task*, and both are approved
where a commitment outliving a turn should be answered for. Note the 600s
`watchdog.HARD_CEILING` bounds ephemeral runs only.

**A script declares its deadline at module scope**, `timeout = 600`, exactly
as it declares `box` — `validator._collect_declarations` reads module-level
assignments, and `facade.inspect` falls back to them for a file with no plugin
class, so this worked from the day scripts existed and simply appeared in no
template, no doc and no test. That is the whole failure worth recording: the
capability was real, nobody could find it, and the natural conclusion from
reading `sdk.scripts.run` — which takes no `timeout` argument — was that 
scripts could not ask for one.

Two numbers bound a run and only one is declarable. The declared deadline is
**running** time (`Execution.running_for` discounts time blocked on the kernel)
and is clamped by `MAX_TIMEOUT_SECONDS`; `HARD_CEILING` is **wall** clock and
is not declarable at all. So ten minutes elapsed is the real ceiling on a
script however it spends them, which is the honest reason the "one that wants
an hour is a task" line above still holds: a subagent crawl that fans out
wider than `max_concurrent_subagents` waits in waves of
`subagent_timeout_seconds` (300 by default, 3600 by config) and can exceed the
wall ceiling without ever exceeding its deadline. Raising the declared timeout
does not help that case, and the ceiling is deliberately global.

`handlers/kernel._script_run` waits in slices rather than blocking, because
cancellation reaches code that is *making* Requests and this handler makes
none while it waits. `provenance.Caller` carries the calling `Execution` for
exactly that — a cancelled turn would otherwise leave the child running to its
ceiling on a held pool worker.

`paths.get` also gained `python` and `platform`. The validator refuses `sys`,
correctly — it is a door to the interpreter, not a fact about it — but which
Python is hosting the app (so `pip install` lands where the app can import
from) and which platform it is on are things a plugin needs and cannot
otherwise learn. Two constants the kernel already knows, closing the only
honest reason to want `sys`.

**The SDK idiom** — Requests return their value and raise on failure; a bare
return is wrapped:

```python
def run(self, sdk, path):
    return len(sdk.fs.read(path).split())     # no Result to unwrap
```

`sdk.Denied` (refused) subclasses `sdk.Failed` (anything). `sdk.ok(x,
llm_summary=...)` only when attaching extras. `sdk.log(...)`, never `logging`.

**A handler is an adapter, and does not catch for itself.**
`Interpreter._execute` already wraps every handler call: it logs
`logger.exception` and returns `Result.failure(f"handler error: {exc}")`. So a
per-handler `except Exception` around kernel code adds nothing and *costs the
traceback* — which is how a `NameError` in `_config_write` once presented as an
ordinary failed config write with the whole suite green.

The rule, in order: **guard foreign code** (a tool, service, command, parser,
model, `pip`, a subprocess, a frontend callback run inline) because "tool 'x'
failed" says something the net cannot; **guard multi-step I/O** where a partial
write needs naming; **otherwise let it raise**. Where the guest supplied the
input — SQL it wrote, a payload it built, a number it passed — catch the
*specific* exception and report `ERROR_INVALID_ARGUMENT`, because that is the
guest's mistake rather than a kernel bug. `fs_net.py` was always the house
style (one broad except in 41KB, otherwise `OSError`,
`subprocess.TimeoutExpired`, `urllib.error.*`); `kernel.py` has been brought to
it, 71 → 36.

Every kept guard also logs `logger.exception`. Attribution and a stack trace
are not alternatives — the message says *whose* bug it is, the traceback says
*where*, and 34 of the 36 used to keep only the first.

**Error codes** (`sandbox/guest/codes.py`). `Result.error` is a sentence, and a
sentence is for a person; `Result.code` is the `ERROR_*` token code branches on.
Which of the two `sdk` exceptions gets raised is `code in DENIAL_CODES` — it
used to be `error.startswith("denied")`, so a handler reporting *"denied by the
remote host"* made a plugin catch `sdk.Denied`, the kernel's own word for
policy, for a web server's refusal. The `"denied: "` prefix survives in the
message text and is now purely cosmetic; nothing reads it.

Two rules keep this a vocabulary rather than a rename of all 166 failure sites.
**An empty code is not a bug** — most failures are only ever read by a person,
and a code exists once a *second* reader needs to branch. And **`retryable`
stays orthogonal**: it is set by whoever knows, never derived from the code.

What carries a code today is the kernel's own failures (timeout, guest fault,
cancelled, shutting down, approval declined, not permitted) plus the two a
plugin actually branches on. `ERROR_NOT_FOUND` is deliberately *one* code
across files, directories, services, commands, tasks and conversations —
falling back when something is absent should not require knowing which
subsystem was asked, nor matching four sentences. `ERROR_UNAVAILABLE` is
separate and comes from `_need`, which guards ~64 sites in one function:
"this kernel has no database" is a different thing from "that conversation
does not exist", and a plugin retrying the second must not retry the first.

Adding a field to `Result` means editing `to_dict` *and* `from_dict`, which
enumerate fields by hand — forget the first and the field is lost only on the
subprocess hop, silent in-process. `tests/test_sandbox_error_contract.py`
derives its expectation from `dataclasses.fields`, so it fails the moment a
field is added without a value.

**Bytes cross, and the codec is not the database's.** JSON has no bytes type,
so a value that is merely *numeric* — an embedding vector, a thumbnail, a BLOB
column — had no way over the wire, and it failed in the worst available
direction: in-process there is no serialization at all, so a plugin writing a
BLOB worked on a thread and raised `TypeError` from inside `json.dumps` only
once the same file ran in a subprocess. `protocol.pack`/`unpack` encode bytes
as `{"__bytes__": "<b64>"}`, applied at the *four serialization boundaries* —
`Request.to_dict`/`from_dict`, `Result.to_dict`/`from_dict`, and the resident
box's `CALL` message either side. That covers `db.write` params, `db.query`
rows, `service.call` arguments and its return values in one place, and every
handler stays written as if bytes were ordinary, because from a handler's
side they are. Nothing about a schema changed: `embedding` is still a real
BLOB. Only a *lone* tag key is decoded, so plugin data containing the string
is not mistaken for an encoding. `fs.read_bytes` predates this and base64s by
hand at the SDK level; it is left alone, since its encoding is part of that
Request's documented answer. What this does **not** fix is volume —
`db.query` caps at 500 rows and a message at 16 MB, so a scan over every
vector in an index is still a bandwidth problem and still has to happen where
the vectors are.

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

**The mode is a standing answer to the dialog, not a new layer**
(`runtime/security_modes.py`, `/mode`). Three values — `lockdown` refuses
anything that would have raised a dialog, `ask` is the default, `yolo`
approves it — read at the one point in `build_approver` where the dialog would
otherwise be drawn. Its **position in that order is the whole of its scope**,
and both neighbours were chosen rather than fallen into. It sits *after*
attendance, so `yolo` never reaches work nobody is watching — a cron tick, a
service poll, a subagent — which is what stops a grant given for a foreground
task being spent by something the person cannot see. It sits *after* the hooks
and the own-secret exemption, so `lockdown` answers only what would have
reached the person: it does not countermand a plugin gate that positively
allowed something, and it does not stop a service reading the credential it was
configured with. Lockdown means "stop asking me, the answer is no", which is a
promise it can keep; "break the plugins I already set up" is not.

`yolo` also pre-answers the state machine's command grant, through
`ConversationState.auto_approve` — a predicate the driver installs beside
`unlocked`, so `state_machine/` stays ignorant of the mode vocabulary and only
ever asks whether it may skip asking. It routes through `_run(approved=True)`,
so the command gets the same `chain.approved` grant a typed yes produces
rather than running ungranted. `lockdown` deliberately stops short of that
layer: it is about a command *the person just typed*, and auto-refusing it
would mean "you may not use your own machine".

**Two properties are load-bearing and neither is a policy.** *Lockdown is not
a trap*: the mode is enforced at the approver, so the act that leaves it must
never arrive there — `session.set_mode` is SAFE for `chain.typed_command`, the
same exemption `config.write` uses, and without it `/mode ask` would be
auto-refused by the very thing it lifts and restarting the app would be the
only way out. *A mode cannot outlive its conversation*, and that is
**structural rather than maintained**: the session stores the mode and the
`conversation_id` it was set against, and the reader answers `ask` when they
disagree. The alternative was resetting at `/new`, `/clear`,
`load_conversation` and the three paths that null the id — a list to keep in
step, in the direction where forgetting one leaves `yolo` running in somebody
else's conversation. Nothing is persisted either, so a restart returns to
`ask`.

Who may change it is mechanisms 5 and 7 together: arriving at `lockdown`
narrows whatever we were in, so an agent may do it unasked; everything else
could widen, so it raises a dialog. `scope="turn"` sets a mode dropped at
`HookRegistry.finish_turn` — stacked there rather than registered as a
`turn_finish` hook, same argument as the compaction layer and the subagent
barrier, because a grant that expires only when some plugin happens to be
installed is not a grant that expires. That slot is what "Allow, and stop
asking for the rest of this turn" writes (`options._rest_of_this_turn`, the
first `OPTION_BUILDERS` entry whose unit is *time*, and the proof the opaque
`remember` closure was worth having — it writes to the session, and neither
`options_for` nor the dialog had to learn it was different).

**Plan mode is deliberately not built, and everything under it is.** It is a
fourth value of the same field plus a `propose_plan` tool: the refusing mode,
the turn-scoped yolo for the turn after approval, the Request that sets the
mode, the per-turn prompt line, and the clearing at turn end all ship here.
`docs/PERMISSIONS_MAP.md` §6a is the fuller writeup, and it corrects what that
file used to claim — that modes belong at `vet_permission`. They do not: a
hook comes from a service, a service is a store package, and a lockdown that
stops working when you uninstall something is worse than none. The mode is
kernel-owned and hook-*shaped*.

**Docs:** `docs/SDK.md` (hand this to an agent writing sandbox code — its examples
are executed by `tests/test_sdk_docs.py`), `docs/MIGRATING_PLUGINS.md` (the
per-plugin procedure), `docs/SECURITY_CONTRACT_APPENDIX.md` (the ~87-Request
catalogue with policy inputs).

**Injecting prompt text is owned by session, not refused outright.**
`session.add_prompt_extra` was `ALWAYS_UNSAFE`, which sounds right and was
not — the capability it guarded is already free. Any loaded plugin puts
arbitrary text into every prompt by declaring `agent_prompt`, and nobody is
asked. So refusing the same text through the Request bought no safety; it only
made the *targeted, removable, per-session* spelling the expensive one, which
is the spelling a hook wants. It is a branch now (mechanism 3, ownership,
aimed at a session): SAFE for the caller's own session, UNSAFE when the `key`
argument names another — which may belong to another user, and is the one
thing `agent_prompt` cannot do.

The handler underneath had never worked. It called
`add_system_prompt_extra(key, text)` against a three-argument method, so every
sandboxed injection raised `TypeError` and came back as an ordinary failed
Request — silent by nature, since guidance that never arrives looks exactly
like a plugin with nothing to say. The Request carries two keys now: `key` is
the *session*, `slot` is the named overlay within it, and the slot defaults to
the calling plugin off the chain, because overlays are a dict and a constant
default would have the second plugin quietly overwrite the first. The slot
comes back as the handle `remove_prompt` takes. Note the overlay is persisted
into the state marker, so a slot outlives a restart until its writer refreshes
it; a stale line of guidance is not a permission, and the next `turn_start`
overwrites it.

**Migration tooling:** `sandbox.validator.validate_file(path).render()` is the
whole of it — it names every line that needs converting and the Request each
effect becomes, and `conforms.` means the file will load in a box. There were
once two more tools (`sandbox.migrate.plan`, a checklist the validator already
prints; `sandbox.parity.compare`, which diffed a migrated plugin's return value
against `git show HEAD:`). Both were deleted: nothing but their own tests ever
called them, and parity could compare only return values, never effects, so it
could not answer the question a migration actually raises. Templates in `templates/` are
migrated and `tests/test_templates.py` keeps them that way — it validates all
nine and fails on any native-contract vocabulary, including in prose.

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
A tool used to be able to declare `background_safe = False` and be refused
there wholesale; that gate is gone, because an unattended chain already
refuses every unsafe Request and the declaration was the contained code
describing its own containment. **Subagents are the kernel capability built
on these
primitives** (`runtime/subagents.py`): a `SubagentRegistry` opens a
`spawn_subagent:<cid>` session, drives `runtime.iterate_agent_turn(...)` on
its own pool, and hands back a **handle**. There is no `is_subagent` flag in
the runtime — a subagent is just a session whose key isn't the active one, and
that is exactly what makes it safe: its Requests build a chain rooted at that
key rather than at `user`, so `Chain.attended` is false and nothing unsafe can
be approved on its behalf.

It got here the way parsing and the LLM did, except there was no
implementation half to demote — spawning is *all* routing. It was a store task
plus a store service reaching into `session.pending_user_messages`,
`session.cancel_event`, and a `task_runs` row located by
`payload_json LIKE '%"conversation_id": N,%'`. None of that survives a
sandbox boundary.

**A handle is what the vocabulary grew for.** `agent.spawn` gained
`agent.collect` and `agent.stop` beside it, the same argument `proc.start`
makes: a background child outlives the Request that made it. `wait=True` is
still one Request answering with one report; `wait=False` returns a handle so a
fan-out is expressible from code that has no turn to hold open — which is every
script, and the reason spawning is in the SDK at all.

**One delivery, decided by whoever collects first.** A finished child's report
sits on its handle until an explicit `collect` takes it, or until the kernel's
end-of-turn **barrier** takes it for children nobody collected. The barrier is
*stacked* in `ConversationLoop._subagent_barrier` rather than registered as an
`end_turn` hook — the same argument the compaction layer makes one moment over:
collecting children must not depend on which plugins are installed. It stands
ahead of the doormen *and* ahead of `DOORMAN_FIRE_LIMIT`, because a turn that
spent its doorman budget must still not abandon agents it started.

**And it is asked twice, on purpose.** The doorways are the *right* place —
they hold agent priority for the whole wait, so no user message lands between
the halves of one logical turn. But they are not the *only* way out of a
drive: a failed action, a priority handoff, an exhausted iteration budget all
leave without reaching one, and on those paths the children were abandoned
with their reports produced and never delivered. That failure is silent, which
is the worst kind — the agent simply reports nothing and, in practice, spawns
the same work again. So `_drive_agent_turn` asks again after `loop.drive()`
returns: one line every drive passes through, whatever route it took.
`barrier()` settles what it collects, so the second ask on a normal turn finds
nothing. The re-drive loop already restores agent priority for a restart set
after `end_turn`, which is what makes the late ask work.

**Depth is lineage on the handle, not chain depth.** A subagent's turn is not
a nested sandbox call — its Requests build a *fresh* `Chain`, so `Chain.depth`
is back to zero for every generation and cannot bound recursion. Each `Handle`
carries `depth` and `parent`, `max_subagent_depth` (default 1) bounds it, and
a spawner whose handle can no longer be identified is read as depth 1 rather
than 0 — being forgotten must not buy unlimited nesting. Cancellation walks
*down* that lineage: `cancel` takes a child's descendants with it, `cancel_for`
is what `/cancel` reaches for, and `cancel_all` is the kill switch shutdown
uses. Stopping an agent while its children keep spending money on a question
nobody is waiting for is the worst of both.

**A child may be given less than its spawner has** — `spawn(profile=...)`, an
agent profile *name* the kernel resolves, the same handle-not-the-thing move
`ModelRequest.llm` makes. Everything under it already existed: `agent_scope.
load_scope` turns an `agent_profiles` entry into an LLM, a prompt suffix and a
tool whitelist, `scoped_registry` applies it (closing a whitelist over
`dependencies_tools`, so a scoped-in tool keeps its helpers callable), and
`runtime_config.profile_for` reads `session.profile_override` first. The only
missing piece was that `_conversation_for` wrote `"default"` into the child's
state marker as a literal. It writes the resolved name now, and `_run` also
calls `set_agent_profile` after `open_session` — the marker alone is not
enough, because a *reused* conversation (a scheduled job pins one) already has
a marker, and because `set_agent_profile` is what rebuilds the tool specs the
turn actually calls through.

Two properties are the point rather than the mechanism. **Naming nothing
inherits the spawner's profile**, not `default`: the literal meant a session
pinned to a narrow profile spawned an unrestricted child, which is a widening
nobody asked for. And **an unknown name raises**, because quietly substituting
`default` would run a background agent with every installed tool while the
caller believed it was confined, and nothing anywhere would say so. Choosing
among profiles needs no classification of its own — they are the user's own
config, naming tools the user installed, so the choice can only narrow. The
store's memory curator is the worked example: a `memory_curator` profile
whitelisting four tools and not `edit_file` is what makes it safe to let an
unattended subagent write.

**A deadline measures running, not waiting.** The clock starts when a pool
worker picks a child up, not when it was submitted: a fan-out wider than
`max_concurrent_subagents` queues, and charging a child for the queue cancels
the tail of a large fan-out before it has said a word. Same distinction the
sandbox watchdog draws between elapsed and blocked time.

**A subagent renders to no frontend.** Its session belongs to nobody, and
`BaseFrontend._live_session_keys` used to return *every* session the runtime
knew about — so a child spawned from Telegram typed its tool calls into the
REPL in front of whoever was sitting there. The filter is ownership, not the
subagent prefix as such: a session no person is looking at has no frontend to
render to.

Two mechanisms died with the migration and should not come back: the
session-side set of "children the parent already cancelled" (needed only to
suppress a stale completion echo from a run with no state of its own — a handle
has state), and the `subagent_llm` escort, which registered the `model_call`
moment that no longer exists and was therefore already dead code.

Deadlines are hard cutoffs: a child still running at `subagent_timeout_seconds`
is cancelled and reported as failed, never silently dropped, because "no news"
is indistinguishable from "still thinking" and an agent that guesses reports
findings nobody produced. Concurrency is `max_concurrent_subagents`, which is
also what `llm/registry._pool_ceiling` derives from — see **The LLM**.

`/schedule` is a kernel command (it manages *any* Timekeeper job, and the
Timekeeper is a kernel service). The store keeps only the two agent-facing
tools, `tool_spawn_subagent` and `tool_schedule_subagent`, both thin over
`sdk.agent`. Scheduled spawns arrive on the kernel-owned `SUBAGENT_SPAWN`
channel, which the registry subscribes to itself.

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
pointer. `active_agent_profile` is user-scoped too: profile definitions
remain global, but the user's selected profile lives with that user. **Conversation ownership is enforced** by `runtime.assert_conversation_access`
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

`BaseFrontend` ([plugins/native/frontend.py](plugins/native/frontend.py)) subscribes
both events and routes them through `render_tool_status(session_key,
payload)`. Rich frontends such as installed Telegram can edit a single status
message in place; the REPL prints the same shapes to stdout.

## Presentation convention: markdown on the wire

Command/tool output is a **string of GitHub-flavored markdown**, built with
the primitives in
[bundled/frontends/helpers/formatters.py](bundled/frontends/helpers/formatters.py):
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

- **Add a slash command**: write a `BaseCommand` subclass from `guest.bases`
  as `command_*.py` in the workspace, installed package tree, or deliberately
  in [bundled/commands/](bundled/commands/) when it is true kernel behavior.
  Commands receive `sdk` in both `form(args, sdk)` and `run(args, sdk)`.
- **Add a tool**: write a `BaseTool` subclass from `guest.bases` as
  `tool_*.py`. It receives `sdk`, not `context`, and the bridge registers it
  like any other tool. See `docs/SDK.md`. There is no second way — a file
  written against `plugins.BaseTool` no longer loads at all.
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
  unattended sessions, where the kernel's default on abstain is to refuse), `llm_call` (**escort** —
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
- **Observe a conversation finishing**: subscribe to
  `SESSION_CONVERSATION_ENDED`, emitted when a session lets go of a
  conversation — switched away from (`/new`, `/clear`, loading another),
  session closed, or deleted out from under it. It is the counterpart to
  `SESSION_CONVERSATION_CHANGED`, which names the conversation being switched
  *to*: right for a frontend redrawing "where am I?", exactly backwards for
  anything treating a conversation as a **unit of work**. Reflection,
  summarization and memory extraction all want the id being left behind, and
  before this channel existed there was no way to learn it — the payload of
  CHANGED simply does not carry it, so a consumer had to keep its own
  previous-conversation state and rebuild it after every restart. A crash
  emits nothing, so anything that must not lose work still needs its own
  idempotent record of what it has handled; this makes reflection *prompt*,
  not exactly-once.
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
  the active one, so their Requests build an unattended chain and nothing
  unsafe can be approved on their behalf.
- **Let an agent run a slash command**: use an installed command/tool bridge if
  one is present in the current tool catalog. The kernel should not hardcode
  command-running tools for packages it may not ship.

## Command plugins

Slash commands now mirror the rest of the plugin system. The repo starts with a
clean command slate: add built-ins as `command_*.py` files under
[bundled/commands/](bundled/commands/), or create workspace commands under
`DATA_DIR/workspace/commands`. The registry in
[plugins/command_registry.py](plugins/command_registry.py)
is only the adapter: it builds context-aware forms, parses one-shot `/cmd ...`
input mechanically, and dispatches structured dict args.

## Sandbox plugin system (the *plugin tree*, not `sandbox/`)

Unrelated to the security sandbox above — this is where agent-authored plugins
live on disk.

The agent can author tools/tasks/services/commands/frontends into
`DATA_DIR/workspace/<family>/` when an editing/package-authoring tool is
installed and in scope. Shell and file-editing tools are not kernel guarantees.
Sandbox and installed plugins are auto-discovered alongside first-party ones in
[plugins/](plugins/). Plugin helpers should use relative imports so files can
move between built-in, sandbox, and installed trees.

## Files that matter most

- [trees.py](trees.py) — the layout authority: which trees exist, which roots
  each holds, and the prefix rule. Everything that walks the layout —
  discovery, the watcher, parsing, llm, isolation, policy, the package
  manager — reads this one table.
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
- [pipeline/sql_functions.py](pipeline/sql_functions.py) — scalar functions
  every query gets, plugin queries included. `vec_cosine` is why a sandboxed
  semantic search can rank a corpus it is never allowed to hold.
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
  authorization surface for sandboxed code, plus `Chain` (provenance). The
  eight mechanisms every argument-conditional branch is built from are
  catalogued in a comment above `classify`.
- [sandbox/shell.py](sandbox/shell.py) — the one family whose *question* is
  hard: what commands does this line actually run. Lexer, decomposition, the
  read-only and remembered recognizers, `render_command`.
- [sandbox/options.py](sandbox/options.py) — what a person may answer an
  approval dialog with. `OPTION_BUILDERS` is the seam for a new kind of
  answer; `remember` is the only sandbox code that writes config.
- [sandbox/interpreter.py](sandbox/interpreter.py) — the drive loop the whole
  sandbox hangs off: serial gate, parallel execution.
- [sandbox/facade.py](sandbox/facade.py) — `Sandbox`: the one API.
  `run()` blocks, `start()` returns a `Run` to wait on or cancel, `open()`
  loads a resident box.
- [sandbox/bridge.py](sandbox/bridge.py) — the only doorway a plugin enters
  by: SDK code in, a native-looking adapter out. Tools, tasks and commands
  in full; services and frontends hand off to `residency.py`.
- [sandbox/residency.py](sandbox/residency.py) — the other half, for the two
  families that hold a process rather than answer a call: the refcounted box
  two services share, declared hooks and bus subscriptions, the poll loop,
  and the frontend's inverted start/render loop.
- [sandbox/guest/sdk.py](sandbox/guest/sdk.py) — what plugin authors actually
  type. Each namespace is exactly one Request family.
- [sandbox/hooks.py](sandbox/hooks.py) — the two-way translation between the
  kernel's doorways (live objects) and the guest's (plain data), plus the
  escort's parked-closure mechanism.
