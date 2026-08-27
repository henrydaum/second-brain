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
  to the SDK) and `frontend_http` live on the store branch. There was a third,
  an MCP server exposing Second Brain to external MCP clients over streamable
  HTTP; it was never migrated — it still imported `logging`,
  `pipeline.database` and `state_machine.conversation_phases`, so the bridge
  could not carry it and the app could not load it at all — and it was
  **deleted** rather than ported, along with the `mcp` service and command that
  went with it. Reviving MCP means writing it against the SDK, not restoring a
  file. Testing the two that remain
  is split by whose behaviour is under test. What the *kernel* claims about
  them — the validator's verdict, the declarations the bridge reads, the
  isolation the tree resolves — is `tests/test_store_frontend_contracts.py`
  (plus `tests/test_frontend_http.py`, which has its own file because that
  plugin is the whole of what a web or native app can reach)
  and runs by default. Their own behaviour (markdown rendering, chunking, the
  streamed-reply tracker, media planning) is marked
  `store` in `pytest.ini` and deselected, since a kernel change cannot break
  it; run it with `pytest -m store`. They reach the store branch through
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
  wizard), `llm`, `conversations`, `clear`, `cancel`, `compact`, `debug`,
  `frontends`, `locations`, `commands`, `tools`, `services`, `tasks`,
  `packages`, `permissions`, `mode`, `schedule`, `quit`, `restart`.
  `quit` and `restart` used to be
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
  `compact` is kernel on that same argument: it triggers the loop's own
  context-safety mechanism, and a lever for the one thing that keeps a long
  conversation alive must not disappear with a package.
  Profile/update commands are package capabilities unless the
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

**And it is the one parser that declares `generic=True`**, which answers a
question `modality` cannot: *is this file's text its own content?* `parse_text`
registers `.py` as text and the store's `parse_gdoc` registers `.gdoc` as
text — one word for a source file and for a 150-byte JSON stub naming a
document that lives in Drive. Anything routing on the modality alone reads the
stub and reports it as the file.

`tool_read_file` did exactly that, and the shape of the failure is the reason
this is a declaration rather than a heuristic. Its rule was *"hand it to a
parser when the bytes do not decode as text"* — deliberately not an extension
list, which would drift from what is installed. That is right for a PDF and
inverted for a pointer: `parse_gdoc` opens with `json.loads(sdk.fs.read(path))`,
so being textual is a *precondition of the parser working*, and being textual
was taken as proof no parser was needed. The parser was unreachable from that
tool by construction, and the agent got a successful-looking read of a JSON
file rather than an error.

Three things about the flag. **`False` is the default and the safe end**, so a
parser author who never hears of it ships a specialist and their format is
routed to them; the failure it prevents is silent, so the default has to be
the one that fails loudly. **It rides on `register`** — the parser is the only
thing that knows whether it decodes bytes or fetches content, the same
argument `parse_modalities` makes one paragraph up. And **the kernel exposes
it, never the extension list**: `parsing.describe_extension` answers
`{modality, known, generic}` and `sdk.parse.modality(ext, detail=True)` is the
Request behind it — an argument on the existing type, because the subject is
the same question. `known` is separate because `get_modality` answers the
*string* `"unknown"`, which a caller comparing against real modalities gets
right only by accident.

The rule a caller wants is then three lines, and the order matters: a
specialist parses **before** any bytes are read, generic and unregistered
extensions stay on `fs.read`. That last part is not tidiness — `parse_text`
applies `clean_text` and a char cap, and `edit_file`'s exact-replacement gate
needs what is byte-for-byte on disk, so routing plain text through the parser
is the obvious fix and a wrong one.

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

**How hard the model thinks is a profile setting, and the kernel names
exactly one param.** `Brain.params` builds the extra provider kwargs a profile
sends on every call. They live in one optional key inside the free-form
`llm_profiles` dict — `llm_extra_params`, forwarded verbatim — with reasoning
effort as an ordinary member under its OpenAI-compatible wire name
`reasoning_effort`, not a field of its own. `/llm` offers it a picker the way
it offers a capability a checkbox: the same relationship
`llm_capability_image` has with `llm_capabilities`, because the value belongs
in the dict beside its neighbours and a picker beats remembering a vocabulary
and JSON syntax to set one member of it. No schema change and no migration —
an old profile simply has no such key.

That is the whole of what keeps it **backend-agnostic**: one standard name, an
opaque dict, and no declaration for a backend to get wrong. A backend that
cannot carry a param degrades it (LiteLLM sets `drop_params`) or reports the
provider's own refusal, which is a sentence the person who typed the value can
act on — and the alternative, a `supported_params` declaration the kernel
filtered against, would make a backend written before this existed silently
drop a setting the user could see in `/llm`. Nothing in `llm/`, `runtime/` or
`/llm` learns a provider's name.

**A profile that says nothing gets `DEFAULT_REASONING_EFFORT`, and a `null`
means send nothing.** Absent is not a decision, so the kernel makes one:
"whatever the provider felt like" differs per model, which left one profile
silently thinking hard and its neighbour not at all — the comparison `/llm`
exists to make readable. That default is what forces the second half: without
a way to decline, `reasoning_effort` would be the one param a profile could
not refuse. It is written as JSON `null` rather than a magic word so it cannot
be confused with `none`, a real level several providers accept meaning "think
as little as possible", and the rule is stated generally — *any* null-valued
extra is dropped — so the next param the kernel defaults needs no second
mechanism. Note the order inside `params`: the default is applied *before* the
nulls are dropped, or clearing the effort would hand it straight back.

Two concessions to the fact that config gets hand-edited. `"off"` is accepted
as an alias for that null, because it is the word `/llm`'s picker *shows* and
therefore the word people type into the file — free to alias, since `off` is a
level at no provider. And an `llm_extra_params` that is not an object is
ignored with a **logged warning**, once per profile version rather than per
call: staying reachable is right, staying silent about it is how somebody
spends an afternoon on a setting they can see in a file and that does nothing.
`/llm` also refuses `api_key`, `api_base`, `model`, `messages`, `tools` and
`stream` as extras and names the field that really sets each — the backend
merges with `setdefault`, so one of those would win over the profile silently,
and a credential there sits in plaintext instead of behind the `secret_`
prefix that declares it one.

**There is one provider this default can break, and it is a known cost.**
LiteLLM translates effort into thinking *token budgets* for Anthropic, and a
Claude turn with thinking on must hand its cryptographically signed
`thinking_blocks` back on the next tool-result turn or the API refuses the
call — *"Expected `thinking` or `redacted_thinking`, but found `tool_use`"*.
LiteLLM does its half in both directions (`anthropic_messages_pt` reads
`thinking_blocks` straight off the assistant message it is given); what breaks
the chain is us. `LLMResponse` has no field for one and `_for_provider`
rebuilds assistant messages from `{role, content, tool_calls}`, so the blocks
are discarded and cannot be handed back. A translation layer can normalize a
*shape*; it cannot normalize *statefulness*. Carrying them would mean the same
five threading points `attachments` and `author` already have — response
field, wire `to_dict`/`from_dict`, history row, `conversation_messages`
column, `_for_provider`. Until then, a Claude profile needs an explicit
`none` or a `null`. The failure is loud, which is what makes the default worth
taking now.

**The merge happens where the brain is resolved, not where `ModelRequest` is
built** (`_invoke_inner`): the profile's params go *underneath* the request's,
so an escort that set one overrides the profile and one that set none inherits
it. Ordering it this way is what makes the paragraph above true — an escort
promoting a turn to a stronger model by naming it must pick up *that* profile's
effort, not carry the cheap model's along.

**It is per LLM profile rather than per agent profile**, which is the same
handle-not-the-thing argument one paragraph up read from the other end: an
agent profile *names* an LLM, so a subagent scoped to a cheap model inherits
that model's dial with nothing to wire, while putting the dial on the agent
profile would mean two places to look for one answer.

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

*Where it lands, and how often it is recomputed.* Both are answered by a
**declared cue** — `agent_prompt_refresh`, an ordinary AST-read literal beside
`agent_prompt` — and the table is `prompt_cues.py`, a top-level module beside
`trees.py` in the same "one table everything reads" spirit. Seven rungs ordered
least to most frequent (`never`, `load`, `config`, `session`, `turn`, `write`,
`call`), each adding one component to the cache key, so a plugin at rank L is
keyed on everything at or below L. That monotonicity is the whole point: a
threshold can be ordered, a set of unrelated triggers cannot.

`never` is the string shape and is **not declarable** — a method that never
recomputes is precisely the permanent cache this replaced. `write` is the
default, so an undeclared method is never stale.

Two rungs cost no call site, and for opposite reasons. `session` is not an event
at all: it is a fact — session key, conversation, user, profile, security mode,
exactly what `sdk.session.get()` answers — read off the `PromptContext` as it is
built, so the turn-scoped `yolo` that `HookRegistry.finish_turn` clears by
writing the session field directly is covered with nothing having to remember to
fire. `load` needs no counter for a plugin's *own* reload, because discovery
builds a fresh adapter and the cache goes with the old object; it has one anyway
for the other reading, which is what an author means by declaring it — a package
install writes its files from kernel code and never passes `Interpreter._settle`,
so a prompt describing what is installed was previously invalidated only by
coincidence. That leaves three fire sites: `_settle` (`write`),
`HookRegistry.start_turn` (`turn`), and `config_manager._emit_config_changed`
(`config`, the one funnel both `save` and `save_plugin_config` pass through, and
skipped on an empty change list because `save` merges DEFAULTS and announces
unconditionally).

The rungs at or below `STABLE_THROUGH` (`config`) cannot move within a
conversation, so their text rides in the semi-stable block of the cacheable
position-0 message; everything finer goes in the dynamic `[SYSTEM CONTEXT
UPDATE]` block with the kernel's own live state — exactly the argument
`_mode_suffix` already makes for itself. Within each block, rarest first, with a
**stable** sort so one rung keeps `_in_scope`'s reading order. That ordering
pays in the prefix and is cosmetic in the dynamic block, which is rebuilt per
call either way.

**The tier is enforced, not promised.** A cue below `session` is answered with
the plain kernel context, so `sdk.session.get()` tells it nothing. Note the
narrower claim this makes: the position-0 message was *never* session
independent — its tool catalog is profile-scoped and its command catalog is
filtered by the session's frontend — it is stable for the life of one session,
and the tier is what stops a plugin being the reason it stops being.

`_collect` takes a `stable` flag now rather than `live`, and the shape is still
its own question: `callable()` decides *how to ask*, because the
string-shadows-a-method tolerance above is what depends on it, while the cue
decides *where it lands*. `_in_scope` still enumerates the populations once for
both passes.

This replaced one global counter (`sandbox/epoch.py`, now folded in as the
`write` rung) plus a `(session_key, security_mode)` variant bolted on beside it.
The counter's own reasoning survives intact and still matters: **`llm.delta` is
excluded** — it is a write and is *not* in `READ_ONLY`, but a streaming backend
sends one per token, so counting them would invalidate everything on every call
and silently undo the caching; the ledger's sandbox sink excludes it from
recording for the same shape of reason and draws the line the same way. And
**refusals do not count**, or `lockdown` would recompute every write-cued prompt
on every denial. The rung stays global rather than per-`Request.family`, and
the cue is why that is now right rather than merely safe: a plugin declares *how
often* its text moves, never *which family* it reads, so scoping the counter
would still mean inferring the dependency. What the declaration removes is the
need to guess the frequency.

One honest cost. Because a rung includes the rarer ones, the default gained a
turn bump, a config bump and an install bump it did not have — a strict superset
of the old invalidation, so nothing can go stale that did not before, but one
extra recompute per turn for a plugin nothing wrote for. The answer is the
declaration: the three store tools that read nothing but the security mode
(`tool_glob`, `tool_read_file`, `tool_run_command`) declare `session` and stop
paying for every `fs.write` the agent does, which is the trade the ladder exists
to make.

The lifetime resets in `residency.py` sit on top of all of it (`forget_prompt`,
which clears text and stamp together — clearing only the text leaves the stamp
matching): a residency's prompt is only knowable while its box is open, which is
a question about the box rather than about the world, and no cue can answer it.

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

  **The act itself is `runtime/compaction.py`, because `/compact` performs the
  same one.** It moved off the loop for the reason every turn-scoped thing
  eventually does: `_active_db` and `_active_conversation_id` are set at the
  top of `drive()` and cleared in its `finally`, so a compaction asked for
  outside a turn could reach neither — and a compaction with no marker row and
  no `has_compaction_checkpoint` is an in-memory shrink that the *next* turn's
  `replace_conversation_messages` writes over the full transcript with. Losing
  the conversation, silently, as the reward for asking to tidy it.

  Two halves stayed behind on purpose. The **swallow** is the loop's:
  compaction observes a turn's context pressure and must never be why the turn
  fails. The command path deliberately does not swallow, and
  `compact_history` therefore *names* each way of doing nothing — nothing to
  compact, no compactor installed, an empty summary, a drive already holding
  the list — where the loop had four identical silent returns. And the
  **notice** is the loop's: the automatic path narrates because the history
  changes under the user's feet mid-turn, while `/compact` was asked for by
  somebody watching a command that narrates itself.

  `ConversationRuntime.compact_session` refuses while `session.busy`, which is
  set only around the agent turn — so it is exactly "a drive owns this history
  list right now", and a command's own phase does not set it. The Request
  takes **no session key**: it scopes to `ctx.session_key`, which is stronger
  than checking an argument.

  **`session.compact` is `ALWAYS_UNSAFE`, and the reason is irreversibility
  rather than destruction.** Those come apart here, which is why it is worth
  stating. Compaction *deletes nothing* — `save_compaction_marker` appends a
  row and every original message survives in `conversation_messages` — so by
  volume of data lost it is strictly gentler than `conv.clear`, which runs a
  real `DELETE` and is `ALWAYS_SAFE`. What it does instead is permanent:
  nothing anywhere removes a marker, `messages_to_history` honours the latest
  one on every load, and there is no un-compact. That is exactly the test the
  db-write branch already states for itself — *the write worth asking about is
  the one that cannot be undone by writing again*.

  There is deliberately **no `chain.typed_command` exemption**, which is where
  this parts company with `config.write` and `session.set_mode`. Both of those
  have a way back, so the person who typed the command can fix a mistake;
  here they cannot, so `/compact` declares `require_approval` and answers for
  the grant up front like any other consequential command. It is in
  `ALWAYS_UNSAFE` *and* `_BRANCHED`, like `conv.delete`: membership alone
  renders "changes what the system can do", which is true of everything in the
  set, and the branch supplies the two facts a person actually needs — that
  there is no way back, and that their transcript is not being deleted.

  One thing this surfaced: `/compact` is the only gated command declaring
  `ui.progress`, so it is the first to ever *render* a grant for it — and the
  `ui` family fallback claimed the command would "ask you questions", which
  progress never does.
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
  **And the write path has to apply the same rule**, which for a long time it
  did not: `_config_write` routed to plugin_config on `scope="plugin"` *or* on
  the key appearing in any plugin's `config_settings`, with no kernel
  exception — so the timekeeper's own `scheduled_jobs` (declared by both) was
  written to two files and announced twice per change, and `rehome_kernel_keys`
  moved the stray copy back at every boot. `config_manager.is_kernel_setting`
  is that one rule, and it overrides the caller's `scope`: a plugin cannot
  rehome a setting it does not own.
- **An announcement names what a person changed, not what a file gained.**
  `save` merges `DEFAULTS`, so the first write after a schema addition
  persists settings nobody touched; diffing against the *file* called every one
  of them changed, and deleting a scheduled job announced `autoload_services`
  alongside it. The diff is against the effective previous config
  (`{**DEFAULTS, **existing}`) — materializing a default is a change to the
  file and to nothing else.
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

**A row says whose work it was**, and for a long time the sandbox origin was
the one that did not. `session_key`/`conversation_id`/`user_id` are columns
every enact site had filled since the beginning; the sink filled none of them,
which is invisible because a NULL column looks exactly like a table that has
not been asked yet. So `idx_ledger_conv` — an index on `(conversation_id, id)`,
built for this exact seek — was dead weight for the *only per-effect record the
system has*, and `my_action_ledger` (which scopes on `user_id`, `sandbox/
users.py`) hid every sandbox row from plugin code including its own. The fix is
`identity_of(context)`, read from the **context** rather than `chain.root`: the
root answers what *caused* the work and `policy.chain_session` recovers a
session from it only for agent-caused calls, while the context is the kernel's
own answer about whose call this is. It is right in both directions — a
`frontend.act` moves chain and context together, and a service polling on its
own initiative is handed `kernel_context(None)` and correctly records nothing,
because a service poll belongs to no conversation.

The context is the one `Interpreter._context_for` resolves, **not**
`execution.context`. Most executions carry none of their own and fall back to
the interpreter's, so reading the attribute directly answers `None` for the
ordinary case — and every unit test passes a context explicitly, so it takes
driving a real write through a real `Sandbox` to see it.

**The four filesystem Requests also copy their paths into `data_json.paths`**
(`FILE_ARGS`), plus `data_json.bytes` from a successful write's own answer.
That is not tidiness: `args_json` is capped at `LEDGER_JSON_CAP` and past the
cap the *object* is replaced by a `head`/`tail` wrapper, and the argument that
blows the cap is the file's own contents — so a reader parsing `args_json` for
a path loses exactly the largest edits. `fs.temp` is deliberately not among
them; its path is in the Result, which the sink does not record, so a scratch
file first appears as the `path` of the write that follows.

**`proc.run` and `proc.start` do the same through `shell.files_touched`**, and
the argument for a command-name table living *next to* the place a command-name
table was deliberately killed is the question it answers. `classify_shell`
decides **safety**, where a wrong "safe" is silent and grants something — Rice's
theorem, and the reason the ~500-line classifier had to go. This decides **what
to draw**, after the fact, where a miss is a row the drawer omits and a false
positive is a file shown that did not change. Both cosmetic, both visible to
whoever is reading the panel. So it is **display-only**, nothing in `classify`
or either recognizer may read it, and
`tests/test_shell_files.py::test_no_authorization_path_reads_the_file_table`
pins that structurally rather than describing it — the drift that undoes this
is one import, and it would look like a simplification at the time.

It abstains generously, which is what makes the table affordable: an unlisted
program, a glob (`rm *.log` names nothing until a shell expands it), and — by
building on `_command_segments` — redirection, substitution, subshells and
anything the lexer refuses. A command that exited non-zero deleted nothing, so
it records nothing. Paths resolve against the Request's `cwd`, and a recorded
path carries `data_json.via = "shell"`, because one read out of a command line
is a weaker claim than one the kernel serviced and the two must not pass as the
same thing. Note `cmd /c rmdir …` abstains correctly — the program is `cmd` —
which is what the `shell` argument exists to avoid writing.

The counterpart on the enact side is `data_json.attachments`, written by
`ConversationLoop._record_ledger` from the tool's `attachment_paths`. Files the
agent *showed* reach a frontend as an `attachments` render frame and were then
gone — a render is an event, and the transcript row for a tool call says
nothing about what it displayed. Between the two, one query answers "what did
this conversation do to files", which is what a client needs and could not
previously ask.

**Files a *person* sent are the other direction, and they are a column**
(`conversation_messages.attachments`, JSON, see **A message's files** below).
The ledger is for what the system *did*; an attachment is part of what a
message *is*, and it belongs on the message. That is why the two are not one
mechanism, and it is worth stating because they look interchangeable from a
client's side.

`ledger.read` grew the arguments to ask it (`conversation_id`, `action_types`,
`since_id`, `origin`, `session_key`) — every one narrowing in SQL, because
"read it targeted, never linearly" is only advice until there is something to
target with. It stays `ALWAYS_SAFE`; what it gained is `_check_access`, since
rows carrying an owner make naming somebody else's conversation a question
worth refusing. Same shape as `fs.search`/`fs.list`: a Request grew arguments
rather than the vocabulary growing a type.

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

## Reading a conversation back

`conv.read` answers with a **page**, and the read is bounded by **bytes**. Both
halves are load-bearing, and the second is the one that lasts.

It was an unbounded `SELECT *`. On a real conversation that came to 20.13 MB,
of which **19.25 MB was state markers** — `save_state_marker` appends a row
carrying the whole packed state on every action, and `to_dict` bounds
`history[-100:]` by *count*, so one `edit_file` argument of 102 KB rode along in
every marker for the next hundred actions. 96% of the table was bookkeeping the
model never sees (`messages_to_history` skips it) and no client renders.

Past `protocol.MAX_MESSAGE_BYTES` the answer stopped being deliverable, and the
shape of what followed is the part worth remembering. The caller was the HTTP
frontend's `poll`; `_deliver` ran *before* `sdk.http.drain()` and was
unguarded, so one oversized answer stopped every other client's request from
being served. `facade.collect_act` deletes an act's result **before** it
crosses, so the one-shot delivery was consumed and destroyed — a browser
waiting forever on an answer nobody still had. And `_drive_polls` resets its
failure count on any success, so it hovered at 1/5 forever and never reached
the five that stop a frontend. A dead UI, and nothing anywhere said so.

**Dropping the markers alone would only have moved the wall**, which is the
reasoning to keep. A transcript grows without limit *independently of any
context window*: compaction shrinks what the model sees and deletes nothing, so
there is no fixed size at which "all of it" stays answerable. Markers out is a
23× reduction and buys a 2M-token context comfortably; paging is what makes the
question answerable forever.

**Bytes rather than rows**, because a row count bounds nothing when one row can
be a 100 KB file edit — the exact thing that caused this. Rows are measured as
they are collected rather than estimated, since JSON escaping is content
dependent and an estimate wrong in the generous direction recreates the bug.
`CONV_MAX_BYTES` is derived from the wire the way `fs_net.MAX_READ_BINARY` is,
for the reason that comment already gives: two constants guessed apart drift,
and what they drift into is an unsendable result.

The paging arguments are **`ledger.read`'s**, not a new Request type —
`before_id` walks backwards, `since_id` walks forwards, and `since_id=0` is
therefore the oldest page with no third `order` argument to get wrong.
`limit=0` is metadata only, for the three callers that pulled a transcript to
read a title. The default is the *newest* page, because that is what opening a
conversation means; the one caller wanting the other end (`task_update_titles`,
which slices `[:12]`) has to say so, and would otherwise have started titling
conversations from wherever they had got to.

Two things stayed as they were, deliberately. `get_conversation_messages` is
still unbounded and still returns markers: its callers rebuild agent history
and `latest_state` needs them, so narrowing it would have broken restart
recovery to fix a display problem. And **compaction markers still cross** — a
state marker is bookkeeping, a compaction marker is a fact about the
conversation, and a client draws a divider from it. `details` no longer returns
the raw `state` blob at all; it is the marker itself, and the two fields
derived from it were what `details` was ever for.

**And the boundary now degrades rather than dying** (`interpreter._deliverable`).
An answer that cannot cross is replaced by a small coded failure at `_settle` —
the one funnel every serviced Request passes through, which is what makes it a
property of the kernel instead of of whichever handler was patched last. Both
runners needed it and only one had anything: in-process `InterpreterChannel`
caught the `ProtocolError`, while `runner_subprocess.send` catches `OSError`
and `ValueError` only, so over a pipe the error escaped `service_until` and
killed the box. `ERROR_TOO_LARGE` is breakage rather than a denial — nobody
said no, it simply did not fit — and the two causes are caught separately,
because `to_dict` refusing a live object is the handler's own bug while
`encode` refusing a payload is a question worth asking again more narrowly.

## A message's files

`conversation_messages.attachments` is a JSON list of records —
`{path, file_name, modality, extension}` — and it exists because a file is a
thing, not a sentence about a thing. It used to be a sentence:
`parse_attachment` welded `[Attached image file: chart.png (cached at …)]`
onto whatever the person typed, and that string *was* the only record. So the
one row that says a file arrived said it in prose. A client could get it back
only by parsing English, a person who typed those characters was
indistinguishable from a file, and `content` answered two questions at once.

**The pointer line is a rendering, and it is rendered at call time.**
`ConversationLoop._for_provider` is the one place history becomes provider
messages, so that is where the record becomes text again and where the key is
*dropped* — `messages` goes to a provider API verbatim, and a field no schema
knows is either rejected or silently believed. The line is deliberately
byte-identical to the old welded one, because every conversation written
before the column still carries it in `content` and the model must not meet
two spellings of one thing.

Three consequences worth knowing. **The record is the durable half**:
`Attachment.record()` drops `parsed_text` and `metadata`, which are a
rendering of the file for one model on one turn and can be four thousand
characters of a PDF — the file is still on disk to be parsed again.
**`replace_conversation_messages` has to carry it**, since
`iterate_agent_turn` rewrites a whole conversation from live history, and a
key dropped there survives until the next background turn and then does not.
And **files count as content** in `absorb_user_action`: the guard used to be
`text` alone and worked only because the pointer made an uncaptioned photo's
text non-empty, so reading it as "did the person say anything" now drops the
only record that a file was sent.

Nothing rewrites a message somebody sent. Rows written before the column keep
their welded text and no record, which is exactly how they have always read;
there is no backfill, because deciding where prose ends and a file begins by
regex over somebody's own words is a guess with no upside.

## Who wrote a row

`conversation_messages.author` answers the question `role` cannot. `'system'`
was taken from the beginning — it means a state or compaction marker, packed
JSON that `messages_to_history` skips wholesale — so **six kernel mechanisms
write `role='user'` rows the person never typed**: the `reveal_user_commands`
note (`command_note`), a doorman's `SendBack` (`doorman_note`), the cancel
notice (`cancel_notice`), the compaction bridge (`compaction`, from both the
live rewrite and the re-derivation on load), the emergency-truncation bridge
(`truncation`), and any plugin's `conv.append`.

**NULL is the meaning**, not an absence of one: it says the row is what its
`role` says, which is true of every row written before the column, so there is
no backfill and nothing normalizes it to `""` the way `_pack_attachments`
normalizes to `[]`. This is deliberately the *opposite* call from the ledger's
identity columns — there NULL was a bug because the column was meant to always
be filled, and a NULL looked exactly like a table nobody had asked yet.

**The vocabulary is open, like a notification's `source` and for the same
reason.** `conv.append` lets a plugin write a user row and no frozen enum can
name it, so `handlers/kernel._conv_append` stamps the provenance chain leaf via
`_notification_source` — the part a box cannot forge. A plugin free to state
its own authorship could leave it blank and write a row indistinguishable from
something the person typed, which is the whole thing the column prevents.

It is threaded exactly where `attachments` is threaded — `save_message`,
`save_history_message`, `replace_conversation_messages` (a key dropped there
survives until the next *background* turn and then does not), `messages_to_
history`, and the `SESSION_MESSAGE` payload, whose `actor_id` collapses every
user-role row to `"user"` and so has the identical problem a table has.

**`_for_provider` strips it, and the check runs before the attachments
shortcut.** That function returned `msg` untouched when there were no files,
and an authored row carries none — so the one fast path was the one case the
key always took, and a field no provider schema knows is either rejected or
silently believed.

Two readers were already wrong and are fixed: `dispatch.latest_user_text` feeds
the **conversation title**, so a `/cancel` titled conversations "[The user
cancelled the previous turn…]"; and `_compact` labels the compactor's
transcript by role, computed separately from `_for_provider`, so a notice
entered the summary as `USER: …` and that misattribution survived into every
later context. `_split_current_turn` and the emergency-truncate scan still ask
only about `role`, deliberately — see their docstrings.

The column is **advisory**: `sandbox/users.py` neither knows nor cares about it
(`conversation_messages` is a kernel table, not user-scoped), so nothing forces
a query to honour it. The store's memory retriever and curator were each
reading authored rows as user speech and now filter on it; a new consumer
must opt in the same way, which is why `tool_sql_query`'s agent prompt says so.

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

One `BaseFrontend` hook is **not** on the wire and a migrated frontend
therefore loses it: `render_conversation_banner` (mirror the conversation title
on a persistent surface). Telegram used it and gave it up. It would fit the
existing shape, so bringing it back means growing `KINDS` in
`sandbox/frontends.py`, the `native_names` map in `_adapt_frontend`, and the
test that pins the set.

There were two. `render_queued_ack` returned True to replace the textual
mid-turn ack with something like a message reaction, and it is **gone** rather
than pending — a worked example of the notification split retiring a hook
rather than needing one. Its whole job was suppressing a sentence, and the
sentence no longer exists: the mid-turn receipt is a `persist=False`
notification now, so a frontend that wants to answer it with a reaction does
that in `render_notification`, which *is* on the wire and does not need a
return value. Carrying `queued_ack` would have meant a render call whose return
value matters, which the one-way `_render` deliberately is not — so the thing
that made it unmigratable is exactly what made it unnecessary.

`BaseFrontend` itself is **not** migrated and should not be: its 880 lines are
host-side routing — fifteen bus subscriptions funnelling into ten `render_*`
methods, and `submit_*` funnelling into `runtime.handle_action`. The base owns
*when*, the guest owns *how*, so the base becomes the adapter. The ten
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

**A frontend may act *as* one of its sessions, and that is what made a real
client possible** (`frontend.act` / `frontend.collect`). A frontend box is
rooted `frontend:<name>`, which names no session, so `attended_now` answers
False for it forever: everything unsafe is refused rather than asked, and
everything reading `ctx.session_key` acts on nothing and reports success. Right
for a frontend acting on its own initiative, wrong for one serving a request a
person just made — and it left an HTTP frontend able only to *read*, with every
write routed back through `submit_text("/command …")`, i.e. through a `FormStep`
flow built for a human. The store's AG-UI plugin was synthesizing
`/conversations 'Main' 7 'Load conversation'`, positionally, with hand-rolled
shell quoting, because `parse_command_line` lexes with `shlex`.

`act` runs one Request rooted at the session, with that session's context. It
exempts nothing: same gate, same `classify`, same ledger row. What changes is
that **attendance now decides, and attendance is what that frontend declared**
through `frontend.attend`. So the grant is exactly "act as a session you own
while somebody is watching it", and marking it unattended takes the authority
back. That self-limiting property is why the chain roots at the session rather
than at `user`, which would be unconditionally attended and would take the
decision away from the mechanism built to hold it. A `frontend:<name>` link is
pushed on top so the ledger can still tell this apart from an agent tool call
in the same session.

It moves **both** chain and context, unlike `PersistentBox.call(for_session=)`
one file over, which deliberately moves only the context because a service
standing at a hook doorway is still acting on its own initiative. A frontend
serving a person is not, so the answer differs — and the context has to move
regardless, or `conv.load` and its neighbours go on acting on nothing.

Three things are host-side and unstatable by the guest. **Ownership** — the
token says who is asking, the runtime's session tags say which sessions they
may speak about; this also closed a gap in `frontend.attend`, which took any
key at all, so one frontend could declare another's session attended and thereby
arrange for somebody else's user to be asked. **Identity** — the kernel supplies
the token for an inner `frontend.*` Request. **Reach** — `act` refuses itself,
`collect`, and the whole `http.*` family, which belongs to the transport rather
than to any session.

And it is **detached**, which is correctness rather than speed: a box serves one
call at a time and the dialog renders back *into the calling box*, so inline it
deadlocks until the dialog expires. Same shape as `_drive`. The proof is
`tests/test_sandbox_frontend_act.py`, whose approver calls into the asking box
exactly as a real one does — with `Sandbox.act` made inline, that test hangs
rather than fails, which is the bug it exists for.

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
`sdk._Markdown` is a copy of `formatters.py`: measured line-for-line when both
still held the full set, the six name-alike pairs (`table`/`md_table`,
`card`/`detail_card`, `quote`/`quote_block`, `plain`/`render_plain`,
`truncate`/`truncate_cell`) shared **0%** of their bodies, and
`align_tables`/`align_md_tables` shared 43%. Four of those host-side halves
have since been deleted for want of a caller; the two that remain agree with
their guest counterparts because a test says so, which is the right
coupling. Same finding holds for the hook
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
`UNSAFE` by default; the migrated tool contains no
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
The structural recognizer also admits conservative `ls` display forms and
`cat` only for the same regular, non-protected, size-bounded files `fs.read`
would expose. Globs, stdin, devices, unsupported flags and non-POSIX aliases
abstain.

**And `protected.py` caps both recognizers, which it did not always.**
`_read_only_cat` consults the deny-list itself, so the structural half was
never the hole; `_remembered_prefix` asks about *coverage* and nothing else, so
one "always allow: cat" made `cat config.json` a SAFE Request that returned
every `secret_*` setting in plaintext with no dialog — the exact back door
`protected.py` exists to close, reopened one layer up where nothing was looking
for it. `_names_protected_path` is the cap, and its shape matters twice over:
it **withdraws an allowance rather than denying**, so the command lands at the
dialog it would have reached with no recognizer at all and "abstain, never
deny" survives intact; and it reads **literal operands only**, because a glob
names nothing until a shell expands it and chasing that is the undecidable
direction. That last gap is closed from the other end — `command_prefix`
declines to name a grant for `cat`/`ls` carrying a path expansion, so there is
nothing for a remembered grant to match.
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
branch: **where** the file is (missing, misplaced and invalid source fails in
launch preflight rather than becoming an approval denial), and **what it
imports** — a foreign library makes it `UNSAFE` and the dialog
names the library. That last rule is deliberately stricter than the plugin
equivalent: an installed package importing one is subprocessed and not asked
because somebody approved it at `plugin.install`, whereas a script was never
approved by anybody. Scripts are subprocessed in *every* tree, the one place
`required_isolation` skips the per-tree answer. The verdict is re-derived by
the kernel from the path and never supplied on the Request — a caller passing
its own report would be the contained code judging its own containment.
Launch revalidates the current bytes and requires proof that this Request was
approved before starting newly foreign code, closing the classification/run
race. Requests made inside the script are still judged one by one.

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
`subagent_timeout_seconds` (600, which is also the config maximum — see
**Deadlines are hard cutoffs** below for why the slider stops exactly at the
wall ceiling) and can exceed the wall ceiling without ever exceeding its
deadline. Raising the declared timeout does not help that case, and the
ceiling is deliberately global.

`handlers/kernel._script_run` waits in slices rather than blocking, because
cancellation reaches code that is *making* Requests and this handler makes
none while it waits. `provenance.Caller` carries the calling `Execution` for
exactly that — a cancelled turn would otherwise leave the child running to its
ceiling on a held pool worker.

**And a detached script needed the handle `proc.start` already argues for.**
`wait=False` shipped answering `{"started": True}` and dropping the `Run` on
the floor, so the result could never be collected, the run never cancelled, and
its finishing never observed — fire-and-forget with no forget-me-not. Both
siblings already had the shape: `agent.spawn`/`collect`/`stop` for a background
*agent*, `proc.start`/`status`/`stop` for a background *process*. So
`script.collect` and `script.stop`, both `ALWAYS_SAFE` on the argument those
two already make — collecting reads a result already produced, stopping
narrows, and neither starts anything `script.run` did not already answer for.
What it buys is fan-out over ordinary code: starting eight scripts and
collecting eight, without every branch having to be a subagent and cost a model
call.

Three details are load-bearing. Ownership is the **chain root**, so two scripts
detached by one turn collect together and a guest cannot claim another's — the
root is precisely the part of a chain a box cannot state about itself. Delivery
is **one-shot**, like a subagent report, because "did I already handle this?"
is not answerable from outside; `Sandbox._collectable` is therefore a separate
registry from `_runs`, which drops a run the moment it finishes (right for
`/cancel`, useless for collection). And an uncollected result is **swept** at
`COLLECT_RETENTION` — nothing obliges a caller to come back, and a leak whose
only symptom is memory is the worst kind.

**A deadline nobody can see can only be discovered by being killed by it.**
`self.budget` is the counterpart, and the argument for a new type is that the
kernel is the sole holder: a guest may read a clock (`time` is an allowed
module), but the deadline measures *running* time — elapsed minus what the
kernel spent owing it an answer — and it can see neither that discount nor the
clamp its declared `timeout` got. So the watchdog was the only thing that ended
a long run, and it ends one by killing the box, discarding everything computed
on the way. `sdk.budget()` lets the loop stop itself and return partial work.

The numbers come from `Execution.remaining()`, and `Watchdog.watch` writes them
onto the execution rather than each of its three callers doing it — the same
argument `watchdog.overdue` already makes for sharing its comparison: two
copies of one deadline drift, and the drift is invisible until something dies
early. `release` clears them only if the ticket matches, since two watches can
overlap on one execution. Nothing in force answers `None` rather than a number,
because a fabricated ceiling would be believed. And it is `READ_ONLY` — a loop
asks every iteration, so counting it as a change would bump the `prompt_cues`
write rung per
tick and silently invalidate every `write`-cued `agent_prompt`, the same trap
`llm.delta` is kept out of the ledger sink for.

**`sdk.retry` is the third, and it is pure guest-side sugar over a signal that
was already there and read by nobody.** `Result.retryable` is set at ~18
handler sites — a locked file, an HTTP timeout, a dead box — and until this
existed the only consumers in the tree were tests asserting it was set. That
is the better arrangement than a guest-side rule about exception types, which
is what LangGraph's `RetryPolicy` has to do: whether a failure is transient is
known where it *happened*. `Denied` is re-raised first and unconditionally,
ordering that is load-bearing because it subclasses `Failed` — a helper
catching the general case first would sweep every refusal into the loop, and
each attempt is a fresh dialog for somebody who already answered.

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
catalogue with policy inputs), `docs/HTTP_PROTOCOL.md` (hand this to an agent
writing a *client* — the twelve render kinds with their real payload shapes, the
Requests worth calling, and the half-dozen rules a working demo cannot show;
`docs/http_reference_client.html` is that demo, for checking the bridge when
the client misbehaves).

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

**Its ceiling is the sandbox's wall clock, and the two must not be equal by
accident.** `wait=True` waits *inside* the calling tool's box, and
`watchdog.HARD_CEILING` is wall time that is deliberately not discounted for
being blocked on the kernel — so the caller's clock starts first, before the
child is even off the concurrency queue, and a child running to a deadline as
long as the ceiling can never be waited out. That is why the config slider
stops at 600 rather than the 3600 it used to offer: a setting whose upper half
silently cannot be reached on the path most callers take is worse than a
narrower one.

**Which limit ends a child used to decide what the agent was told.** Reaching
its own deadline produces a *report* — `state == "cancelled"`, carrying
`conversation_id`. Being killed by the caller's wall ceiling produced nothing
of the sort: a starved box never resumes, so the tool's error branch is
unreachable and what reached the agent was `runner.py`'s generic *"timed out
after 60.0s (declared None)"* — the **declared running-time** deadline, which
is not the limit that fired and points at a knob that cannot help. An agent
told only that a tool timed out concludes the work is gone. It is not, and the
conversation id is the whole of the way back to it.

**So a blocking handler asks two questions, and they live on `Caller`.**
`abandoned` is "does the caller still want an answer"; `out_of_time`
(`provenance.WAIT_MARGIN` of wall clock left) is "can it still be given one".
Four handlers wait like this — `agent.spawn`/`agent.collect` and
`script.run`/`script.collect`, which is two kinds of child times two phases,
both splits real. The rule was not: it was written out four times, and three
copies were missing half of it, with `agent.collect` blocking in a single call
that neither `/cancel` nor the ceiling could reach. `handlers/kernel.
_give_up_waiting` is the shared answer, and `registry.collect(stop=...)` is
what let the fourth site have a loop at all. What to *do* stays per-site,
because it genuinely differs: the two that started a child cancel it and name
its conversation, while the two that are collecting hand back what is ready
and leave the rest collectable.

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

## Notifications

**Telling the user something is not the same as saying it to them**, and for a
long time the system had no way to draw that line. Everything out of band went
through one channel — `CHAT_MESSAGE_PUSHED` → `runtime.push_message` →
`BaseFrontend.on_bus_message_pushed` → `render_messages` — so a plugin
registering itself and the agent answering a question arrived as the same call.
A frontend could not separate them because it was never told they were
different. The channel's payload had documented `title`/`kind`/`source`/
`source_id` from the beginning and producers set them; `on_bus_message_pushed`
read `message`, `attachments`, `title`, `session_key` and nothing else, so the
attribution half of the contract existed and had zero readers.

`NOTIFICATION_PUSHED` is the second channel, `notification` the eleventh render
kind, and `notifications` a table.

**The split is by who was speaking, and it is drawn at each emit site.** Two
things on the old channel are the agent's own turn and stay there: the model's
mid-turn narration (`conversation_loop.py`, deduped against streaming by
`_consume_streamed`) and `sdk.ui.render`, a tool showing the user a file.
Everything else — the plugin watcher, config announcements, a scheduled agent's
result or failure, compaction notices — is the system, and moved. The rule
"a push made while the agent's turn owns the session" is real but is
deliberately *not* what the code tests, because a future producer would land on
the wrong side of it silently: guidance that never appears looks exactly like a
plugin with nothing to say. `runtime/notifications.notify` is the one door, and
`ConversationRuntime.notify` the delegate that fills in the database and the
`user_id` off the originating session, the same way the ledger's `identity_of`
reads ownership from a context rather than from the caller.

**Frontends opt in, which is what made this cost nothing.** A transport
declaring `supports_notifications` receives the payload whole; one that does
not gets it flattened into markdown and sent through `render_messages` — byte
for byte what it saw before the kind existed. So the REPL and Telegram needed
no edits, and `frontend_http` needed none either, since it forwards any kind
generically. Exactly the bargain `supports_streaming` already makes for deltas,
and for the same reason: a client quietly ignoring a kind looks merely quiet.
Note the fallback has to live on the *bus handler*, not on `render_notification`
— `residency.RENDER_METHODS` replaces that method wholesale with the box
forwarder, so a default implementation there would never run for a sandboxed
frontend, which is all of them.

That map is module-level in `residency.py` rather than a local inside
`_adapt_frontend`, and only because of what it costs to get wrong: a kind in
`frontends.KINDS` and not in `RENDER_METHODS` (or the reverse) shows a person
*nothing* and raises nothing. The test that claimed to pin exactly this could
only ever check the guest half, because the host's half was unreachable from
outside the function. It is pinned as set equality now.

**`messages` means the conversation, and everything else was moved out of it.**
The notification split was the first of four; the rest followed the same
argument and finished it. A refusal is the `error` kind
(`RuntimeResult.add_action_result` also stopped delivering one *three* times —
`ActionResult.fail` sets `message` **and** `error`, so both branches appended
the same sentence before `error` was even populated). An announcement is a
notification: only two were left worth moving, `new_conversation` and the
restore-on-start notice, and moving the second fixed a silent gap, since
`_load_last_active`'s other caller — an identity switch — discarded the
returned string and announced nothing at all. And what a command *returned* is
`callable_output`, the twelfth kind, gated by `supports_callable_output` on the
same bargain: REPL and Telegram needed zero edits.

**Three sites kept their `messages` for a while, and the reason they did is the
trap worth naming.** A `RuntimeResult` is two things at once: what
`BaseFrontend._render_result` draws, and the return value of `conv.load` and
`session.cancel` — which a command reads back to build its own output. So the
*reader* was deciding the channel. "Loaded conversation: Main" and "Cancelled."
stayed on the chat kind because `command_conversations` and `command_cancel`
read `messages`, and routing them anywhere else made `/cancel` fall through to
`return "Cancelled."` and announce the opposite of the truth.

`handlers/kernel._runtime_answer` is the fix, and it is one key: **the answer
carries every channel and the caller picks.** A command reads `callable_output`
first and falls back to `messages`, so where the kernel puts a line is once
again only a question about the person looking at it. Both moved to
`callable_output` with it.

`Cancel.execute`'s own "Cancelled." was the third and had a different excuse —
that cancelling ends the turn, so it is conversation. It is not: `handle_action`
short-circuits every base-phase and busy cancel before dispatch, so an action
that reaches `Cancel.execute` always has a *frame* to pop, and
`BaseFrontend.submit_text` intercepts `/cancel` before the command runs either
way. What was actually on that channel was a settings form's own Cancel button
typing into the transcript — the same thing "Back." and "Skipped." were doing,
and it is marked `FORM_NAVIGATION` alongside them now.

**Cancelling is two gestures, and only one of them is a callable.** The three
"Cancelled." sites went to `callable_output` together, which was right for the
`Cancel` *action* and wrong for the two early returns in `handle_action`.
Typing `/cancel` invokes a callable by name; pressing a Cancel button invokes
nothing, so a client drawing command output in a panel had no command to put
the answer against. The React UI synthesized a phantom run named `output` to
hold it, and stopping the agent opened the settings screen.

The two early returns raise a `persist=False` **notification** and answer with
`data: {cancelled, subagents_stopped}`. The argument is already made two
branches below them: a message sent mid-turn does exactly this, and is the same
kind of event — mid-turn, user-initiated, worth seeing for a moment and worth
nothing afterwards. It is also the split `new_conversation` and `command_new`
already draw, where the action notifies and the command returns text.
`command_cancel` words its own answer from `data`, because a command invoked by
name that answers with nothing reads as having silently failed. Frontends with
no notification surface flatten it back into the chat, so nothing is lost on a
terminal — though not byte-identically, since "Cancelled." becomes a title over
a body the way every other notification already reads there. That flattening is
also why the whole thing was invisible until a client drew the kinds apart:
three channels look like one from a REPL.

**Every population reaching a person, and what carries it:**

| What it is | Channel | Written by |
|---|---|---|
| The agent's reply, and the person's own words | `messages` | `_drive_agent_turn` only |
| What a command or user-invoked tool returned | `callable_output` | `dispatch.echo_callable_result` |
| A form or approval acknowledging its own navigation | `callable_output` | `add_action_result`, on `FORM_NAVIGATION` |
| Anything that failed | `error` | `add_action_result`, stamped with `action`/`name` |
| The system telling the user something — a stopped turn included | `notification` | `runtime.notifications.notify` |
| A running command narrating itself | `tool_status` | `sdk.ui.progress` → `COMMAND_CALL_PROGRESSED` |
| The model's mid-turn narration; `sdk.ui.render` files | `messages` | `CHAT_MESSAGE_PUSHED` |

The last row is the only remaining producer on `CHAT_MESSAGE_PUSHED`, and it
belongs there: both are the agent's turn speaking.
`tests/test_message_channels.py` pins the complete set of writers to `messages`
by AST, in the style of `test_kernel_boundary.py` — because all three
stragglers were found by eye, and a line of chat nobody said is exactly the
failure nothing reports.

**None of this is visible from a terminal.** The REPL declares neither
`supports_callable_output` nor `supports_notifications`, so `BaseFrontend`
flattens both into `render_messages` and the output is byte-identical whichever
channel it travelled on. That is deliberate — it is what made the split cost
zero frontend edits — and it is also why getting the channel wrong is never
caught by trying it.

**`source` is stamped by the kernel, never stated by whoever asked.** For
sandboxed code `handlers/kernel._notification_source` reads the leaf of the live
provenance chain — available because `interpreter._execute` holds
`provenance.serving(...)` around every handler call — exactly as
`approval.describe_asker` does. Attribution is what a reader leans on to decide
whether to care, so a plugin able to name itself `plugin_watcher` would be
forging the only field that makes the panel worth reading. Kernel producers pass
a literal they do not choose at runtime.

**Raising one grew an argument; reading them back needed types.**
`sdk.session.push(..., notify=True, title=…, level=…)` — pushing text and
raising a notification are the same act aimed at a different surface, so the
vocabulary did not grow. `notification.list` / `notification.mark_read` are two
new types because no existing Request's subject is notifications, the same
standing `ledger.read` has. Neither takes a `user_id`: they scope to
`ctx.user_id` in SQL, which is stronger than checking an argument because there
is no argument to get wrong.

**Persistence is what a panel needs and a bus cannot give.** The stream only
ever answers "what happened since you connected", so a fresh page load would
start empty. Rows carry no foreign keys, for the ledger's reason — a
notification about a conversation must outlive that conversation being deleted,
or it vanishes exactly when its explanation is wanted. The write is guarded
*separately* from the emit, so losing the panel's copy never costs the live
delivery. Retention folds into the single `data_retention_days` sweep; don't add
a knob. `persist=False` exists for progress ("Compacting conversation…"), which
is worth interrupting for and worth nothing an hour later.

**One thing moved out of the message text.** `emit_fallback_push` used to weld
``Load this conversation: `/conversations 'Main' 7 'Load conversation'` `` into
the body — a terminal affordance inside prose. `conversation_id` is structured
on the payload now and `load_hint` carries the pre-rendered command for surfaces
with no better way; a client that can open a conversation itself uses the id and
ignores the hint.

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
the primitives in `sdk.md`
([sandbox/guest/sdk.py](sandbox/guest/sdk.py)): `table` for data tables,
`card(title, pairs)` for describe-style key/value cards, `quote` for prose
under a card (descriptions, previews, payloads), and fenced code blocks for
multi-line technical dumps
(/debug, /locations — rich renderers collapse single newlines in prose).
Tables must start their own block (blank line before), or GFM parsers fold
them into the preceding paragraph.

**They live in the guest because the callers do.** Every one of these was a
function in
[bundled/frontends/helpers/formatters.py](bundled/frontends/helpers/formatters.py),
back when commands and frontends were native code that could import it. A
sandboxed guest cannot import a host module at all, so the primitives were
rewritten guest-side and the host copies lost their callers one migration at a
time — the file kept fourteen of them for a while after the last one left.
What is still there is `md_table`, which `plugins/command_registry.py` uses to
build the command catalog before any guest is involved, and `render_plain`,
which is the oracle `sdk.md.plain` is pinned against. Reach for `sdk.md.*`;
add to `formatters.py` only for something the *kernel* itself renders.

Each frontend then renders by policy, not by sender: the REPL runs
`sdk.md.plain` (aligns tables, strips fence
markers); Telegram's rich path renders markdown natively but compacts
detail-card-shaped tables into code blocks, and its HTML fallback renders
tables/quotes as `<pre>`/`<blockquote>`. Don't invent a structured message
type for this — markdown is deliberately the interchange format (it is also
what the LLM emits, so frontends need exactly one rendering path).

`BaseFrontend` also exposes one optional per-frontend polish hook:
`render_conversation_banner` (mirror the session's conversation title on a
persistent surface; fed by the `SESSION_CONVERSATION_CHANGED` bus channel).

## Where to plug in

- **Add a slash command**: write a `BaseCommand` subclass from `guest.bases`
  as `command_*.py` in the workspace, installed package tree, or deliberately
  in [bundled/commands/](bundled/commands/) when it is true kernel behavior.
  Commands receive `sdk` in both `form(args, sdk)` and `run(args, sdk)`.
  **What it returns is its output and returning is the only route** — see the
  channel table under **Notifications**. `sdk.ui.progress` narrates a slow
  body, `sdk.session.push(notify=True)` raises a notification, and
  `sdk.session.push` without it speaks into the conversation, which a command
  never does. None of this is visible from the REPL.
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

  **How two hooks at one doorway settle a disagreement is per doorway, and
  follows the cost of being wrong.** `end_turn` is first-answer-wins, so an
  early `Allow` silences later doormen — a doorman that guesses wrong costs a
  turn. `vet_permission` is **deny beats allow**: every gate is asked and any
  refusal wins however late it comes, because a gate that guesses wrong costs
  a capability, and under first-wins a permissive gate loaded ahead of a
  restrictive one decided policy by *filename order*. `shape_scope` folds,
  each shaper narrowing what the last left. A malformed answer is an
  abstention everywhere.

  **`end_turn` is consulted on two of the nine ways out of
  `ConversationLoop.drive`** — a cancel, a priority handoff and a failed
  action all leave without asking anybody. `turn_finish` fires on all nine and
  carries `TurnOutcome.reason` (`model_finished`, `budget_exhausted`,
  `cancelled`, `priority_handoff`, `action_failed`, `no_action`, `crashed`,
  `redrive`), which is where a doorman finds out about the exits it was not
  asked about. The loop labels its own exits into `self._exit_reason` — a
  local would die at `drive`'s single `return`, and the caller has to be able
  to read it in a `finally` even when `drive` raised. `redrive` reaches an
  observer only when the drive budget voided a restart, since observers fire
  once per *logical* turn.

  **A shaper is never handed `runtime.tool_registry` itself.**
  `narrow_scope` writes `visible_tool_names` in place, and
  `active_tool_registry`'s layers are all conditional — with no profile scope
  and no pinned extras the "deepest layer" is the global singleton. So a
  per-session narrowing escaped process-wide *and* ratcheted, because the
  intersect is against the previous consultation's answer: a shaper whose
  answer legitimately varies could narrow but never widen back. It detaches a
  copy first, guarded by `hooks.has(SHAPE_SCOPE)` so installs with no shaper —
  every install today — pay nothing. Note the doorway is consulted `3 + one
  per model call` times per turn, and also at conversation load where
  `ctx.attended` is `False` because no session is active yet.
- **Ship a task with a schedule**: create the Timekeeper job from `on_install`
  (`sdk.services.call("timekeeper", "create_job", self.name, self.job)`,
  read-then-skip so an edited cron survives) and remove it from
  `on_uninstall`. Declare `service.call`.

  This was `default_jobs`, a declaration the orchestrator seeded at **every
  registration** — boot, install, hot-reload — skipping only a job that
  existed at that moment. A job the user deleted did not exist, which is
  indistinguishable from one never installed, so it came back at the next
  restart, wrote config to say so, and announced itself in chat. There was no
  way to say no; the base class even claimed the timekeeper tombstoned
  removals, which it never did, and the documented workaround was to disable
  rather than delete. The failure is the same shape as install-time config
  seeding: **an act that outlives a turn belongs at the moment somebody asked
  for the package**, and registration is not that moment — it happens on its
  own, repeatedly, for reasons the user never initiated. `on_install` is, and
  it already existed.

  Removal had to go with it: it was only ever the counterpart of seeding, and
  alone it would delete a user's schedule on every hot-reload of the file.
  `default_jobs` is now in `validator.RETIRED_DECLARATIONS` beside
  `isolation` — dropped from `declarations` so nothing can later read it as
  authoritative, and reported at its line, because a plugin whose schedule
  never appears looks exactly like a plugin with no schedule.
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
- [prompt_cues.py](prompt_cues.py) — when a plugin's `agent_prompt` goes
  stale, and therefore which block of the prompt it rides in. The rung a
  plugin declares is the whole of both answers; the fire sites are three.
- [parsing/registry.py](parsing/registry.py) — the file-type authority:
  routing, discovery, and `parser_for` (the importable half). Not a service,
  on purpose.
- [llm/registry.py](llm/registry.py) — the model authority: profiles to
  `Brain`s, the box pools, load/unload. Not a service, for the same reason.
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
  loads a resident box, `act()` sends one Request on its own thread so the
  box that asked stays free to render the dialog it may raise.
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
