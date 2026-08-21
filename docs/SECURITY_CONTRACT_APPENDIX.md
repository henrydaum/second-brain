# Appendix A — The Request Catalogue

*Companion to The Second Brain Security Contract. Every capability the kernel
exposes to sandboxed code, derived from the live call sites in the kernel tree
and on the `store` branch.*

---

## How to read this

Each Request is listed with the **policy inputs** that decide its security
level. The kernel policy function computes that level from three things, as
stated in the contract:

- **what** — the nature of the Request (its type and arguments)
- **who** — the chain of provenance (which plugin, called by which, rooted in
  a user turn / cron job / subagent)
- **where** — the destination (a path, a host, a table, a user)

The **Default** column is the level when nothing unusual is present. It is a
starting point for the policy function, not a fixed property of the Request —
`fs.write` to a scratch directory is safe; the same Request aimed at
`main.pyw` is not.

Three mechanisms referenced throughout are defined at the bottom: **secret
handles**, **per-user views**, and the **provenance root**.

---

## Not Requests: the SDK

Anything that does not touch disk, network, clock, or process is a plain
function in the SDK. It runs inside the sandbox, costs nothing, needs no
approval, and never reaches the ledger.

| SDK area | Contents |
|---|---|
| Text | truncation, tokenizing, diffing, normalization, slugs |
| Formatting | markdown tables, detail cards, quote blocks, code fences |
| Math | cosine similarity, vector ops, statistics |
| Time | cron parsing and description (`croniter`, `cron_descriptor`), date math |
| Encoding | JSON, base64, hashing, mimetype lookup |
| Logging | `sdk.log(...)` — writes to the kernel's sink, not the filesystem |

**The test:** does it touch disk, network, clock, or process? If no, it is SDK.
If yes, it is a Request.

Logging is the deliberate edge case — it does reach disk, but the SDK routes it
to the kernel's log sink so the plugin author never writes a Request for it.
Reuse that pattern wherever a Request would be too noisy to write by hand.

---

## Isolation: decided by provenance, never declared

How isolated a plugin runs is **not** a Request, and it is not a declaration
either. It was one — `isolation = "subprocess"`, read off the file by AST — and
that put the choice of containment in the hands of the code being contained. An
agent authoring a plugin could author its own escape from the process boundary
by leaving a line out. Code does not get a vote on how contained it is.

The replacement is provenance, which a file cannot assert about itself: writing
into `workspace/` is what *makes* something an agent-authored plugin.
Three trees, three answers (`sandbox/isolation.py`):

| Tree | Isolation | Why |
|---|---|---|
| `workspace/` | always subprocess | agent-authored; this is what the boundary is for, and it is what buys free authorship above |
| `plugins/` | always in-process | first-party kernel code, trusted by definition; a pipe would buy nothing and cost every call |
| `installed/` | subprocess **iff** it imports something unmediated | a store package that is pure computation over the SDK is as inspectable as kernel code |
| anything else | subprocess | unknown provenance fails closed |

"Unmediated" is **computed, not declared** — for exactly the same reason the
tree is. `dependencies_pip` is a declaration and would reintroduce the bug one
level down, so the answer comes from the validator's import walk
(`report.unmediated`): foreign libraries, plus the stdlib modules that do their
own path I/O (`sqlite3`, `zipfile`, `tarfile`). Declaring no dependencies while
importing `fitz` still gets you a subprocess.

A file that still declares `isolation` gets an advisory note saying it is
ignored, and the value is dropped rather than carried — a stale declaration
that reads as authoritative is how this became a vulnerability in the first
place.

**Boxes cannot be used to escape this.** Files group into a shared box by
declaring `box = "name"`, and a box takes the tightest isolation any member
asked for. The worry is a `workspace` file naming the bundled tree's box to
ride in-process beside it; it cannot, because isolation is computed per file
from that file's own path *before* any grouping, and tightest-wins can only
ever tighten from there. The worst a mislabelled file achieves is dragging its
box into a subprocess it did not need — a performance mistake, not an
escalation.

A user-facing override (a config allowlist) is planned. That is a different
thing and stays a different thing: a person may decide what the code may not.

---

## 1. Filesystem

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `fs.read(path)` | File contents as text | path | safe, except protected files |
| `fs.write(path, data, mode)` | Create, overwrite, or append text | path | safe in scratch, agent workspace, or user-configured writable folders; else unsafe |
| `fs.read_bytes(path)` | File contents as raw bytes | path | safe, except protected files |
| `fs.stat(path)` | Metadata for one file or directory | path | safe, except protected files |
| `fs.write_bytes(path, data, mode)` | Create, overwrite, or append bytes | path | safe in scratch, agent workspace, or user-configured writable folders; else unsafe |
| `fs.list(path, pattern, details, recursive, files_only, sort, limit)` | Directory listing, glob, metadata, pruning walk | path | safe |
| `fs.search(pattern, root, glob, regex, mode, ...)` | Content search across a tree | root path | safe; protected files skipped |
| `fs.delete(path)` | Remove a file or tree | path | safe in scratch, agent workspace, or user-configured writable folders; else unsafe |
| `fs.move(src, dst)` | Copy, rename, replace | both paths | safe when both paths are freely writable; else unsafe |
| `fs.temp()` | Allocate a scratch file or directory | — | safe, always |

`fs.temp` exists so that "I need somewhere to put this" never requires a policy
decision. Scratch space is granted under `workspace/temp`, not the operating
system's shared temp directory, so plugins cannot alter other programs' temp
files.

**The agent writes its own plugins freely.** Every path under
`workspace/` is writable, deletable and movable without a dialog, and
that grant is the return on the whole boundary. Code in that tree runs in a
subprocess — not because it asked, but because of where it is (see
*Isolation*) — so it is contained before it ever runs. Approving each edit
would buy nothing containment has not already bought, while costing the thing
that makes an authoring agent worth having: writing a plugin is a dozen edits,
and a dialog on each is a dozen interruptions to approve something that cannot
act unmediated anyway.

Be precise about what this does *not* grant, because it is the LibOS
invariant exactly. Writing a file changes what the system can **ask**. It does
not change what it may **affect**: the new plugin's own Requests are classified
like anybody else's, and it inherits no authority from having been written
without a dialog. Free authorship, unchanged authorization.

There is one separate write grant: `fs_writable_dirs` lists folders the
**user** has opened to the agent. Create, overwrite, move, and delete operations
inside those folders are safe because the user chose the destination in
advance, not because the agent owns it. Treat their contents as user data and
change only what the task requires. Second Brain's source and installed-package
trees remain protected from this list even when a listed parent contains them.
Writing anywhere covered by neither grant is unsafe as before.

`fs.search` is derivable from `fs.list` + `fs.read`, and is a separate Request
anyway: doing it by hand costs one round trip per file.

**Both grew a second shape rather than a second Request** (`sandbox/walk.py`).
A real tree search needs regex over file contents, junk-directory pruning, an
enumeration cap, and ripgrep when it is installed — none of which sandboxed
code can do for itself, and all of which the first plugin to want them would
otherwise have built privately behind `proc.run`. So the engine moved host-side
and the two Requests grew *arguments*: pass any of them and the answer arrives
as a dict with `truncated` / `scan_truncated`; pass none and the original bare
list comes back byte-identical. The authorization surface did not move — which
types exist, and what `classify` says about each, are exactly as before. This is
the same shape as the parsing and LLM migrations: the boundary got *narrower* by
adding kernel code, not by widening what a plugin may ask for.

The ripgrep fast path is host code, so it costs no `proc.run` dialog — but it
knows nothing about `protected.py`, and content hits carry matching lines. Its
results are therefore filtered through `is_protected` before the limit is
applied, or the fast path would hand back exactly the config lines the slow path
exists to withhold.

`fs.stat` is the one-path metadata Request: it adds `is_file`, `is_dir`,
`is_symlink`, `size` and `mtime` (`st_mtime_ns`, an int so it survives JSON
exactly — compare with `!=`, since a restored older version also changed).
`sdk.fs.exists` uses the same Request with a missing-safe argument. `fs.list`
still returns those core details when `details=True`, but callers no longer
need to disguise a metadata lookup as a listing.

**Protected files** (`sandbox/protected.py`) are the one place a read Request
refuses on the path alone: `config.json`, `plugin_config.json`, and the SQLite
database with its sidecars. Both exist because a control enforced on one
Request was walkable around on the next. `config.read` hands back a
`<secret:…>` handle and `secret.reveal` prompts — but `config.json` holds the
same credentials in plaintext, so an unrestricted `fs.read` made the handle
decorative. The database is reachable through `db.query`, which scopes rows per
user and refuses `password_hash`; reading the file walks around all of it.
`fs.search` counts as a read here because its hits carry matching *lines* —
`pattern="secret_"` would do the job by itself. Writes need no separate read
protection rule: these files sit outside the agent workspace, and the
`fs_writable_dirs` grant explicitly carves out protected kernel data, so editing
them still asks.
Directories are not protected; listing a folder reveals nothing the path
constants do not already say.

**Bytes are a separate pair, not a flag.** `fs.read` decodes UTF-8 with
replacement, which is right for text and silently destructive for anything
else — a JPEG through it is no longer a JPEG. The byte Requests carry their
payload base64-encoded because JSON has no bytes type; the SDK encodes and
decodes, so a plugin only ever sees `bytes`. Policy treats each pair
identically: the same act with a different encoding must get the same answer,
or the encoding becomes a way around the rule. The one asymmetry is the size
cap — binary reads get a larger one, because a 20 MB video is ordinary where a
20 MB text file is a mistake.

## 2. Database

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `db.query(sql, params, max_rows)` | Read rows (reads only; capped at 500) | resolved tables/columns, user | safe |
| `db.write(sql, params)` | Insert, update, delete | leading keyword, mentioned tables/columns | safe; **unsafe** for `DELETE` on a kernel table; **refused** for DDL on one |
| `db.define(ddl)` | Create or alter a plugin-owned table | leading keyword, mentioned tables | safe; **refused** on kernel tables |

Reads stay deliberately unrestricted. Free reads are safe **because the exits
are gated** — a plugin that reads everything still cannot send anything
anywhere without passing `net.http`, which is always checked. Do not trade away
the agent's reach over its own database to solve a problem that egress control
already solves.

Two narrow exceptions, both structural rather than policy:

- `users.password_hash` is denied at the column level. It is the only secret
  column in the schema.
- User-scoped tables are reached through **per-user views**, not base tables.

**For writes the line is data versus structure, not kernel versus plugin.**
Changing rows cannot change how the kernel works — only changing structure can,
because every query the kernel issues keeps working against edited rows and
stops working against a dropped column. So:

- **Rows in kernel tables are writable.** The named Request is still the better
  route where one exists (`conv.append`, `user.write`, `ledger.record`,
  `file.register`, the `task.*` family) because it carries the owner check and
  emits the bus event frontends redraw from. But not every kernel *column* has
  a Request behind it — `conversations.last_title_check_message_count` is a
  high-water mark with no `conv.*` verb — and a task maintaining one should not
  have to mirror rows it can already read into a shadow table.
- **Schemas in kernel tables are not.** `CREATE`/`DROP`/`ALTER`/`RENAME`/
  `REINDEX` naming one is refused.
- **`DELETE` on a kernel table is unsafe rather than safe**, so it prompts. It
  is the one row write that cannot be undone by writing again: an `UPDATE` with
  a bad `WHERE` leaves the rows there to fix, and a `DELETE` with the same bad
  `WHERE` leaves somebody asking where their conversations went. Legitimate but
  irreversible is what a dialog is for. Plugin-owned tables are excluded — a
  dialog every time a plugin tidies its own cache is how people learn to stop
  reading dialogs.
- **`PRAGMA`, `sqlite_master`/`sqlite_schema`, `ATTACH`/`DETACH`/`VACUUM` are
  refused outright.** The first two because DDL is not only spelled `CREATE`:
  `PRAGMA writable_schema=ON` followed by `UPDATE sqlite_master SET sql=…` is
  schema surgery that starts with `UPDATE`, and without these the keyword check
  would ship with a hole. The rest because they act on the database *file* — and
  `ATTACH DATABASE '/etc/x.db'` names no table at all, so a per-table check
  waves straight through what is really filesystem access spelled in SQL.
- **`users.password_hash` is refused in writes as in reads.** Not a table rule:
  every other column on `users` is metadata a frontend may maintain, and this
  one has no bookkeeping use and an unrecoverable accident.

The check is an **identifier and first-token** check, not a statement parser.
Which names a statement mentions and which word it begins with are both
answerable by looking; what an arbitrary statement *does* is not. What makes the
first token sufficient here — and makes this different from the shell classifier
that died — is that `Database.execute_write` uses `conn.execute`, which runs
exactly one statement and raises on a `;`-chained script. There is no second
statement hiding behind a separator, so there is no race against chaining or
quoting to lose.

**One gap is left open deliberately.** A row write carries no owner check. Reads
solve that with the `my_` virtual name and writes cannot: SQLite will not
`UPDATE` a subquery, so there is nothing for the trick to expand into. While
every frontend is single-user this costs nothing; it becomes real the day a
`per_user` frontend is, and closing it then means inspecting `WHERE` clauses —
the fragile-parser shape this section otherwise refuses to build.

Raw `db.conn` and `db.lock` access is withdrawn — every current use migrates to
these three Requests. Transaction scoping becomes an argument
(`db.write(..., atomic=[...])`), not a borrowed lock.

## 3. Conversations

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `conv.create(title, activate)` | Start a current-user conversation, optionally loading it | user, session | safe |
| `conv.read(id, limit, before_id, since_id)` | One page of messages, plus metadata | id, owning user | safe (own), unsafe (other user) |
| `conv.list(filters)` | Enumerate conversations | user | safe |
| `conv.append(id, message)` | Add a message | id, owning user | safe (own) |
| `conv.set_title(id, title)` | Retitle | id, owning user | safe (own) |
| `conv.set_category(id, cat)` | Categorize | id, owning user | safe (own) |
| `conv.set_notification_mode(id, mode)` | Notification mode | id, owning user | safe (own) |
| `conv.load(id)` | Load saved state into the current session | id, owning user, session | safe (own) |
| `conv.new()` | Let go of this session's conversation; the next message creates one | session | safe |
| `conv.clear(id)` | Drop messages, keep conversation | id, owning user | safe (own) |
| `conv.delete(id)` | Delete conversation and messages | id, owning user | unsafe |
| `conv.enact(id, action)` | Drive an agent turn | id, owning user, root | unsafe from an unattended root |

**`conv.read` answers with a page, not a conversation.** It was an unbounded
`SELECT *`, and on a real conversation that came to 20.13 MB — of which 19.25
MB was the state machine's own marker rows, re-serialized in full on every
action. Past `protocol.MAX_MESSAGE_BYTES` the answer stopped being deliverable
at all, and because the caller was a frontend's `poll`, the failure took the
whole transport out rather than one request.

Two changes, and the second is the one that lasts. Bookkeeping is filtered
kernel-side (compaction markers survive — those are a fact about the
conversation, and a client draws them). And the read is bounded by **bytes**,
because a row cap bounds nothing when one row can be a 100 KB `edit_file`
argument, and because a transcript grows without limit whatever the model's
context window is: compaction shrinks what the model *sees* and deletes
nothing, so there is no fixed size at which "all of it" stays answerable.

The paging arguments are `ledger.read`'s, which had the same problem first —
`before_id` walks backwards from a row, `since_id` walks forwards from one, and
`since_id=0` is therefore the oldest page with no third argument to get wrong.
`limit=0` asks for metadata only, for the callers that came for a title.
`CONV_MAX_BYTES` is derived from the wire exactly as `fs_net.MAX_READ_BINARY`
is, so the two cannot drift into an unsendable result again.

Ownership is checked on every id-bearing Request, mirroring
`runtime.assert_conversation_access`. Cross-user access is refused and recorded,
never silently filtered.

## 4. Sessions

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `session.get(key)` / `session.list()` | Inspect live sessions | — | safe |
| `session.open(key)` / `session.close(key)` | Session lifecycle | key | safe |
| `session.push(key, message, title, notify, level)` | Proactive message to the user, or a notification when `notify` | key, attendance | safe |
| `session.state_get/set/clear(key, ns)` | Per-session plugin scratch state | namespace | safe |
| `session.set_attended(key, bool)` | Declare human presence | key | unsafe |
| `session.cancel(key)` | Cancel the running turn | key | safe |
| `session.compact()` | Summarize this session's history and shrink what the model is shown | — (scoped to the calling session; takes no key) | **unsafe** |
| `session.add_attachment(path, key)` | Stage a file for the next model call | — (handler applies the read deny-list) | safe |
| `session.set_profile(key, profile)` | Switch agent profile | profile | unsafe |
| `session.add_prompt_extra(text, key, slot)` | Inject system prompt text into one named overlay | `key` (which session) | **safe** for your own session; unsafe when it names another |
| `session.remove_prompt_extra(handle, key)` | Withdraw injected text | — | safe |
| `session.add_tool(key, name)` | Widen the agent's scope | tool name | unsafe |
| `session.remove_tool(key, name)` | Narrow the agent's scope | tool name | safe |
| `session.set_mode(mode, key, scope)` | How this conversation answers approvals | mode, `chain.typed_command` | **safe** for `lockdown` or a typed `/mode`; unsafe otherwise |

The asymmetry is intentional and runs through the whole catalogue: **widening
capability is unsafe, narrowing it is safe.** Adding a tool, injecting prompt
text, or claiming attendance changes what the agent may do next; the reverse
never does.

`session.set_mode` is that rule applied to the approval dialog itself, and it
is the one entry whose *effect* is on this table rather than in it — the mode
is what the approver answers with in place of asking, so it decides the
outcome of every later unsafe Request in the conversation. Two rules keep it
from being a way around the catalogue:

- **Polarity.** `lockdown` is the tightest of the three values, so arriving
  there narrows whatever we were in and an agent may do it unasked. Every
  other value could widen and is asked about — which the agent cannot answer
  for itself, since a chain nobody is watching is refused before the mode is
  ever consulted.
- **Provenance.** A slash command the person typed is its own consent, scoped
  to the command's own code exactly as `config.write` is. This is not a
  convenience: the mode is enforced *at* the approver, so the act that leaves
  lockdown must never reach it, or `/mode ask` would be auto-refused by the
  thing it exists to lift.

`scope="turn"` sets a mode the kernel drops at the end of the agent turn. It
is the first grant in the system whose unit is time rather than a destination,
and it needs no revocation surface for that reason — it is gone before anyone
could look for it.

`session.add_attachment` sits in the table looking like a fourth widening and is
not one: it changes what the agent can *see*, never what it may *do*. Its
authority is exactly `fs.read`'s — the guest could already read those bytes and
hand them back as an `llm_summary` — so the handler applies the same
`protected.reason_for` deny-list, and with that in place staging is the shorter
route to the same place rather than a new one.

## 5. User interaction

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `ui.ask(prompt, title, type, choices, required, default)` | Question with typed answer | attendance | safe if attended, refused if not |
| `ui.approve(action, justification)` | Explicit approval for a sensitive action | — | **always unsafe** |
| `ui.render(paths, caption)` | Show files to the user in chat | paths | safe |
| `ui.progress(message)` | One line about what a running slash command is doing | — | safe |

`ui.progress` is safe because it reaches nothing to decide about: the handler
resolves the *running command's* call id from the session and abstains when
there is none, so an agent-invoked tool, a task or a service calling it emits
nothing at all. A dialog would also be self-defeating — progress is emitted in a
loop, so asking would cost more interruptions than the work being narrated. It
is in `prompt_cues.RENDERING` and the ledger's `unrecorded` set for the same pair of
reasons `llm.delta` is: per-iteration volume, and text on a screen is not state
a prompt can read back.

`ui.ask` is definitionally safe when a human is present — it *is* the approval
channel. In an unattended session it is refused rather than queued, matching the
kernel's existing default at the `unattended_call` gate.

The handler translates `choices` into the state machine's `enum` and assembles
the prompt through `form_step_display`, so a sandboxed question renders with the
same assistance as a native form step. Both belong here rather than in the
asking plugin because the guest cannot import `state_machine` at all — and the
translation was missing for long enough to prove the point: `choices=` went
straight through to a parameter that does not exist, so every multiple-choice
question died as a `TypeError` reported back as "could not ask".

## 6. Configuration

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `config.read(key, scope)` | Global or user-scoped setting | `secret_` prefix | safe, **`secret_*` returned as handles** |
| `config.write(key, value, scope)` | Change a setting | key, scope | unsafe |
| `paths.get(name)` | Resolve a named application location | fixed name allowlist | safe |

Scope (`global` / `user`) is an argument, not a separate Request.

Two of `paths.get`'s names are not locations: `python` (the interpreter
hosting the app) and `platform` (`sys.platform`). They are here because the
validator refuses `sys` — correctly, it is a door to the interpreter rather
than a fact about it — while these two facts are things a plugin legitimately
needs and cannot otherwise learn: which Python `pip install` should target so
the package lands where Second Brain can import it, and which shell a command
line is being built for. Both are constants the kernel already knows, so
answering them costs nothing and closes the only honest reason to want `sys`.

This — not the database — is where the contract's "private information" clause
belongs. API keys and OAuth tokens live in config and the environment. See
**secret handles** below.

## 7. Users

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `user.read(id)` | User row, minus denied columns | id vs. current user | safe (self), unsafe (other) |
| `user.list()` | Enumerate users | — | unsafe |
| `user.write(id, fields)` | Update type or config blob | id, fields | unsafe |
| `user.set_credentials(id, ...)` | Set username / password hash | — | unsafe, always |
| `user.delete(id)` | Remove a user | id | unsafe, always |

`password_hash` is never returned by any Request, at any level.

## 8. Plugin lifecycle

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `plugin.list(family)` | Enumerate installed plugins | — | safe |
| `plugin.describe(name)` | Metadata, path, dependencies | — | safe |
| `plugin.validate(path)` | Lint a source file against this contract | path | safe |
| `plugin.register(path)` | Load a plugin live | path, family | unsafe |
| `plugin.unregister(path=... / name, family)` | Unload | recognized path or registered identity | unsafe |
| `plugin.reload(path=... / name, family)` | Reload in place | recognized path or registered identity | unsafe |
| `plugin.install(stem)` | Install from the store | stem, store commit | unsafe |
| `plugin.uninstall(stem)` | Remove, with dependency scan | stem | unsafe |
| `plugin.update()` | Update installed store packages | store commit | unsafe |

This family is the literal subject of the LibOS quote: the agent extends itself
here, and every widening Request in it is unsafe by default.

`plugin.validate` is the exception, and sits with the listings rather than with
`register` and its neighbours. It changes nothing: the validator is a pure AST
walk that never imports or executes the file it reads, so a `validate` that
returns "will not load" has left the system exactly as it found it. It is what
an agent authoring a plugin uses to check its own work after every edit, and a
dialog in that loop would only teach the agent to stop checking. Writing the
file was already free inside `workspace/`; *loading* it is the step that
asks.

## 9. Services

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `service.list()` | Loaded services and status | — | safe |
| `service.call(name, method, args)` | Invoke a method in the service's container | target service, method | safe — the callee's own Requests are gated with the caller in the chain |
| `service.load(name)` | Load a service | name | unsafe |
| `service.unload(name)` | Unload | name | unsafe |

`service.call` is safe *because of provenance*, not despite it. A service's own
Requests are classified with the caller in the chain, so nothing is laundered by
routing through a service — the earlier "calling a service is safe because
services are sandboxed" reasoning is replaced by this.

Native objects never cross the boundary. A service holding a model in memory
keeps it inside its persistent container; callers get simple data back.

## 10. Tools and commands

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `tool.list()` / `tool.schema(name)` | Discover callable tools | — | safe |
| `tool.call(name, args)` | Tool-to-tool composition | target tool | safe — callee's Requests gated with the chain |
| `command.list()` | Discover slash commands | — | safe |
| `command.call(name, args)` | One-shot slash command | target command name and arguments | unsafe; an approved call satisfies the command's own gated action and only its declared Request grant |

`tool.call` is safe and `command.call` is not, which is worth the distinction.
A tool is narrowed by the agent's scope and written to be called by other code,
and its own Requests are classified with the caller still in the chain — so
routing through one launders nothing. A command is the surface a *person*
types: not scope-narrowed, and the set includes package installation and config
editing. Running one on somebody's behalf gets a sentence, and the dialog names
the command.

Commands declaring `require_approval` are refused here rather than dispatched.
That answer comes from the state machine, which sets the approved flag on the
execution it authorized; nothing on this path can obtain it, and passing it
anyway would forge the consent the mechanism exists to collect.

## 11. Agent

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `agent.complete(prompt, schema)` | A model call | — | safe |
| `agent.spawn(prompt, wait, …)` | Run a subagent now | root, depth | safe |
| `agent.collect(ids, timeout)` | Take finished children's reports | — | safe |
| `agent.stop(id)` | Cancel a running child | — | safe |
| `agent.schedule(prompt, cron, …)` | Run a subagent later | cron, root | unsafe |
| `agent.escalate(reason)` | Re-drive on the strong model | — | safe |

`agent.complete` is its own Request and never a generic `service.call`. Keys,
sockets, and provider details stay kernel-side; the sandbox sees a prompt and a
schema.

`agent.spawn` is safe because a subagent **can approve nothing**. Its turn runs
on a session key that is never the active one, so the Requests it makes build a
chain rooted there rather than at `user`: `Chain.attended` is false, which
refuses `ui.ask` outright and denies every unsafe Request instead of asking
about it. The parent's `approved` grant is not inherited either — a turn is not
a nested call, and its chain starts fresh. A child therefore reaches strictly
less than whoever started it, which is the property that matters. The kernel
also refuses a spawn made *from* a subagent session, so the tree is one deep.

(This entry used to read "the child's Requests are gated with the parent in the
chain". That is not what happens — a subagent turn does not run inside the
spawning execution — and the real reason is the stronger one.)

`agent.collect` and `agent.stop` speak about children this caller already
started. Collecting reads a report the child has already produced; stopping
narrows, and is safe for the reason `session.cancel` and `proc.stop` are — an
agent that needs a dialog to end something it started will leave it running.
`agent.collect` is in `READ_ONLY`, so the ledger's sandbox sink drops it: with
`timeout=0` it is a poll, and a fan-out loop would otherwise write a row a tick.

`agent.schedule` is unsafe because it creates *unattended* future work, where no
one is present to answer a dialog.

`session.compact` is unsafe on **irreversibility**, which is the criterion the
db-write section states for itself: the write worth asking about is the one
that cannot be undone by writing again. Nothing anywhere removes a compaction
marker — `latest_compaction` finds it and `messages_to_history` honours it on
every load, forever — so a conversation folded into a summary has no way back
to being read in full.

Note this is *not* an argument about data loss, and the distinction is worth
keeping straight because it points the other way: compaction **deletes
nothing**. `save_compaction_marker` appends a row, and every original message
survives in `conversation_messages`, queryable. `conv.clear` — which runs an
actual `DELETE` — is `ALWAYS_SAFE` one set over. So on destructiveness alone
this would be the safer of the two. It is unsafe because the *effect* is
permanent, not because the *rows* are gone, and the dialog says both halves
so nobody answers no for the wrong reason.

There is deliberately **no `chain.typed_command` exemption**, unlike
`config.write` and `session.set_mode`. Those two have a way back — write the
setting again, set the mode again — so a person who typed the command can
undo a mistake. Here they cannot, so `/compact` declares `require_approval`
and answers for itself up front like any other consequential command. That is
the price of nothing else in the system being able to rewrite somebody's
conversation without saying so.

It scopes to `ctx.session_key` and takes no key argument, which is stronger
than validating one: there is no argument to get wrong.

It is also **not** in `READ_ONLY`. It changes what the next call sees, so it
belongs in the ledger and it should bump the `prompt_cues` write rung — a cached
`agent_prompt` computed against the pre-compaction world is stale the moment
this succeeds.

## 12. Scheduling

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `cron.list()` / `cron.get(name)` | Inspect jobs | — | safe |
| `cron.describe(name)` | Next fire time, human-readable schedule | — | safe |
| `cron.create(name, def)` | New job | channel, payload | unsafe |
| `cron.update(name, patch)` | Change schedule or payload | name | unsafe |
| `cron.enable(name, bool)` | Enable or disable | name | safe to disable, unsafe to enable |
| `cron.remove(name)` | Delete a job | name | unsafe |

Everything that creates recurring unattended work is unsafe, for the same reason
`agent.schedule` is.

## 13. Events and hooks

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `event.emit(channel, payload)` | Publish | channel | safe |
| `event.request(channel, payload, timeout)` | Blocking request/response | channel | safe |
| `llm.proceed(request)` | Place the call an escort is holding | token | safe |
| `llm.delta(text)` | Push streamed assistant text out of a backend | token | safe |
| `llm.list()` | Configured profiles, installed backends, the default | — | safe |
| `llm.load(name)` | Open one profile's box pool | name | **unsafe** |
| `llm.unload(name)` | Close it | name | **unsafe** |

`event.emit` is answered *before* delivery. `EventBus.emit` runs handlers on
the caller's thread, and a guest calling it may be holding its box's single
call lock — a service emitting from `poll` always is. So a subscriber that
called back into a service blocked on a lock the publisher could not release
until the emit was answered, which deadlocked the box permanently and then ate
one sandbox worker per later call into it. The kernel now queues the payload to
its own dispatcher thread and answers immediately. Nothing observable changes:
a guest never sees a subscriber, and `project()` already strips the round-trip
keys so a sandboxed subscriber cannot satisfy one.

**The last three are the model *authority*, not a call to a model.** They exist
because profiles stopped being services when `service_llm.py` was deleted, and
`/llm` went on asking the service registry about them — reporting every profile
uninstalled and unloaded while conversations resolved those same profiles
without trouble. `cron.*` fronting the timekeeper is the same shape: a kernel
subsystem a command needs to reach.

`llm.list` is safe for the reason `service.list` is — names, endpoints and
context sizes. The API key is a `secret_*` setting and comes back as a handle
like every other one. `llm.load`/`llm.unload` sit with `service.load`/
`service.unload` and for the same argument: opening a brain starts a pool of
real processes, each holding a provider SDK and a credential, and closing one
ends calls that may be in flight.

**Receiving is not a Request, and neither is standing at a doorway.** Both were
once going to be (`event.subscribe`, `hook.register`) and both became
*declarations* instead — `subscribed_channels = [...]` and
`hooks = {moment: method}`, read from the file without importing it. The
reasoning is the same for each: a subscription and a hook are standing
capability, and a Request that grants standing capability leaves something the
plugin holds and can forget to release. A declaration cannot leak, because the
plugin never registered anything — the kernel did, and the kernel undoes it at
unload. It is also visible at install time, which a runtime call never is.

`llm.proceed` is safe for an unusual reason: it is the only Request whose
handler is a **per-call closure**. The kernel parks an escort's `proceed` under
a one-shot token for exactly the duration of one `llm_call` visit. Code
holding no token reaches no call, so the limit is reachability rather than a
verdict, and "proceed" only ever means *place the call the kernel already
decided to make*.

`llm.delta` is scoped the same way, one call further in: the kernel parks a
sink for the duration of one backend `chat` call. It is also the only Request
sent **one-way** — a `notice` on the wire rather than a `request`, so the guest
does not block for an answer. That is not a hole in the gate: a notice is still
classified, recorded and executed by the same handler; the only thing given up
is knowing the outcome. It exists because a reply per token would turn a
stream into several hundred round trips, which would make streaming from a
subprocess slower than not streaming at all.

**There is deliberately no way to tell a backend to stop.** The old native
contract had one — `on_delta` returned a bool — and it does not survive the
boundary, nor should it: a rule that careless code can ignore is not a control.
Stopping is cancellation, which the kernel already owns. Cancel the execution
and the guest's next Request raises `Terminated`, a `BaseException` that a bare
`except Exception` cannot swallow.

**Note the latency cost.** Hooks fire inside the agent turn's hot path, and
`llm_call` wraps every model call. A hook on a subprocessed service pays IPC
on each fire. This is the strongest argument for validated in-process execution
being the default for services, with subprocess as opt-in. Bus deliveries have
the same shape but a worse failure mode: handlers run on the thread that
*emitted*, so a slow subscriber slows the publisher down.

## 13a. Frontends

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `frontend.submit(session_key, input_kind, ...)` | Carry a person's input into the state machine | token, kind | safe |
| `frontend.cancel(session_key)` | Stop what a session is doing | token | safe |
| `frontend.bind(session_key, external_id, ...)` | Say whose data a session is | token, external_id | safe |
| `frontend.attend(session_key, present)` | Say whether a person is watching | token | safe |
| `frontend.pending(session_key, details=False)` | Whether an approval is still waiting; with `details`, the pending approval or form step itself | token | safe |
| `frontend.resolve(session_key, value, request_id)` | Answer a pending approval | token, request_id | safe |
| `frontend.act(session_key, request_type, args)` | Run one Request as one of your sessions | token, session ownership | safe |
| `frontend.collect(handle)` | Take that Request's answer | token, owner | safe |

These are the same shape as `llm.proceed`: **scoped by reachability, not by a
verdict.** When a frontend's box opens, its native adapter is parked under a
token that is handed into the box; every one of these carries it back and
resolves to *that adapter and no other*. A tool, a service, or a script that
imported the same namespace holds no token, reaches no adapter, and is refused.
The desk is cleared when the frontend stops, so a leaked token then reaches
nothing.

They are `safe` because carrying a person's input into the state machine is the
entire job of a frontend — a dialog per keystroke would be nonsense. The one
that deserves argument is `frontend.bind`, which touches identity and might
look like it belongs with `user.write`. It does not: asking a user to approve
their own login would make a `per_user` frontend unusable, and a native
frontend already binds sessions freely. Which of the two native paths runs is
decided by whether an `external_id` was named rather than by the plugin picking
a method, so a frontend cannot upgrade a session to an arbitrary user.

Authentication is deliberately outside this boundary. The kernel stores
`password_hash` opaquely and ships no crypto; a frontend that binds a session
is asserting it did the work, and the kernel takes its word. `user_type` is
frontend-defined metadata, never a kernel admin bypass.

### `frontend.act`, and why it is safe

`act` looks like the widest thing in this document — it runs *any* Request —
and buys no authority at all on its own. The Request it carries goes through
the same gate and the same `classify` as one made from anywhere else. The only
thing it changes is the **chain**, from `frontend:<name>` to the session.

That matters because `frontend:<name>` names no session, so `attended_now`
answers False for it forever: a frontend's own Requests are unattended, and
anything unsafe is refused rather than asked. Correct for a frontend acting on
its own initiative — a poll tick nobody caused — and wrong for one serving a
request a person just made. Rooted at the session, `attended_now` instead asks
`runtime.is_attended`, which reads what *this same frontend* declared through
`frontend.attend`. So the grant is exactly:

> a frontend may act as a session it owns, while it says somebody is watching.

Declare the session unattended and the authority is gone. That self-limiting
property is the argument for rooting at the session rather than at `user`,
which would be unconditionally attended and would take the decision away from
the mechanism built to hold it.

Three things are host-side and cannot be stated by the guest. **Ownership**:
the token says which frontend is asking, and the runtime's session tags say
which sessions it may speak about — a session belonging to another frontend is
refused, and so is `frontend.attend` on one, which was previously unchecked.
**Identity**: for an inner `frontend.*` Request the kernel supplies the token
itself, so a caller cannot claim to be somebody else. **Reach**: `act` refuses
itself, `frontend.collect`, and the whole `http.*` family, which belongs to the
frontend's transport rather than to any session.

It is also **detached**, which is a correctness requirement rather than a
performance one. A box serves one call at a time and an approval dialog renders
back *into the calling box* to be seen, so answering inline deadlocks until the
dialog expires — the same failure `handlers.kernel._drive` exists to prevent.
The answer is collected later by handle, one-shot, swept if nobody comes back.

### The console

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `console.read()` | Take the next line typed at this machine | token, claim | safe |
| `console.write(text, end)` | Put text on the console | token, claim | safe |

Scoped harder than the rest of the family: not merely "a frontend", but **the
one frontend holding the claim**. A frontend declares `uses_console = True`,
the kernel lends the console to the first claimant and refuses the second —
two readers would split a person's keystrokes between them non-deterministically,
which presents as the machine dropping characters. Release names the token, so
a frontend that already lost the claim cannot revoke its successor's.

Safe because neither reaches past the console: reading takes only what a person
already typed at this machine's own keyboard, and writing puts text on the
screen in front of them. Gating them would mean asking permission to draw the
prompt that asks permission.

`input()` stays refused. It blocks (holding the box, so the frontend cannot
render), a subprocess box's stdin is the wire protocol (reading it corrupts the
transport), and a rule that works in-process and breaks under isolation is
worse than no rule. Inverting it — kernel reads, guest drains — removes all
three, and lets a console frontend be subprocess-isolated, which `input()`
never could.

### The port

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `http.drain(limit)` | Take the requests that have arrived | token, claim | safe |
| `http.respond(request_id, status, headers, body, stream)` | Answer one, or open a stream | token, claim | safe |
| `http.push(request_id, data, event, ident)` | Write one SSE frame | token, claim | safe |
| `http.close(request_id)` | End a reply | token, claim | safe |

The console's inversion one layer out, for the same reason and with the same
scoping. `socket` and `http` are refused to a guest and `sdk.net.http` dials
*out*, so a frontend could talk to the world and never be talked to — fine for
a transport that polls somebody else's servers, impossible for one a client
connects to. A frontend declares `serves_http = <port>` (a default; the config
key `<name>_port` overrides it), the kernel binds it on **loopback**, parses
with `http.server`, and the guest drains what arrived. Exposing that port is a
tunnel's job, not a declaration's.

The guest sets its own response headers and the kernel only fills gaps —
`Content-Length`, `Connection`, and the SSE content type. Notably **no CORS**:
which origins may reach a frontend is a fact about a deployment, not something
the boundary can decide, and a kernel-chosen `Access-Control-Allow-Origin`
would be either uselessly strict or a hole nobody asked for.

Four rather than two because **a reply may outlive the call that opened it**.
An SSE stream stays open for a whole conversation and takes frames one at a
time, which no answer-and-return Request expresses — the same argument
`proc.start` makes for not folding itself into `proc.run`. The connection stays
host-side and the guest holds only an id, so holding one is enough to answer
and only enough to answer: the split `project_approval` already makes.

Safe on reachability rather than on a verdict. Binding is the kernel's act and
not a Request, so none of the four can open a port; each reaches only the
socket the kernel already opened for the frontend that claimed it. Gating them
would mean a dialog per SSE frame. Note this is **not** an inbound `net.http` —
that Request dials out and is classified on where it is dialling, and there is
no destination to classify here because the client came to us.

`http.push` is in `prompt_cues.RENDERING` and dropped by the ledger's sandbox sink,
both for the reason `llm.delta` is: an SSE frontend sends one per token, so
counting it would undo every cached `agent_prompt` and recording it would write
a row per token. `http.drain` is a read; the other two are per-request and stay
recorded.

Rendering has no Request at all — the kernel calls `render` on the frontend.
An `approval` crosses as a question (`id`, `title`, `body`, `type`, `enum`,
`default`) and never as the decision: the pending action and the live
`threading.Event` the state machine waits on stay kernel-side, so holding the
id is enough to answer and only enough to answer.

## 14. Pipeline and tasks

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `task.enqueue(name, paths)` | Queue work | task name | safe |
| `task.status(name, path)` | Check state | — | safe |
| `task.output(name, path=None)` | Read task output | — | safe |
| `task.list(details=False)` | Inspect registered tasks | — | safe |
| `task.graph()` | Render the dependency pipeline | — | safe |
| `task.pause(name, paused=True)` | Pause or resume a task | task name, desired state | pausing safe; resuming asks |
| `task.reset(name, failed_only=False)` | Reset task state | task name, reset scope | asks |
| `task.trigger(name, payload=None)` | Manually enqueue an event task | task name, schema-filtered payload | safe |
| `task.output(name, filters)` | Read a task's output table | table | safe |
| `task.reset(name, scope)` | Re-run, clear state | scope | unsafe |
| `file.register(path, meta)` | Add to the watched-file table | path | safe |
| `file.unregister(path)` | Remove from the table | path | safe |
| `file.list(filters)` | Query the file registry | — | safe |

## 15. Parsing

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `parse.file(path, modality)` | Parse to text via the registry | path | safe |
| `parse.modality(ext, detail=False)` | Resolve a file's modality; `detail` also reports whether a specialist parser owns it | — | safe |

### 15a. LLM backends

A backend is not a plugin and makes no Request of its own beyond
`fs.read_bytes` (for attachments) and `llm.delta` (when streaming). It is
listed here because of what it *holds*.

**`request.api_key` is plaintext, and that is a deliberate exception.**
Everywhere else a credential is a `<secret:...>` handle the kernel substitutes
inside `net.http`, so plugin code uses a credential it never has. That works
only because the kernel makes the call. A provider SDK opens its own socket,
so there is no outbound Request to substitute into, and the key must be inside
the box to be usable at all.

What remains is still worth having: the key lives in a separate process whose
only route to the world is Requests the kernel classifies, and whose code was
validated before it ran. What is *not* claimed is that the key is protected
from the backend itself — it isn't, and closing that would need real OS
containment rather than a linter. A backend that can reach its provider over
plain HTTP should use `sdk.net.http` and keep the handle.

Capability declarations (`supports_streaming`, `supports_tool_choice`,
`native_modalities`) are read from the file by AST, never by importing it —
so asking what a backend can do never costs a provider-library import.

## 16. Ledger

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `ledger.record(action, ok, data)` | Write an audit row | — | safe |
| `ledger.read(limit, conversation_id, origin, session_key, action_types, since_id)` | Targeted query | conversation ownership | safe; a conversation the user does not own is **refused** |

Every Request that reaches the kernel is itself a ledger row, with its chain of
provenance as a column. `ledger.record` exists for plugin-level events that are
not Requests.

## 16a. Notifications

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `notification.list(limit, since_id, unread_only)` | Read this user's notifications | — (scoped to `ctx.user_id` in SQL) | safe |
| `notification.mark_read(ids, before_id)` | Settle notifications | — (scoped to `ctx.user_id` in SQL) | safe |

Raising one is `session.push(notify=True)`, not a Request of its own: pushing
text and raising a notification are the same act aimed at a different surface,
and growing an argument is cheaper than growing the vocabulary. Reading them
back is not the same act, and there is no existing Request whose subject is
notifications — the same standing `ledger.read` has.

Neither takes a `user_id`, which is a stronger arrangement than checking one.
`_check_access` exists because `conv.read` must *name* a conversation; nothing
here has to name anybody, so there is no argument to refuse and none to get
wrong. `mark_read` narrows by user inside the same `UPDATE`, so naming another
user's row changes nothing rather than being refused — there is no information
in the difference, and the returned count already says what happened.

**`source` is never stated by the caller.** The handler reads the leaf of the
live provenance chain (`_notification_source`), exactly as
`approval.describe_asker` does and for the same reason the ledger takes
`actor_id` from the chain: a plugin naming its own source could claim to be the
plugin watcher, and attribution is what a reader leans on to decide whether to
care. It is the part of a chain a box cannot state about itself.

Both carry the asking session's `session_key`, `conversation_id` and `user_id`,
so a row says *whose* work it was and not only what happened. That is what makes
`conversation_id` a usable filter — it seeks `idx_ledger_conv` — and what lets
`my_action_ledger` show a plugin the rows describing its own effects.

Every filter narrows in SQL rather than in the caller. The guidance to read this
table targeted rather than linearly is only actionable if there is something to
target with: the ledger is write-optimized filler by volume, so an unfiltered
read scans the whole flight recorder. `since_id` is the incremental form, for a
reader that already holds rows up to *N*.

Naming another user's conversation is refused rather than asked about, matching
`conv.read` — ownership is not a thing an approval dialog should be able to
grant. Filtering alone raises no dialog: this only ever reads.

The four filesystem Requests (`fs.write`, `fs.write_bytes`, `fs.delete`,
`fs.move`) additionally copy their path arguments into `data_json.paths`, and a
successful write its byte count into `data_json.bytes`. A `proc.run` /
`proc.start` that exited zero does the same for the paths its command line
names (`shell.files_touched`), tagged `data_json.via: "shell"` because a path
read out of a command line is a weaker claim than one the kernel serviced, with
`data_json.deleted` for the subset removed.

That extractor is **display only and must stay so** — nothing in `classify` or
either recognizer may read it, pinned by `tests/test_shell_files.py`. It is a
table of command names, which is what the dead classifier was, and the only
thing separating them is the question asked: that one decided *safety*, where a
wrong "safe" is silent and grants something, while this decides *what to draw*,
where a miss is an absent row and a false positive is a file shown that did not
change. It abstains on an unlisted program, a glob, a redirect, substitution, a
subshell, and any command that failed. `args_json` is capped and
past the cap the *object* is replaced by a `head`/`tail` wrapper — and the
argument that blows the cap is the file's own contents, so the rows whose paths
are hardest to recover would otherwise be exactly the largest edits.

## 17. Network

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `net.http(url, method, headers, body, params, json)` | Any outbound HTTP | the URL's host | safe for a host in `net_allowed_hosts`, otherwise **unsafe** |

One Request, one gate. The verb is irrelevant: a `GET` with data in the query
string is exfiltration exactly as much as a `POST` body is, so only the
destination is consulted. This is the single control that makes free filesystem
and database reads safe.

**The one thing that relaxes it is a host allowlist the user maintains** — the
kernel config setting `net_allowed_hosts`, empty by default. A bare domain also
covers its subdomains, matched on a dot boundary so `example.com` covers
`api.example.com` and not `notexample.com`. Path and query are not matched: the
host is what a person can usefully decide about once, where "may this plugin
fetch /v1/search?q=…" has no stable answer.

It is deliberately **not** a declaration on the plugin. An `endpoints = [...]`
line read off the source would make the code being contained the authority on
its own reach — the same bug `isolation = "subprocess"` had, one level down, and
an agent authoring a plugin would author its own egress by typing a hostname. A
person deciding what the app may talk to is a different act.

`policy._NET_RECOGNIZERS` exists beside the allowlist for the reason the shell
has recognizers: somewhere for a remembered or structural allowance to live
later, visible in the policy rather than inside the plugin it authorizes. It
ships empty.

Two details that fail closed. The host is parsed from the URL *before* secret
substitution, so a `<secret:…>` standing where a hostname goes resolves to no
recognisable host and is asked about rather than allowed. And an unparseable URL
yields the empty host, which no allowlist entry can equal.

**The answer includes error statuses.** `net.http` returns
`{status, body, headers}`, and an HTTP error status arrives that way too rather
than collapsing into a failure — a 429's body is where an API says which limit
and for how long, and a caller that gets `http 429` and nothing else cannot act
on it. Only a request that got no reply at all (DNS, refused, timed out) is a
failure. The body is UTF-8-decoded with replacement, so this Request answers
about text and text only; binary egress is absent by design, since the things
wanting it are foreign libraries doing their own I/O inside their own box.

**Redirects are answers, never implicit Requests.** The host HTTP client does
not follow a 3xx. It returns the status and `Location` header, and following it
requires another `net.http` Request. That makes a redirect to a different host
cross the same policy gate as any other outbound destination instead of
spending the original host's approval twice.

Secret handles are substituted here, on the way out, after the policy function
has already decided.

## 18. Process

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `proc.run(argv, timeout, cwd, shell)` | Run to completion | the command line | unsafe unless a recognizer covers it |
| `proc.start(argv, cwd, shell, label)` | Start something and keep it, return a handle | the command line | unsafe unless a recognizer covers it |
| `proc.status(id, tail)` | Ask after one, with the tail of its output | — | safe |
| `proc.stop(id)` | End one and forget it | — | safe |
| `proc.list()` | Everything still tracked | — | safe |

**Everything that starts a process is asked about unless a narrow recognizer
vouches for the exact invocation.**
The earlier plan here was to lift `tool_run_command`'s: decompose a compound
command at unquoted `&&`, `||`, `;`, `|` and newlines, auto-run only when every
segment matches a read-only whitelist, and send redirection, command
substitution, backgrounding and unbalanced quotes down the approval path. Five
hundred lines, modelled on Claude Code's and Codex's, and it worked — mostly.

"Mostly" is the objection. Deciding what an arbitrary command line *does* is
undecidable, so a classifier of that shape is a whitelist racing against
quoting forever, and it loses in the invisible direction: a wrong "unsafe"
gets reported as a bug, a wrong "safe" gets reported as nothing. It also sat
inside the plugin it authorized, which is the wrong place on principle —
authorization does not live in the code being authorized.

So the family defaults unsafe and the dialog is the fallback. Two recognizers
ship in `_SHELL_RECOGNIZERS`: a deliberately incomplete structural recognizer
for known read-only invocations, and remembered `(program, subcommand)` grants.
The structural recognizer includes conservative `ls` display operations and
`cat` only when every operand is an existing regular, non-protected file within
the same aggregate size cap as `fs.read`. Globs, devices, directories, stdin,
redirection, substitution, unsupported flags, and non-POSIX shell aliases
abstain. Adding coverage remains a deliberate policy widening.

The three read-and-narrow members are safe. `status` and `list` read a
registry the kernel owns, which holds nothing that was not approved at
`start`. `stop` is safe for the reason `session.remove_tool` is: it narrows.
A dev server the agent cannot kill without a dialog is a dev server the agent
will not start, and the alternative to stopping one is leaving it running.

The kernel builds the invocation from `shell` (`None` = exec the argv
directly, `"default"` = the platform shell, `"powershell"`, `"cmd"`) rather
than letting the guest wrap its own, because `cmd.exe` does not understand the
backslash-escaped quotes `subprocess` produces from a list — a guest passing
`["cmd", "/c", line]` would have every embedded quote silently mangled.

Building it host-side is also the only place the platform differences can be
kept honest, and there are three: `"default"` resolves to `cmd.exe` on Windows
and `/bin/sh` elsewhere; `"powershell"` is the Windows-only 5.1 binary on
Windows and PowerShell Core's `pwsh` everywhere else; `"cmd"` is refused by
name off Windows rather than failing as a missing executable. Ending a process
is asymmetric for a reason that matters — `taskkill /T /F` is already a hard
tree kill, while POSIX `SIGTERM` is a *request* that a server may trap, so
`stop` escalates to `SIGKILL` and reports `stopped: False` if even that loses.
Without the escalation `proc.stop` reported success on a process still running
and no longer tracked.

The registry is in-memory: a `Popen` handle is not serializable, so nothing
survives a restart. What survives is the log file. A process still running
when the app exits is orphaned rather than killed, which is why the agent
prompt is emphatic about stopping them.

## 18a. Scripts

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `script.run(path, entry, args, wait)` | Run a file of SDK code that is not a plugin | the directory the file is in; what it imports | safe when contained; unsafe for foreign imports |
| `script.collect(ids, timeout)` | Take detached scripts' results | — | safe |
| `script.stop(id)` | Cancel a detached script | — | safe |

**This is the shell's job, moved somewhere it can be answered.** Section 18
concludes that a command line cannot be classified, so every command is asked
about — which leaves the agent's *cheapest* capability also its most dangerous
one, and nothing safe to reach for instead. A script is that alternative. It is
Python over the SDK, so there is nothing to interpret: every effect inside it
arrives at the gate on its own, individually, with the script still in the
chain. Running one therefore widens nothing, which is the same argument that
makes `tool.call` and `service.call` safe.

Two things are checked, and both are read off the *destination* — the same
shape as the filesystem branches, which ask where a write is aimed rather than
what it contains.

**Where the file is.** Only `<tree>/scripts/*.py`, top level, at any of the
three tree roots. The directory is the whole declaration: a script has no
family prefix, no base class and no entry point, so there is nothing else about
the file that could say what it is. A missing, unreadable, misplaced, or invalid
file reaches launch preflight and returns an ordinary actionable failure; it is
not presented as an approval question or a lockdown denial.

**What it imports.** A script whose imports the validator cannot see inside is
**unsafe**, and the dialog names the library. This is deliberately stricter
than the equivalent rule for plugins: an installed package importing a foreign
library is subprocessed and *not* asked, because a person approved it once at
`plugin.install`. A script was never approved by anybody, and a foreign
library's own actions are the one part of a script that does not come back
through this function — so this is the only moment there is to ask.

**Scripts are always subprocessed**, wherever they live, which is the one place
`required_isolation` does not consult the per-tree answer. An installed plugin
that is pure computation over the SDK earns in-process execution because it is
a declared, registered, reviewed capability; a script is none of those things.

The verdict is re-derived by the kernel from the path, never supplied on the
Request. A caller passing its own report — or a digest standing in for one —
would be the code being contained acting as the authority on its own
containment, which is the bug `sandbox/isolation.py` exists to prevent.
The handler revalidates immediately before launch. If the current bytes gained
a foreign import, execution proceeds only when this `script.run` Request was
actually approved; lockdown therefore refuses it while ask/YOLO approval is
honoured. Every SDK Request made after launch remains independently gated.

Ephemeral only. A script that wants to stay resident is a service, and a
script that wants to run for an hour is a pipeline task; both are families that
already exist and both are approved at the point they are created, which is
where a commitment that outlives a turn should be answered for.

**`collect` and `stop` speak only about a script this caller already started**,
which is why neither needs the two checks above: `run` answered them when the
work began, and neither of these starts anything. Collecting reads a result
already produced. Stopping *narrows*, and is safe for the reason `proc.stop`
and `agent.stop` are — a fan-out the caller cannot abandon without a dialog is
one it will not start.

Ownership is the caller's chain **root**, so two scripts detached by one turn
are collectable together and a different root reaches neither. The root is
also the part of a chain a guest cannot state about itself, so a box cannot
claim somebody else's results by asking for them. An id belonging to another
owner answers "no such run" rather than refusing — that names the caller's
actual mistake without disclosing that the run exists.

`script.collect` is in `READ_ONLY`, so the ledger's sandbox sink drops it: a
fan-out polls with `timeout=0` and would otherwise write a row per tick, the
same problem `console.read` and `agent.collect` are in that set for. Delivery
is one-shot, and a finished result nobody collects is swept after
`COLLECT_RETENTION` so an abandoned fan-out is not a leak.

## 19. Self and ambient

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `self.respond(result)` | Return a result and terminate | — | safe |
| `self.terminate()` | End without a result | — | safe |
| `self.yield()` | Persistent containers: sleep until next input | — | safe |
| `self.budget()` | How much of this execution's deadline is left | — | safe |
| `env.read(name)` | Environment variable | name sensitivity | safe, **credential-looking names returned as handles** |
| `secret.reveal(name)` | A credential in plaintext | — | **unsafe, always** |

`time.now()` is SDK, not a Request — determinism is not a goal.

`self.respond` is invalid for persistent containers, which use `self.yield`.

`self.budget` reads a clock the kernel keeps **about the caller itself** — no
other execution is visible through it — and the answer only ever causes the
guest to do less, so there is nothing here for a dialog to be about. It is
`READ_ONLY` for the reason `script.collect` is, and more sharply: a long loop
asks every iteration, so counting it as a change would bump the `prompt_cues`
write rung per
tick and silently invalidate every `write`-cued `agent_prompt` in the process.

It exists because the kernel is the only party that can answer. The guest may
read a clock, but a deadline measures *running* time — elapsed minus whatever
the kernel spent owing it an answer — and it can see neither that discount nor
the ceiling its declared `timeout` was clamped to. Without the Request the only
way to discover a deadline is to be killed by it, which discards everything
computed on the way; with it, a run that is going to be too long can hand back
what it has.

## 20. The application

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `app.stop(restart=False)` | End the process, optionally starting it again | — | **unsafe, always** |

One type with an argument rather than `app.quit` plus `app.restart`: stopping
and stopping-then-starting are the same act with a different tail. Unsafe in
both forms — coming back up is not a mitigation, since everything in flight
still dies either way.

This is what `/quit` and `/restart` are, and it is the reason they can be
ordinary sandboxed commands rather than native ones built in the composition
root. The kernel owns the delay: the answer reaches the frontend first, then the
process goes away, because otherwise nobody is told why it ended.

---

## The return contract

Every Request returns simple data — never a live object. A dataclass is
converted to a dictionary before crossing, and the SDK rebuilds it on the far
side from the same class definition, so plugin code keeps attribute access
without a kernel object ever crossing the boundary.

Three outcomes, one shape:

| Outcome | Returns |
|---|---|
| Success | The requested data, or a success report for action Requests |
| Failure | A failure report — reason, and whether retrying could help |
| **Denial** | A failure report with reason `denied` |

**A denied Request is an ordinary failure, not an exception and not a kill.**
The plugin is resumed and may handle it, retry differently, or terminate. This
is deliberate: one error path is learnable, two are not, and code that treats
denial as fatal is the most likely thing a careless author will write.

---

## Three supporting mechanisms

**Secret handles.** A config setting holding a credential is *named*
`secret_something`; that prefix is the declaration, matching how the rest of
the system declares things by name. A Request that would return one returns an
opaque handle instead — `<secret:secret_brave_api_key>`. The
handle can be passed into other Requests; the kernel substitutes the real value
at the point of use, inside `net.http`. Sandboxed code can therefore *use* a
credential it can never *read*, which is exactly the property a careless plugin
needs. This is what the contract's "private information" clause means in
practice.

**The limit of that, stated plainly.** Substitution is possible because the
kernel performs the effect. A plugin driving a foreign library that does its
own network I/O — an OAuth client, a provider SDK — offers no such moment, and
no amount of handle machinery changes that. It is the same limit the contract
already names for foreign libraries, not a second one.

So plaintext is reachable, through one Request classified **unsafe**:
`secret.reveal`. The ledger keeps the record either way, and the dialog names
the secret, the plugin, and the chain.

It is not asked every time, because that would be noise rather than security.
A plugin declares its `config_settings`, so the kernel knows which plugin a key
belongs to: **the owner reads its own credential silently — configuring the key
for that service was the consent — and any other plugin reaching for the same
key is asked.** That is the question with actual information in it.

Which leaves the honest ceiling: a credential handed to a foreign library is
past the kernel's reach, and no arrangement of Requests recovers it. Only real
OS-level containment would, and until then this is a known, accepted cost of
running useful code.

**Per-user views.** Tables carrying a `user_id` are exposed to sandboxed code as
views filtered to the session's current user, not as base tables. Scoping is
structural: the plugin cannot forget to filter, and cannot misreport its
identity, because the kernel bound the view. Free-form SQL is preserved.

**The provenance root.** A chain does not begin at a plugin. It begins at the
thing that caused the work — a user turn, a cron job, a subagent, a frontend
event — and the root is what makes a permission dialog answerable. `cron:nightly_index
→ task_index → net.http` tells the user everything; `task_index → net.http`
tells them nothing.

The kernel maintains the chain as its own stack: push when it begins driving a
plugin, pop when that plugin terminates. Plugins never report their own
identity, so they cannot misstate it. A persistent container, being in no one's
call stack, carries the chain captured at its creation. The stack is capped for
depth, which also detects cycles.

An approved command carries that one-shot approval on its host-maintained
chain, and the authority disappears when the command returns. The approval
token is generated and consumed by the state machine, never accepted from
plugin arguments.

**The approval is scoped, not a skeleton key.** `Chain.approved` is a *set of
Request types* — the command's own `requests` declaration, read by AST at
adapt time — never a boolean. Requests inside that set do not ask a second
time. Requests outside it fall through to their ordinary branch and are asked
about on their own, so a command reaching past its manifest is caught rather
than riding in on the one "yes" the user already gave. `push` copies the grant
down unchanged: a callee can spend what the approved command was given and can
never widen it by declaring more itself.

The grant is the declaration rather than an argument allowlist because that is
the only question with a decidable answer. Predicting what `git pull` will do
is Rice's theorem; asking whether the command declared that it runs a shell is
a set membership test. It is also what the user is actually being asked — a
person approves a *command*, and the honest statement of a command's scope is
the capability classes it declared. `requests` is therefore load-bearing, and
the validator checks every name against the closed Request vocabulary: a
misspelling would otherwise grant nothing and surface as a dialog the user
thought they had already answered.

**The dialog states the scope.** A grant nobody is shown is not consent, so
the prompt is rendered from the declaration rather than the command name:

```
/update wants to:
  - run shell commands
  - look up application folders
```

`approval.describe_grant` builds it, from the same `requests` list the bridge
turns into the grant, so the question asked and the authority handed over
cannot drift apart. Phrases are the type-level counterpart to `_action_line`:
that renders one concrete effect as it happens, this summarises a capability
class before anything runs. The table is **total** over Request types and
tested to stay that way — a new Request with no phrase would render as a
dotted name, which is a question nobody can answer. Read-only members of
write-shaped families (`plugin.list`, `conv.read`) carry their own phrases,
because overstating a grant erodes trust in the dialog as fast as
understating it erodes safety. A command declaring nothing falls back to the
bare `Approve /x?` — a command that performs no consequential effect.
