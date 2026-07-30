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
| `fs.write(path, data, mode)` | Create, overwrite, or append text | path | safe in scratch and the agent's plugin tree, else unsafe |
| `fs.read_bytes(path)` | File contents as raw bytes | path | safe, except protected files |
| `fs.write_bytes(path, data, mode)` | Create, overwrite, or append bytes | path | safe in scratch and the agent's plugin tree, else unsafe |
| `fs.list(path, pattern, details, recursive, files_only, sort, limit)` | Directory listing, glob, stat, pruning walk | path | safe |
| `fs.search(pattern, root, glob, regex, mode, ...)` | Content search across a tree | root path | safe; protected files skipped |
| `fs.delete(path)` | Remove a file or tree | path | unsafe |
| `fs.move(src, dst)` | Copy, rename, replace | both paths | unsafe outside scratch |
| `fs.temp()` | Allocate a scratch file or directory | — | safe, always |

`fs.temp` exists so that "I need somewhere to put this" never requires a policy
decision. Scratch space is granted, not requested by path.

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
without a dialog. Free authorship, unchanged authorization. The grant is also
scoped to that one tree — writing into `plugins/` or anywhere else is unsafe as
before, because containment does not apply there.

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

`fs.list` is also the **stat**: `details=True` adds `is_dir`, `size` and
`mtime` (`st_mtime_ns`, an int so it survives JSON exactly — compare with
`!=`, since a file restored to an older version has also changed). Pointed at
a *file* it answers for that one entry, which is how a plugin asks "does this
exist, and has it changed since I looked?" without a second Request type.
Routing that through "list the parent and filter" made callers build a glob
out of a filename, which breaks on any name containing `[` or `*`.

**Protected files** (`sandbox/protected.py`) are the one place a read Request
refuses on the path alone: `config.json`, `plugin_config.json`, and the SQLite
database with its sidecars. Both exist because a control enforced on one
Request was walkable around on the next. `config.read` hands back a
`<secret:…>` handle and `secret.reveal` prompts — but `config.json` holds the
same credentials in plaintext, so an unrestricted `fs.read` made the handle
decorative. The database is reachable through `db.query`, which scopes rows per
user and refuses `password_hash`; reading the file walks around all of it.
`fs.search` counts as a read here because its hits carry matching *lines* —
`pattern="secret_"` would do the job by itself. Writes need no rule: a write
outside scratch is already unsafe, so editing these asks like any other.
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
| `conv.read(id)` | Messages and metadata | id, owning user | safe (own), unsafe (other user) |
| `conv.list(filters)` | Enumerate conversations | user | safe |
| `conv.append(id, message)` | Add a message | id, owning user | safe (own) |
| `conv.set_title(id, title)` | Retitle | id, owning user | safe (own) |
| `conv.set_category(id, cat)` | Categorize | id, owning user | safe (own) |
| `conv.set_notification_mode(id, mode)` | Notification mode | id, owning user | safe (own) |
| `conv.load(id)` | Load saved state into the current session | id, owning user, session | safe (own) |
| `conv.clear(id)` | Drop messages, keep conversation | id, owning user | safe (own) |
| `conv.delete(id)` | Delete conversation and messages | id, owning user | unsafe |
| `conv.enact(id, action)` | Drive an agent turn | id, owning user, root | unsafe from an unattended root |

Ownership is checked on every id-bearing Request, mirroring
`runtime.assert_conversation_access`. Cross-user access is refused and recorded,
never silently filtered.

## 4. Sessions

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `session.get(key)` / `session.list()` | Inspect live sessions | — | safe |
| `session.open(key)` / `session.close(key)` | Session lifecycle | key | safe |
| `session.push(key, message)` | Proactive message to the user | key, attendance | safe |
| `session.state_get/set/clear(key, ns)` | Per-session plugin scratch state | namespace | safe |
| `session.set_attended(key, bool)` | Declare human presence | key | unsafe |
| `session.cancel(key)` | Cancel the running turn | key | safe |
| `session.add_attachment(key, path)` | Stage an attachment for the turn | path | safe |
| `session.set_profile(key, profile)` | Switch agent profile | profile | unsafe |
| `session.add_prompt_extra(key, text)` | Inject system prompt text | — | unsafe |
| `session.remove_prompt_extra(key, id)` | Withdraw injected text | — | safe |
| `session.add_tool(key, name)` | Widen the agent's scope | tool name | unsafe |
| `session.remove_tool(key, name)` | Narrow the agent's scope | tool name | safe |

The asymmetry is intentional and runs through the whole catalogue: **widening
capability is unsafe, narrowing it is safe.** Adding a tool, injecting prompt
text, or claiming attendance changes what the agent may do next; the reverse
never does.

## 5. User interaction

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `ui.ask(prompt, title, type, choices, required, default)` | Question with typed answer | attendance | safe if attended, refused if not |
| `ui.approve(action, justification)` | Explicit approval for a sensitive action | attendance | safe |
| `ui.render(paths, caption)` | Show files to the user in chat | paths | safe |

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
| `command.call(name, args)` | One-shot slash command | target command name | unsafe; **refused** if the command needs approval |

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
| `frontend.pending(session_key)` | Whether an approval is still waiting | token | safe |
| `frontend.resolve(session_key, value, request_id)` | Answer a pending approval | token, request_id | safe |

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
| `parse.modality(ext)` | Resolve a file's modality | — | safe |

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
| `ledger.record(...)` | Write an audit row | — | safe |
| `ledger.read(filters)` | Targeted query | user, conversation | safe (own), unsafe (other users) |

Every Request that reaches the kernel is itself a ledger row, with its chain of
provenance as a column. `ledger.record` exists for plugin-level events that are
not Requests.

## 17. Network

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `net.http(url, method, headers, body)` | Any outbound HTTP | the URL's host | safe for a host in `net_allowed_hosts`, otherwise **unsafe** |

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

Secret handles are substituted here, on the way out, after the policy function
has already decided.

## 18. Process

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `proc.run(argv, timeout, cwd, shell)` | Run to completion | the command line | **unsafe** |
| `proc.start(argv, cwd, shell, label)` | Start something and keep it, return a handle | the command line | **unsafe** |
| `proc.status(id, tail)` | Ask after one, with the tail of its output | — | safe |
| `proc.stop(id)` | End one and forget it | — | safe |
| `proc.list()` | Everything still tracked | — | safe |

**Everything that starts a process is asked about, and there is no classifier.**
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

So the whole family is unsafe and the dialog is the control. That is annoying
rather than wrong, and annoying is the failure mode to prefer. Where it gets
better is `_SHELL_RECOGNIZERS` in `policy.py`: a recognizer reads the rendered
command line and returns a reason to allow it, or `None` to abstain. Two kinds
are expected — *structural* ("every segment of this pipeline is a read-only
command", the old classifier rebuilt where the policy can see it) and
*remembered* ("the user already approved exactly this, at this scope"), the
second being the more useful. The list is empty today, and adding to it is a
deliberate widening.

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
| `script.run(path, entry, args, wait)` | Run a file of SDK code that is not a plugin | the directory the file is in; what it imports | safe |

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
the file that could say what it is. A path anywhere else is refused rather than
asked about, because the containment story rests entirely on the answer.

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

Ephemeral only. A script that wants to stay resident is a service, and a
script that wants to run for an hour is a pipeline task; both are families that
already exist and both are approved at the point they are created, which is
where a commitment that outlives a turn should be answered for.

## 19. Self and ambient

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `self.respond(result)` | Return a result and terminate | — | safe |
| `self.terminate()` | End without a result | — | safe |
| `self.yield()` | Persistent containers: sleep until next input | — | safe |
| `env.read(name)` | Environment variable | name sensitivity | safe, **credential-looking names returned as handles** |
| `secret.reveal(name)` | A credential in plaintext | — | **unsafe, always** |

`time.now()` is SDK, not a Request — determinism is not a goal.

`self.respond` is invalid for persistent containers, which use `self.yield`.

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
bare `Approve /x?`, which is every unmigrated native command.
