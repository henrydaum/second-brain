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
`fs.write` to the scratch directory is safe; the same Request aimed at
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

## 1. Filesystem

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `fs.read(path)` | File contents as text | path | safe |
| `fs.write(path, data, mode)` | Create, overwrite, or append text | path | safe in scratch/memory/sandbox, else unsafe |
| `fs.read_bytes(path)` | File contents as raw bytes | path | safe |
| `fs.write_bytes(path, data, mode)` | Create, overwrite, or append bytes | path | safe in scratch/memory/sandbox, else unsafe |
| `fs.list(path, pattern)` | Directory listing, glob, stat | path | safe |
| `fs.search(pattern, root)` | Content search across a tree | root path | safe |
| `fs.delete(path)` | Remove a file or tree | path | unsafe |
| `fs.move(src, dst)` | Copy, rename, replace | both paths | unsafe outside scratch |
| `fs.temp()` | Allocate a scratch file or directory | — | safe, always |

`fs.temp` exists so that "I need somewhere to put this" never requires a policy
decision. Scratch space is granted, not requested by path.

`fs.search` is derivable from `fs.list` + `fs.read`, and is a separate Request
anyway: doing it by hand costs one round trip per file.

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
| `db.query(sql, params)` | Read rows | resolved tables/columns, user | safe |
| `db.write(sql, params)` | Insert, update, delete | resolved tables, user | unsafe on kernel tables, safe on plugin-owned |
| `db.define(ddl)` | Create or alter a plugin-owned table | table name | safe for new tables, unsafe to alter kernel tables |

Reads stay deliberately unrestricted. Free reads are safe **because the exits
are gated** — a plugin that reads everything still cannot send anything
anywhere without passing `net.http`, which is always checked. Do not trade away
the agent's reach over its own database to solve a problem that egress control
already solves.

Two narrow exceptions, both structural rather than policy:

- `users.password_hash` is denied at the column level. It is the only secret
  column in the schema.
- User-scoped tables are reached through **per-user views**, not base tables.

Raw `db.conn` and `db.lock` access is withdrawn — every current use migrates to
these three Requests. Transaction scoping becomes an argument
(`db.write(..., atomic=[...])`), not a borrowed lock.

## 3. Conversations

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `conv.create(title)` | Start a conversation | user | safe |
| `conv.read(id)` | Messages and metadata | id, owning user | safe (own), unsafe (other user) |
| `conv.list(filters)` | Enumerate conversations | user | safe |
| `conv.append(id, message)` | Add a message | id, owning user | safe (own) |
| `conv.set_title(id, title)` | Retitle | id, owning user | safe (own) |
| `conv.set_category(id, cat)` | Categorize | id, owning user | safe (own) |
| `conv.set_notify(id, mode)` | Notification mode | id, owning user | safe (own) |
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
| `ui.ask(title, prompt, type)` | Question with typed answer (text/bool/choice) | attendance | safe if attended, refused if not |
| `ui.approve(action, justification)` | Explicit approval for a sensitive action | attendance | safe |
| `ui.render(paths, caption)` | Show files to the user in chat | paths | safe |

`ui.ask` is definitionally safe when a human is present — it *is* the approval
channel. In an unattended session it is refused rather than queued, matching the
kernel's existing default at the `unattended_call` gate.

## 6. Configuration

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `config.read(key, scope)` | Global or user-scoped setting | `secret_` prefix | safe, **`secret_*` returned as handles** |
| `config.write(key, value, scope)` | Change a setting | key, scope | unsafe |
| `paths.get(name)` | Resolve a named application location | fixed name allowlist | safe |

Scope (`global` / `user`) is an argument, not a separate Request.

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
| `plugin.register(path)` | Load a plugin live | path, family | unsafe |
| `plugin.unregister(name)` | Unload | name | unsafe |
| `plugin.reload(name)` | Reload in place | name | unsafe |
| `plugin.install(stem)` | Install from the store | stem, store commit | unsafe |
| `plugin.uninstall(stem)` | Remove, with dependency scan | stem | unsafe |
| `plugin.quarantine(name, reason)` | Disable a misbehaving plugin | name | safe |

This family is the literal subject of the LibOS quote: the agent extends itself
here, and every widening Request in it is unsafe by default. Quarantine is safe
because it only removes capability.

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
| `command.call(name, args)` | One-shot slash command | target command, `require_approval` | inherits the command's own declaration |

## 11. Agent

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `agent.complete(prompt, schema)` | A model call | — | safe |
| `agent.spawn(prompt, wait)` | Run a subagent now | root, depth | safe |
| `agent.schedule(prompt, cron)` | Run a subagent later | cron, root | unsafe |
| `agent.escalate(reason)` | Re-drive on the strong model | — | safe |

`agent.complete` is its own Request and never a generic `service.call`. Keys,
sockets, and provider details stay kernel-side; the sandbox sees a prompt and a
schema.

`agent.spawn` is safe because the child's own Requests are gated with the parent
in the chain — you cannot buy authority by having someone else ask.
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
| `model.proceed(request)` | Place the call an escort is holding | token | safe |
| `model.delta(text)` | Push streamed assistant text out of a backend | token | safe |

**Receiving is not a Request, and neither is standing at a doorway.** Both were
once going to be (`event.subscribe`, `hook.register`) and both became
*declarations* instead — `subscribed_channels = [...]` and
`hooks = {moment: method}`, read from the file without importing it. The
reasoning is the same for each: a subscription and a hook are standing
capability, and a Request that grants standing capability leaves something the
plugin holds and can forget to release. A declaration cannot leak, because the
plugin never registered anything — the kernel did, and the kernel undoes it at
unload. It is also visible at install time, which a runtime call never is.

`model.proceed` is safe for an unusual reason: it is the only Request whose
handler is a **per-call closure**. The kernel parks an escort's `proceed` under
a one-shot token for exactly the duration of one `model_call` visit. Code
holding no token reaches no call, so the limit is reachability rather than a
verdict, and "proceed" only ever means *place the call the kernel already
decided to make*.

`model.delta` is scoped the same way, one call further in: the kernel parks a
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
`model_call` wraps every model call. A hook on a subprocessed service pays IPC
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

These are the same shape as `model.proceed`: **scoped by reachability, not by a
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
`fs.read_bytes` (for attachments) and `model.delta` (when streaming). It is
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
| `net.http(method, url, headers, body)` | Any outbound HTTP | host, method, secret handles present | **always checked, never auto-safe** |

One Request, one gate. The verb is irrelevant: a `GET` with data in the query
string is exfiltration. This is the single control that makes free filesystem
and database reads safe, so it does not get exceptions.

Secret handles are substituted here, on the way out.

## 18. Process

| Request | Purpose | Policy inputs | Default |
|---|---|---|---|
| `proc.run(argv, timeout)` | Run to completion | argv classification | safe if every segment is read-only, else unsafe |
| `proc.start(argv)` | Start a persistent process, return a handle | argv | unsafe |
| `proc.stop(handle)` | Terminate | handle | safe |

The read-only classifier already exists and is battle-tested — `tool_run_command`
decomposes compound commands at unquoted `&&`, `||`, `;`, `|`, and newlines and
auto-runs only when every segment is read-only, sending redirection, command
substitution, backgrounding, and unbalanced quotes down the approval path. Lift
it wholesale rather than rewriting it.

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
