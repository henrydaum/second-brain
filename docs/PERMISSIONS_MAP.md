# The Permission Map

**Short answer to "do we have a clear map of these?" — there wasn't one.**
Seven layers can stop or gate an action, spread across `runtime/agent_scope.py`,
`sandbox/isolation.py`, `sandbox/users.py`, `sandbox/policy.py`,
`sandbox/approval.py` and `plugins/native/command.py`. Each is documented well
individually; none of the docs showed the *order*.

This file is the missing overview. `docs/SECURITY_CONTRACT_APPENDIX.md` remains
the per-Request catalogue — this is the machinery around it.

---

## 1. The pipeline, in order

An action passes through these in sequence. Anything can stop it; only layer 5
can *ask*.

| # | Layer | Decides | Where | Can ask? |
|---|---|---|---|---|
| 0 | **Scope** | whether the capability exists at all | `runtime/agent_scope.py`, `plugins/command_registry.py` | no |
| 1 | **Isolation** | which process it runs in | `sandbox/isolation.py` | no |
| 2 | **Structural refusal** | hard no, never a dialog | `sandbox/users.py`, `validator`, token checks | no |
| 3 | **Grant short-circuit** | already approved as part of a command | `policy.classify` (first branch) | n/a |
| 4 | **`classify()`** | SAFE (execute) or UNSAFE (ask) | `sandbox/policy.py` | n/a |
| 5 | **The approver** | allow / refuse / dialog | `sandbox/approval.py` | **yes** — the only doorway |
| 6 | **State-machine approval** | gates a command *before* its body runs | `plugins/native/command.py`, `state_machine/action.py` | **yes** |

The **mode** (`/mode lockdown|ask|yolo`) is not a layer of its own: it is a
standing answer *inside* layer 5, and — for `yolo` only — inside layer 6. See
§6a.

Layer 6 runs *earlier in wall-clock time* than 3–5 but is listed last because
it is a parallel system: it produces the grant that layer 3 consumes.

---

## 2. Layer 0 — Scope

Not permission, but it is the first thing that makes an action impossible, and
it is the only layer that acts by *absence* — the agent never learns the
capability was there.

| Mechanism | Unit | Modes | Set by |
|---|---|---|---|
| `agent_profiles[p].tools_list` | tool name | whitelist / blacklist | config, per user (`active_agent_profile`) |
| `frontend_profiles[f].commands_list` | command name | whitelist / blacklist | config, per frontend |
| `shape_scope` hook | tool name | **narrow only** | any service, per session |
| `enabled_frontends` | frontend name | list | config |

`shape_scope` can only hide. Widening is a separate act — `sdk.session.add_tool`,
which is `ALWAYS_UNSAFE`.

## 3. Layer 1 — Isolation

Decided by **provenance, never declaration**. Listed here because it is what
makes the rest affordable, not because it asks anything.

| Tree | Isolation |
|---|---|
| `bundled/` | in-process |
| `workspace/` | subprocess, always |
| `installed/` | subprocess iff the validator sees an unmediated import |
| `scripts/` (any tree) | subprocess, always |

Two things tighten it that a file cannot lie about: a declared
`parse_modalities`, and a declared helper's own imports (`_imports_foreign_code`).

## 4. Layer 2 — Structural refusals

These never produce a dialog. There is no "allow" answer.

| Refusal | Rule |
|---|---|
| `sandbox/users.py` | DDL naming a kernel table; `password_hash`; `sqlite_master`/`PRAGMA`; cross-user reads |
| `MAX_DEPTH = 8` | chain deeper than 8 |
| cycle detector | any name twice in the chain |
| unclassified type | a new Request with no decision is refused |
| desk token | `frontend.*` from code that is not that frontend |
| one-shot token | `llm.proceed` / `llm.delta` outside the call it belongs to |
| console claim | second frontend claiming `uses_console` |
| caps | `DB_MAX_ROWS` 500, 16 MB message |
| validator | a plugin importing a kernel module simply does not load |

## 5. Layers 3–4 — `classify()`

111 Request types. The partition:

| Set | Count | Meaning |
|---|---|---|
| `ALWAYS_SAFE` only | 71 | narrows capability, or affects only this execution |
| `ALWAYS_UNSAFE` only | 16 | changes state the kernel owns, whatever the arguments |
| argument-conditional branch | 18 | the interesting ones |
| in a set *and* branched | 6 | `fs.temp`, `fs.delete`, `conv.delete`, `agent.schedule`, `session.add_tool`, `session.add_prompt_extra` |

Enforced at import: `_UNDECIDED` must be empty, so a new Request cannot be added
without somebody deciding about it.

### The grant short-circuit (layer 3)

```python
if chain.approved and kind in chain.approved:
    return Decision(SAFE, "approved command")
```

`chain.approved` is a **frozenset of Request types**, never a boolean — the
command's own AST-read `requests` declaration, fixed when the user answered.
`push` copies it down unchanged, so a callee can never widen it.

### The eight ways a branch decides

Every conditional branch uses one of these, and they are the vocabulary any
*new* rule should be built from. Also written down in `sandbox/policy.py`
itself, above `classify` — this table is the summary of that comment:

| # | Mechanism | Question | Used by |
|---|---|---|---|
| 1 | **Destination** | is the path inside a scratch root? | `fs.write`, `fs.write_bytes`, `fs.move`, `fs.delete` |
| 2 | **Allowlist** | is the host in `net_allowed_hosts`? (dot-boundary subdomain match) | `net.http` |
| 3 | **Ownership** | did this plugin *declare* this setting? | `secret.reveal`, `config.write` |
| 4 | **Shape** | does the SQL delete from a kernel table? | `db.write` |
| 5 | **Polarity** | does this widen or narrow? | `task.pause` (pause safe, unpause unsafe) |
| 6 | **Attendance** | is a person there? | `ui.ask` |
| 7 | **Provenance** | is this the command the user just typed? (`chain.typed_command`) | `config.write` |
| 8 | **Recognizer** | does a pluggable predicate vouch for it? | `proc.run`/`proc.start` (`sandbox/shell.py`), `net.http` |

**`shell._SHELL_RECOGNIZERS` now holds two** (`sandbox/shell.py`, split out
once the shell family grew a lexer of its own) — a structural read-only check and a
*remembered* one reading `shell_allowed_prefixes`. `_NET_RECOGNIZERS` is still
empty, because egress is served by its allowlist directly. Both remain the
designed extension point, and a recognizer can only ever widen.

## 6. Layer 5 — The approver

Runs only for UNSAFE. In order, first decisive answer wins:

| # | Stage | Source of truth | Notes |
|---|---|---|---|
| 1 | `vet_permission` hooks | any service | stage is `approval` or `unattended_call` by attendance |
| 2 | secret ownership | setting registry | a plugin reading the credential it declared |
| 3 | attendance | `policy.attended_now` | nobody home ⇒ refuse, never block |
| 4 | **the mode** | `runtime.security_mode(session)` | `lockdown` ⇒ no, `yolo` ⇒ yes, `ask` ⇒ fall through. See §6a |
| 5 | dialog | `runtime.request_input` | 300 s; timeout and cancel both mean **no**. Its *options* are where a yes can be kept — `sandbox/options.py` |

## 6a. The mode — a standing answer, not a new layer

`/mode` sets what a conversation answers *instead of* drawing the dialog at
step 5. Three values, and they are the three answers a person can give in
advance: `lockdown` (no), `ask` (the default), `yolo` (yes).

Its position in the order is the whole of its scope, and both neighbours are
deliberate:

- **After attendance (3),** so `yolo` never reaches work nobody is watching.
  A cron job, a service poll tick or a subagent is refused whatever the
  foreground conversation is set to — `policy` rests the safety of
  `agent.spawn` on exactly that.
- **After the hooks and the secret exemption (1–2),** so `lockdown` answers
  only what would otherwise have reached the person. It does not countermand
  a plugin gate that positively allowed something, and it does not stop a
  service reading the credential it was configured with. Lockdown means "stop
  asking me, the answer is no" — not "break the plugins I already set up".

Two limits worth stating wherever the mode is offered, because a grant that
overstates itself erodes trust in the dialog as fast as one that understates
it erodes safety:

- **`yolo` is not root.** Every layer-2 refusal stands. Those never produced a
  question, so there is no answer for a mode to stand in for.
- **`lockdown` is not a trap.** It is enforced here, so the one act that
  leaves it must never arrive here: `session.set_mode` is SAFE for
  `chain.typed_command`, the same exemption `config.write` uses. `/mode ask`
  therefore always works.

**Where it lives.** `runtime/security_modes.py` holds the vocabulary — kernel
rather than `sandbox/`, because two layers read it. The value is an ephemeral
field on `RuntimeSession`, scoped to a conversation *structurally*: the
session stores the mode and the `conversation_id` it was set against, and the
reader answers the default when they disagree. So there is no list of reset
sites to keep in step with `/new`, `/clear` and `load_conversation`, and a
mode cannot leak into the next conversation because there is nowhere for it to
leak from. Nothing is persisted, so a restart returns to `ask`.

**Who may change it** is mechanisms 5 and 7 together: arriving at `lockdown`
narrows whatever we were in, so an agent may do it unasked; every other value
could widen, so it raises a dialog unless the person typed `/mode` themselves.

**The turn scope.** `session.set_mode(scope="turn")` sets a mode the kernel
drops at `HookRegistry.finish_turn` — stacked there rather than registered as
a `turn_finish` hook, because a grant that expires only when some plugin
happens to be installed is not a grant that expires. This is what "Allow, and
stop asking for the rest of this turn" writes, and what an approved plan will
hand the turn that follows it.

There was a step between 1 and 2: **`skip_permissions`**, a user-scoped list of
plugin names whose dialogs were auto-approved. It was the only durable answer
the system had, which is why its unit had to be that broad. Once an answer
could be kept at the grain the question was asked at, a whole-plugin bypass was
strictly worse than the thing it stood in for, so it was removed rather than
left as a blunter option.

## 7. Layer 6 — State-machine approval (commands)

The *preferred* path, and the only one that states its scope before any work
happens.

| Declaration | Granularity |
|---|---|
| `require_approval = True` | the whole command |
| `approval_actions = (...)` | named actions |
| `approval_action_prefixes = (...)` | action prefixes |

**`yolo` reaches here too; `lockdown` deliberately does not.** A conversation
in `yolo` pre-answers this dialog via `ConversationState.auto_approve`, which
routes through the normal `_run(approved=True)` path — so the command gets the
same `chain.approved` grant a typed "yes" produces rather than running
ungranted. Lockdown stops at layer 5: this dialog is about a command *the
person just typed*, with them sitting right there, and auto-refusing it would
make lockdown mean "you may not use your own machine" — including, fatally,
the `/mode` that leaves it.

Read by **AST**, so they must be literals — `tuple(ACTIONS)` reads as nothing.
Answering yes sets `context.approved_by_state_machine`, which becomes
`chain.approved` = the command's declared `requests`.
`tests/test_command_approval_declarations.py` pins that every command touching
`policy.ALWAYS_UNSAFE` declares a gate.

---

## 8. Sources — what each chain root can reach

Attendance is a fact about **why** something is running, not about which family
it belongs to. There is no per-family rule anywhere, and there should not be.

| Root | Origin | Attended | Can spend a grant? |
|---|---|---|---|
| `user` | a person's action | yes¹ | no |
| `user:command` | a slash command they typed | yes¹ | **yes** — and `typed_command` exempts `config.write` at depth ≤ 1 |
| `<session_key>` | the agent's own tool call | iff that session is | inherits the caller's |
| `spawn_subagent:<cid>` | a subagent | **no** — its session is never active | no |
| `service:<name>` | poll tick, hook, bus delivery | no | no |
| `frontend:<name>` | a frontend's own loop | no | no |
| `cron:<job>` | scheduled work | no | no |
| `kernel` / `agent` | no session at all | no | no |

¹ subject to the frontend's own opinion via `runtime.set_session_attended`.

A service reached *from* an attended tool inherits that tool's root and can ask.
The same service waking on its own poll tick cannot. A script is attended
exactly when whatever ran it was. That is the whole rule.

---

## 9. Your wishlist, mapped

| Wanted | Status | Where it goes |
|---|---|---|
| Allow once | **exists** — the dialog | — |
| Deny once | **exists** — the dialog | — |
| Allow always (per tool/plugin) | **removed** — was `skip_permissions` | superseded by the three destination grants |
| Allow web domain | **exists** — an answer option, writing `net_allowed_hosts` | — |
| Allow writable folder | **exists** — an answer option, writing `fs_writable_dirs` | — |
| Allow command prefix | **exists** — an answer option, writing `shell_allowed_prefixes` | matched as `(program, subcommand)`, never a string prefix |
| Allow until end of turn | **exists** — an answer option, writing `turn_security_mode` | `options._rest_of_this_turn` |
| Deny forever | **missing** | an `OPTION_BUILDERS` entry — `build_approver` already runs `remember` for denying options |
| Auto accept all | **exists** — `/mode yolo` | §6a |
| Auto deny all | **exists** — `/mode lockdown` | §6a |
| Default / manual | **exists** — `/mode ask`, the default | — |
| Plan mode | **not built** — the substrate is | a fourth mode value plus a `propose_plan` tool; see below |

**The shape of what is left.** The three *destination* grants turn one answer
into an entry in a list the user keeps, with `/config` as the undo — which is
the whole reason they live in config rather than the database. The grant scoped
to **time** now exists too, and needed no store: its unit expires on its own,
so there is nothing for the person to find and revoke later, which is exactly
the difference between it and the three lists. Only "Deny forever" is left, and
nothing durable remains whose unit is a whole plugin.

Two things this settled, both of which were guesses in an earlier draft of this
file and both of which turned out slightly wrong:

- **`OPTION_BUILDERS` is the designed home** for a new kind of answer, and
  `_rest_of_this_turn` is the proof: an `Option` is
  `(value, label, allow, remember)` where `remember` is an opaque closure, so
  it wrote to the session instead of to config and neither `options_for` nor
  the dialog had to learn it was different.
- **Modes are *not* a `vet_permission` gate**, which is what this file used to
  say. A hook comes from a service, and a service is a store package — a
  lockdown that stops working when you uninstall something is worse than none.
  So the mode is kernel-owned and hook-*shaped*: it stands at the same point in
  the order a gate would, without being registered. Same argument as the
  compaction layer and the subagent barrier.

**Plan mode**, when it comes, is a fourth value of the same field plus a tool.
Everything else it needs is built: a mode that refuses (`lockdown`), a
turn-scoped yolo for the turn after approval, a Request that sets the mode, a
per-turn prompt line stating it, and the clearing at turn end. The retired
implementation on `origin/store` (`service_plan_mode.py`, `tool_propose_plan.py`)
does not load under the current sandbox, but its three-choice approval is
exactly the new turn option plus a `set_mode` call.

---

## 10. Findings from compiling this

Three things that were not design, and were not visible until the layers were
laid side by side. **Two are now fixed**; they are kept here because the shape
of each is worth recognising again.

**~~There are two approval doorways.~~ Fixed.** `sdk.ui.approve` was
`ALWAYS_SAFE`, executing a handler that reached `context.approve_command` — a
parallel doorway with its own hook call, its own reading of `skip_permissions`
(by **tool name**, not by chain), and **no attendance check at all**, so it
would block 300 s on a dialog in a session nobody was watching.

The fix was not to teach it to agree but to delete it. The distinction that
resolves it: **`ui.ask` gathers information, `ui.approve` seeks
authorization** — only the second is a policy decision, so it belongs to
`classify`. It is now unconditionally UNSAFE, which makes *the Request itself
the question*: the gate runs the whole pipeline (hooks → trusted list →
attendance → dialog) and the handler has nothing left to do but report that
the answer was yes. The justification becomes `Decision.reason`, which is
exactly the slot the dialog prints under "Why it needs asking".

**~~`skip_permissions` is read two ways.~~ Fixed** — it collapsed with the
doorway. One reader, `runtime.user_setting`, matching the whole chain.

**Standing grants are now enumerable and withdrawable** — `/permissions`
([bundled/commands/command_permissions.py](../bundled/commands/command_permissions.py))
lists all three settings and revokes from any of them, with the two things the
lists do not say: scratch and the workspace tree are free without any grant,
and the app's own files are never grantable whatever is listed.

**No layer records what it decided at the grain it decided it.** Still open,
and it is the blocker for everything in §9. The ledger records the Request and
whether it was refused, but a "yes" leaves no trace of *what was approved* —
the chain, the host, the rendered command. `shell.render_command` already
exists as the one renderer the dialog, the ledger and a future recognizer
share. That sharing is the seam to build on.
