# The Standard — Design Ontology of Second Brain

*"Everything has pros and cons."*

## Preface: from The Whole to Second Brain

Second Brain did not begin as software. It began as a study of the real
nervous system. Before this project, there was **The Whole**, a short
book that maps the mind, body, and brain: how the senses feed a fast-acting
"Doer" and a slow, predictive "Thinker," how the spinal cord connects brain
to body, and how the parts fit together into one organism that learns,
survives, and occasionally breaks free of suffering. You do not need to have
read it. What matters here is the picture it left behind — a whole made of
parts, and rules about how the parts may touch.

Second Brain is the next project in that lineage: an attempt to *build* a
small version of what the book *describes*. The LLM is the cortex, the
Thinker. The conversation state machine is the spinal cord. Plugins are the
body: senses, muscles, viscera. The Timekeeper is a circadian clock; the
action ledger is memory of one's own actions; the supervisor and heartbeat
are an immune system; installing and uninstalling packages is plasticity —
learning without surgery.

The analogy is the ancestor of the design, not a requirement to follow. Never
contort code to fit it. But like the book, this document is a **map for
navigating a wilderness** — here, the wilderness of arbitrary code — and maps
do not show every shrub. It is more descriptive than prescriptive, and
extremely pragmatic on purpose.

This document is for three readers: human contributors, dev agents working on
the repo (Claude Code, Codex, …), and the in-app Second Brain agent authoring
or reviewing sandbox plugins. If anything here conflicts with an explicit
instruction from Henry, the instruction wins — then this document is updated
to match. Companions: [CLAUDE.md](CLAUDE.md) (architecture),
[AGENTS.md](AGENTS.md) (dev-agent quickstart), `templates/*_template.py`
(per-family contracts). This is the layer above them: how to judge code that
no template anticipates.

## Introduction: the parts, and Rice's theorem

Second Brain's standard can be understood as comprising eight parts — the
core values:

- I) Correctness
- II) Modularity
- III) Safety
- IV) Simplicity
- V) Efficiency
- VI) Elegance
- VII) Practicality
- VIII) Readability

The main question is: **does this code align with Second Brain's core
values?** Rice's theorem says that no non-trivial semantic property of
arbitrary code is decidable — we cannot build a checker that answers this
question perfectly. We do not try. Instead we cheat the theorem the only way
available: refuse to evaluate arbitrary code, and evaluate **observable
commitments** instead — which base class it inherits, what it declares
(`dependencies_files`, `dependencies_pip`, `config_settings`,
`background_safe`, `requires_services`, `trigger_channels`), which seams it
plugs into (registries, hooks, the bus), and which seams it avoids (kernel
edits, raw `db.*` bypasses, hardcoded paths). Code that makes all the right
commitments can still misbehave, but each part below narrows the space where
misbehavior can hide, and the ledger records whatever slips through. The goal
is to keep asking decidable questions until the undecidable remainder is
small, visible, and auditable.

Each part is given in the same shape:

1. a definition and explanation,
2. its **hard rules** — the gates, pass/fail; any failure rejects the code
   regardless of every other score,
3. its **graded question** — scored 0 (violates) / 1 (strained) / 2 (sound) /
   3 (exemplary), with sub-questions to guide the score, and
4. its **guardian tests** — the files in `tests/` that make the part
   executable, in the same spirit as `tests/test_kernel_boundary.py`. Where
   the guardians are thin, the gap is stated honestly rather than papered
   over.

---

## Part I: Correctness

Correctness is code doing what it claims across the whole input domain, not
just the demo path.

This is value zero. Much of it Second Brain absorbs by construction — the
state machine makes actions legal or illegal by phase, phase frames are
serializable so interrupted flows restore exactly, the pipeline derives its
DAG rather than trusting declarations — but construction only covers the
code that uses it. New code must earn correctness the ordinary way: by
stating what must hold and proving it against the ugly inputs.

### Hard rules

- **State what must hold.** Non-obvious invariants are written down where
  they can fail — in the code, in a test, or in the stress oracle
  (`stress/invariants.py`), which exists precisely to check live state
  against the rules this document can only describe.
- **Error paths are designed, not accidental.** Every failure leaves
  persisted state consistent: marker and history write atomically, tasks are
  idempotent and keyed by `run_id` so re-delivery and retry are safe, and a
  cancelled or crashed flow restores rather than corrupts.
- **The edges are part of the domain.** Empty input, duplicate delivery,
  restart mid-flow, cancellation mid-stream, a fresh empty DATA_DIR — these
  are enumerated and handled, not discovered by users.

### Graded question

> **Does it do what it claims when the input is ugly?**

- Is the claim itself precise enough to falsify — or does "works" just mean
  "worked once"?
- Does every failure path leave state a restart can trust?
- Are the invariants written where a regression would trip them?

**Guardian tests:** `tests/test_state_machine.py`, `tests/test_database.py`,
`tests/test_pipeline.py`, `tests/test_message_queue.py`,
`tests/test_timekeeper.py`, `tests/test_parser_registry.py` — plus the
stress oracle, which guards the running system the same way these guard the
code.

This is the explanation of Correctness as part of the standard.

---

## Part II: Modularity

Modularity is capability arriving and leaving as a unit, through the seams
the kernel provides, without the kernel knowing its name.

Like a learned skill, a capability is acquired (installed), exercised
(discovered and loaded), and forgotten (uninstalled) without surgery on the
core. The kernel supplies generic seams — discovery by file presence and
prefix, declared metadata, `SecondBrainContext`, `self.services`,
`bind_runtime`, `runtime.hooks`, and the event bus — and a modular plugin
touches the system *only* through them.

The body does not have two spinal cords. Modularity is not just how parts
attach — it is that they all attach to the *same* spine. Every conversation
action, whether from the REPL, Telegram, or a scheduled subagent, flows
through the same labeled `cs.enact(...)` sites; every tool dispatch through
the tool registry; every prompt through `agent/system_prompt.py`. One
mechanism per problem: forms are `FormStep`s on `CallableSpec`, output is
markdown on the wire built with `formatters.py` (never a new structured
message type), and a question with one answer gets one reader (as
`is_attended` does for attendance). Symmetry is the tell — the same code
behaves identically across frontends, attended or unattended, built-in or
installed. An asymmetry is either justified in a comment or a bug.

The event bus deserves its own statement, because it is the connective
tissue between organs. Bus channels exist mostly for communication between
frontends/tasks and the state machine's surroundings: progress, status,
notifications, lifecycle signals. **An event-driven task declares its own
`trigger_channels`; it never borrows or reaches into another plugin's
channels.** Services emit events; they never import the orchestrator or the
tasks that consume them. Tasks never reference each other by name —
dependencies are derived. Emitters and consumers meet only at the named
channel, so either side can be uninstalled without the other noticing.

### Hard rules

- **The kernel boundary is sacred.** Core code hard-imports exactly two
  plugin modules (`service_llm`, `parser_registry`), pinned by
  `tests/test_kernel_boundary.py`. Never change the kernel to accommodate a
  single plugin; if a plugin needs a new seam, the seam must be generic —
  the kernel must not know any package's name. New product capability is a
  package, not a built-in.
- **Kernel / DATA_DIR separation.** The repo tree is the kernel; everything
  mutable lives in DATA_DIR. Paths come from `paths.py`, never hardcoded.
  Plugin helpers use relative imports so files move freely between built-in,
  sandbox, and installed trees. Store files are read by AST-parsed metadata,
  never imported before install.
- **Everything needed is declared.** `dependencies_files`, `dependencies_pip`,
  `requires_services`, `default_jobs`, `config_settings`, `trigger_channels` —
  so install, uninstall, gating, scheduling, and `/config` work with no
  special cases. New kernel entries in `requirements.txt` are presumed wrong.
- **No bypasses of the spine.** No side channels around the state machine,
  tool registry, or prompt assembly; no frontend owning conversation logic;
  no direct transcript mutation. Conversation access by id goes through
  `runtime.assert_conversation_access`; raw `db.*` is the system path only.

### Graded question

> **Could this arrive and leave as a package — and when it leaves, does every
> trace leave with it?**

- Does capability enter through discovery and declaration rather than edits
  elsewhere?
- Does it reach peers only through the provided seams (context, services,
  bus, hooks)?
- On uninstall, do its prompt text, jobs, settings, tools, and channels all
  vanish — would any other file need editing? (If yes, it is not modular.)
- Is its altitude right: seams in the kernel, product capability in the
  package?
- Does it reuse the existing primitive instead of inventing a parallel one —
  would this be the system's second spinal cord?

**Guardian tests:** `tests/test_kernel_boundary.py`,
`tests/test_package_store.py`, `tests/test_plugin_system.py`,
`tests/test_config.py`, `tests/test_default_jobs.py`,
`tests/test_state_machine.py`,
`tests/test_conversation_loop.py`, `tests/test_commands.py`,
`tests/test_formatters_tables.py`, `tests/test_frontend_mcp.py` (an outside
transport uses the same spine without stealing the active-session slot).

This is the explanation of Modularity as part of the standard.

---

## Part III: Safety

Safety is the user consenting to consequential actions, the system surviving
its plugins, and nothing acting, leaking, or persisting silently.

The body survives its organs: a failing organ dims a function without killing
the organism, the immune system quarantines infections, and the conscious
mind — not the reflexes — decides the consequential acts. Would you be
comfortable with this code running from a scheduled subagent at 3 a.m.,
unattended? That is the whole question, asked precisely.

Security is part of Safety, with an honest adversary model: plugins run
in-process, so the real trust boundary is **install time** — the store is
vetted, and nothing may blur the line between vetted and unvetted (store
files are AST-parsed, never imported, before install). Everything that
crosses into the system from outside — frontend traffic, parsed files,
attachments — is untrusted at the seam where it enters, and identity is
checked there, not downstream.

### Hard rules

- **Consent.** The agent never edits, deletes, or executes anything
  consequential without permission. Consequential actions flow through the
  approval system; trusted-tool bypasses (`skip_permissions`) are a
  user-scoped, user-visible setting, never a plugin's unilateral decision.
  Permission policy extends through `runtime.hooks` gates, not patched call
  sites. Important decisions surface as human-readable `config_settings`
  (with `{"scope": "user"}` where personal) — not constants, not magic files.
- **Attendance.** Tools needing a human declare `background_safe = False`;
  the registry refuses them from unattended sessions. A background failure
  goes back to the agent — the user is never prompted from a session they
  are not attending. "Is a human present?" is asked only through
  `runtime.is_attended(...)`.
- **Identity ≠ authorization.** `user_id` answers "whose data";
  `frontend_profile` answers "what is allowed." No privileged admin user;
  `user_type` is metadata, not a bypass. Session identity is ephemeral and
  never persisted in the marker; ownership lives on `conversations.user_id`.
  Refused cross-user access must not leak existence. A `per_user` frontend
  points anonymous traffic at a guest, never the operator.
- **The ledger sees everything.** Every action lands in the action ledger
  (`user_enact`, `agent_enact`, `system`); ledger writes are best-effort and
  never break the system; logged args are names only, never values. No
  secrets in code, logs, ledger, or LLM context.
- **Survive your plugins.** Missing plugins degrade silently and correctly
  (prompt sections vanish, routing falls back, errors point at the fix,
  e.g. `/setup`). Plugins tolerate timeouts, cancellation, and quarantine;
  `unload()` is safe even if `_load()` never ran; no `os._exit()`, no
  unkillable threads, no blocking the conversation loop.

### Graded question

> **What happens when this runs unattended, or goes wrong?**

- Are consequential actions approval-gated and honestly labeled?
- Is every action ledger-visible, with names-not-values logging?
- Does failure point toward the fix and clean up after itself? (Whether
  state stays *consistent* is Correctness's question; whether the failure is
  *contained and visible* is Safety's.)
- Does it ever widen whose data a session can see, or persist the ephemeral?
- Where does untrusted input enter, and is it treated as untrusted there?

**Guardian tests:** `tests/test_ledger.py`, `tests/test_user_isolation.py`,
`tests/test_command_reveal.py`, `tests/test_agent_scope.py`,
`tests/test_supervisor.py`, `tests/test_supervisor_integration.py`,
`tests/test_store_task_jobs.py`.

This is the explanation of Safety as part of the standard.

---

## Part IV: Simplicity

Simplicity is the smallest correct thing.

Evolution does not gold-plate; it keeps what earns its energy. Every
dependency, abstraction, and knob is a metabolic cost charged against the
value it delivers. The kernel stays close to pure Python and boots fast
because everything that could be optional *is* optional.

### Hard rules

- **Stdlib first, dependency-light.** Optional deps belong in plugin
  metadata, so they exist only where the capability does. (*When* an import
  runs is Efficiency's business; *whether* the dependency exists at all is
  Simplicity's.)
- **No migration code, while we can.** Migration scaffolding accretes
  forever and is never deleted — each schema change carries every previous
  one on its back. As long as no live audience depends on stored data, don't
  write it: create schema fresh, regenerate, or reset. This rule is only
  possible *because* there are no users yet whose data must survive; the day
  there are, it flips — and the flip is made deliberately, in this document,
  not one ad-hoc migration at a time.
- **No speculative generality.** No abstraction with one implementation, no
  option nobody set, no config knob without a real user decision behind it —
  surfacing real decisions is Safety; multiplying knobs is a Simplicity
  failure.
- **One knob per unbounded thing.** Retention is the single
  `data_retention_days`; new unbounded tables fold into it. Default jobs are
  silenced by disabling, not deleting.

### Graded question

> **Is this the smallest correct thing, and is the diff proportionate to the
> problem?**

- Could a dependency, class, layer, or setting be deleted with nothing lost?
- Does a one-plugin need produce a kernel refactor (red flag), or does the
  plugin copy kernel logic instead of asking for one small generic seam
  (equally red)?
- Would a fresh install remain almost empty, fast, and quiet?

**Guardian tests:** `tests/test_kernel_boundary.py` (the boundary doubles as
a bloat gauge), `tests/test_config.py` (defaults stay complete and minimal).
*Gap, stated honestly:* there is no test yet pinning `requirements.txt` to a
kernel allowlist; adding one is welcome and would graduate this part from
judgment to guardrail.

This is the explanation of Simplicity as part of the standard.

---

## Part V: Efficiency

Efficiency is respecting the scarce resources: the user's attention, the
LLM's context window, and the machine's time.

Second Brain's scale is one user on one machine, so efficiency here is not
big-O heroics. It is latency where a human is waiting, economy where tokens
cost money, and restraint where a loop runs on every message. The body
budgets the same way: the brain is 2% of the mass and 20% of the energy, and
it still falls asleep to save power.

### Hard rules

- **The conversation is the hot path.** Background work yields to the
  attended session (the DB lock's HIGH/LOW priority); nothing background may
  starve, block, or lag the human who is present.
- **Context is money.** The system prompt stays cache-friendly (static →
  semi-stable → dynamic); plugin prompt text loads only when the plugin is in
  scope; the ledger is read targeted, never linearly; tool results put the
  model-facing summary in `llm_summary` and keep bulk payload in `data`.
- **Discovery stays cheap.** Heavy imports go inside `_load()`/`run()`, never
  at module top level. Work that runs per message or per prompt build must
  not hit the network or disk unthrottled — stamp a clock, keep a cache.
- **Measure before optimizing; think before pessimizing.** Don't tune what no
  one has measured — but a structural cost in a hot path (per-message I/O, an
  unbounded scan) is a design error, not a candidate for later profiling.

### Graded question

> **Where does this code spend someone else's resources — and does it know?**

- What runs while a human waits, and how often?
- What lands in the context window, and does it earn its tokens?
- Is anything unbounded — a scan, a log, a loop — that scales with time
  rather than with need?

**Guardian tests:** `tests/test_store_location.py` (prompt builds stamp a
clock instead of hammering the network),
`tests/test_system_prompt_capabilities.py` (prompt sections gated by scope —
context economy as behavior). *Gap, stated honestly:* there are no
benchmarks; latency and token spend are judged, not pinned.

This is the explanation of Efficiency as part of the standard.

---

## Part VI: Elegance

Elegance is beauty worth its cost.

Elegance is the odd one out — in a sense, the opposite of the other values.
Sometimes you build a thing not because it is simple, or practical, or even
strictly necessary, but because it is *cool as hell*, and that is a
legitimate reason. Evolution knows this value too: the peacock's tail is
cumbersome, expensive, and aids survival not at all — and it exists anyway,
because some things are selected for being magnificent. Token streaming that
renders silky-smooth in Telegram, a conversation that resumes mid-form after
a crash exactly where it stopped, a system prompt engineered so precisely it
stays cache-warm — none of these were the minimum viable anything. They are
the tail feathers, and Second Brain would be poorer without them.

This is also where the "nice to have but not necessary" features live.
Simplicity asks *can this be deleted?* — and for an elegant feature the
honest answer is often yes. Elegance answers back: *but should it be?* A
delightful touch earns its place not by being needed but by being worth it —
the cost accounted, the payoff felt every time someone uses it.

### Hard rules

Elegance is the one part that grants no exemptions and therefore needs few
of its own — its rules are about paying for beauty honestly:

- **Beauty never buys a gate.** No feature is cool enough to bypass Safety,
  the kernel boundary, or the spine. A gorgeous hack is still a hack; the
  peacock still has to survive the season.
- **Pay full price.** An elegant feature meets *every other part's* bar —
  packaged modularly, tested practically, readable, degradable. Polish that
  only works on the happy path is not polish.
- **Beauty is deletable.** Because it is not load-bearing, an elegant
  feature must uninstall or disable cleanly. A nice-to-have that something
  necessary depends on has silently become a must-have — promote it and
  judge it as one, or cut the dependency.

### Graded question

> **Is there something here worth admiring — and is it worth what it costs?**

- Does anything about this code make you want to show it to someone?
- Is the delight felt by the *user* (or the next developer), or only by its
  author?
- Was the cost paid honestly — tested, packaged, degradable — or does the
  beauty lean on an exemption?
- Inverted, for load-bearing code: was an opportunity for grace missed where
  it would have cost nothing?

A score of 3 is code that is both correct *and* has the tail feathers. A 2
is sound code that took its chance at grace where cheap. A 1 is beauty
bought with someone else's budget. A 0 is a hack defended as art.

**Guardian tests:** `tests/test_token_stripper.py`,
`tests/test_streaming_frontend.py`, `tests/test_store_telegram_streaming.py`,
`tests/test_store_telegram_rich.py`, `tests/test_restart_recovery.py` —
the polish features, tested as seriously as the load-bearing ones, which is
exactly the point.

This is the explanation of Elegance as part of the standard.

---

## Part VII: Practicality

Practicality is code that works today, on a real machine, for a real user —
correctness over ceremony.

The Whole is "more descriptive than prescriptive, and extremely pragmatic";
so is the codebase. Ship the 90% solution with a clean seam for the rest —
the store has no versioning yet, on purpose, but records per-file SHA-256
provenance: the seed is planted. That is the pattern.

### Hard rules

- **Tested the way it runs.** `python -m pytest -q` passes before a change is
  done, with real output reported. Tests stub external services (LLM,
  network); new logic gets a test that fakes its dependencies. Absent
  optional native deps are expected, not regressions.
- **Windows-first reality.** Paths, encodings, and shell quirks respected;
  cross-platform via `paths.py`, not assumption.
- **The ugly cases are the real cases.** Restarts mid-form, cancelled
  streams, crashes, empty fresh installs, and frozen processes are handled —
  restart recovery and the heartbeat exist because they happen.

### Graded question

> **Does it actually work, here, today — including the ugly cases?**

- Was it run, not reasoned about? ("It should work" is not a state.)
- Was it exercised in the running app — a real boot, a real conversation —
  not only under pytest? (Whether the *edges* are handled is Correctness's
  question; whether the thing was *actually driven* is Practicality's.)
- Is a complete design that doesn't ship being preferred over a working 90%
  with a seam?

**Guardian tests:** `tests/test_restart_recovery.py`,
`tests/test_heartbeat.py`, `tests/test_service_lifecycle.py`,
`tests/test_repl_stop.py` — and, as the final guardian, the whole suite.

This is the explanation of Practicality as part of the standard.

---

## Part VIII: Readability

Readability is code that a future reader can understand, trust, and change —
durability through legibility.

Code is written once and read for years, mostly by someone who wasn't there:
Henry in six months, a contributor after the store opens, or an agent with no
memory of this conversation. A durable piece of code carries its own
explanation. This applies to the *running* system too: a system whose errors
are invisible is unreadable at runtime no matter how clean its source.

### Hard rules

- **Match the surrounding style.** There is no linter; the codebase is the
  style guide. No drive-by reformatting.
- **Comments state constraints, not narration.** The deliberate decisions are
  written down where they live ("deliberately NOT persisted", "no FKs on
  purpose"); a rule that has been load-bearing twice gets promoted into this
  document.
- **Smart logging: errors are visible, logs are quiet.** Every real failure
  is logged where an operator will find it — never swallowed silently
  (best-effort layers like the ledger swallow *and log*). But do not clog
  the log: no per-iteration chatter, no success spam, no duplicate reports
  of one failure up a call stack. The log is for what went wrong and the few
  state changes that matter; the ledger is for what the system *did* — don't
  conflate them. Log messages never contain secret values (names only,
  never values — same rule as the ledger).

### Graded question

> **Could a stranger read this cold, understand why every line is there, and
> safely change it — and when it breaks at runtime, will anyone know?**

- Are the non-obvious decisions commented with their *why*?
- Do names say what things are, in the codebase's own vocabulary?
- Would a failure in this code surface in the log exactly once, with enough
  context to act on — and would healthy operation stay quiet?

**Guardian tests:** `tests/test_design_values.py` (the manifest test: every
guardian named in this document must exist, so the ontology and the test
suite cannot silently drift apart). *Gap, stated honestly:* readability of
prose and comments is judgment; the guardian only pins the doc-to-tests
binding.

This is the explanation of Readability as part of the standard.

---

## Closing: evaluating with — and evaluating — the rubric

To evaluate any artifact (a sandbox plugin, a store PR, a kernel diff, an
agent's output):

1. **Gates.** Check each part's hard rules the code touches, citing
   file/line for failures. Any gate failure ⇒ **Reject** (fix and
   re-review; a gate failure is never "noted and accepted").
2. **Scores.** Grade each part's question 0–3. Do **not** average: a 23/24
   total can hide a 0 in Safety — the profile *is* the grade.
3. **Verdict.** Any 0, or two or more 1s ⇒ **Rework**. All ≥ 1 ⇒ **Accept
   with changes** (list them). All ≥ 2 ⇒ **Accept**.
4. **Residue.** Then ask the question the checklist cannot decide: *would
   Henry, reading it cold, recognize this as part of the same organism?* If
   it passes the letter but smells wrong, say so and name the smell.

```markdown
## Evaluation: <artifact>
Gates: I ✓/✗ … VIII ✓/✗ (evidence per failure)
Scores: Correctness n/3 · Modularity n/3 · Safety n/3 · Simplicity n/3 ·
        Efficiency n/3 · Elegance n/3 · Practicality n/3 · Readability n/3
Verdict: Accept / Accept with changes / Rework / Reject
Required changes: 1. …
Residue: <what passes the rubric but smells wrong — or "none">
```

And the honest coda: Rice's theorem is not beaten by this document.
"This code aligns with Second Brain's core values" is exactly the kind of
semantic property no rubric can decide. So the framework must be judged as a
*rubric*, not an oracle — by whether it catches the failures that actually
occur, whether two independent evaluations of the same code agree, whether
its gates stay executable as tests, and whether following it produces code
that survives contact with users. When a bad plugin passes or a good one
fails, that is data: amend the part it exposes, in the same change as the
code, the way `test_kernel_boundary.py` forces deliberate amendment of the
boundary. The standard is impermanent; subject to change — that is how it
stays true.

A future tool (a refactored `tool_test_plugin`, or a dedicated evaluator)
should apply this document mechanically where it can — gates and guardian
tests — and hand the graded questions and the residue to an intelligent
reader. That division of labor *is* the design: decidable questions to the
machine, the remainder to a mind.

This is the explanation of the standard as a whole.

## Notes

When parts conflict: Correctness is not traded at all — the other values are
worthless applied to wrong code. Safety wins over everything else. Simplicity
wins over Practicality *inside the kernel* and loses to it *inside plugins* —
a plugin may be pragmatic; the kernel may not. Elegance is the licensed
exception: it may overrule Simplicity's "delete it" when the beauty is worth
the cost, but it never overrules a gate.