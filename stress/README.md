# `stress/` — kernel stress-testing harness

QA scaffolding for the lite kernel, modelled on how real kernels are tested
(layered: selftests → coverage-guided fuzzing → sanitizers + CI). Nothing here
is imported by the kernel; it only imports *from* it.

Install dev deps once: `pip install -r requirements-dev.txt`

## The three layers

| Layer | File | Kernel analog | What it catches |
|---|---|---|---|
| **Sanitizer** | `invariants.py` | KASAN/KMSAN | Latent corruption made loud: broken DB integrity, orphaned messages, **cross-user ownership**, leaked phase frames / threads, registry drift. |
| **Fuzzer** | `fuzz_runtime.py` | syzkaller/syzbot | Bugs in *orderings* of lifecycle ops against the runtime (the 940-line dispatcher): conversations, users/identity, profile switches, plugin/session state, system-prompt extras, persist→reload round-trip. Oracle = `invariants` after every step. |
| **Fuzzer** | `fuzz_packages.py` | syzkaller/syzbot | Install/uninstall *orderings* against the package store with a fake catalog (deps, shared deps, bundles). Oracle = store integrity (files↔receipts, no orphans, no dangling `requires`). |
| **Fuzzer** | `fuzz_packages_cleanup.py` | syzkaller/syzbot | The hairy uninstall **cleanup form** (all/none/specific × per-package config/tables/pip), driven through the *real* `PackagesCommand` form. Oracle = shared config/tables/pip declared by a still-installed package are never collateral, and kernel-requirement pip is never removed. |
| **Driver** | `driver.py` | (no kernel analog) | Semantic/UX bugs a blind fuzzer can't reach — an intelligent agent (MiniMax) as the brain, a human/Claude as the adversarial user. |
| **Real-LLM stress** | `stress_compaction.py` | — | Drives a real MiniMax conversation with a tiny `context_size` so the compactor service fires; asserts compaction happens, invariants hold across it, and a fact planted pre-compaction survives the summary. |

Supporting: `boot.py` (headless full-kernel boot against a throwaway or
persistent data dir, pluggable LLM) and `fake_llm.py` (network-free
`ScriptedLLM` + seeded `MonkeyLLM` fuzzing brain).

## Running

```bash
# Fuzz the runtime + the package store (deterministic, no network). Every
# failure shrinks to a minimal reproducing op-sequence — turn it into a tests/
# regression test.
pytest stress/fuzz_runtime.py stress/fuzz_packages.py -q
pytest stress/fuzz_runtime.py -q --hypothesis-seed=random   # deeper search

# Real-LLM compaction stress (uses MiniMax).
python -m stress.stress_compaction

# Drive the kernel as a user with MiniMax as the agent brain (real network).
# State persists across invocations in .stress_driver/ so you drive turn-by-turn.
python -m stress.driver say "hi — what can you do?"
python -m stress.driver say "/commands"
python -m stress.driver history
python -m stress.driver check        # run the invariant oracle on live state
python -m stress.driver reset        # fresh conversation
python -m stress.driver --profile openai/deepseek-ai/deepseek-v4-pro say "..."
```

## Using the oracle in your own tests

```python
from stress.boot import boot_kernel
from stress.fake_llm import MonkeyLLM
from stress.invariants import check_invariants

k = boot_kernel(llm=MonkeyLLM(seed=0))
# ... drive operations ...
assert not check_invariants(k)
k.close()
```

## Findings log

- **Identity hot-swap under a live conversation** — *fixed*. Found by the fuzzer
  on day 1: `runtime.set_session_user(key, other_user)` overwrote
  `session.user_id` while the session still held the *original* user's
  conversation, leaving the new identity able to read/append to another user's
  thread (the ownership guard runs on load/mutate-by-id paths, not on identity
  reassignment). Fix: `set_session_user` now treats an identity change on a live
  session as an **account switch** — it remembers the departing user's
  conversation as their last-active, detaches it, and loads the new user's own
  last-active (lazy-creating a fresh one on the next turn if they have none).
  Regression tests: `tests/test_user_isolation.py::test_set_session_user_*`.

- **Deleting a conversation a live session still holds** — *fixed*. The mirror of
  the above (a conversation-side mutation that didn't reconcile sessions). A
  conversation can be deleted from a different session than the one viewing it
  (another tab/frontend, the agent, or `/conversations` deleting the
  currently-open conversation). The holding session kept `conversation_id`
  pointing at the deleted row and **hard-crashed on its next message with a
  `FOREIGN KEY constraint failed`**. Fix: `runtime.delete_conversation` now
  detaches any live holder to `None` (next turn routes through the
  no-conversation guard / lazily creates a fresh one) and drops stale last-active
  pointers. Regression test:
  `tests/test_user_isolation.py::test_delete_conversation_detaches_live_sessions`.

- **Dangling `active_session_key` after `close_session`** — *fixed*. Surfaced by
  the fuzzer's `raw_delete_then_drive` rule. `close_session` removed the session
  but left `runtime.active_session_key` pointing at it; since `is_attended`
  compares against that pointer, every *other* live session read as unattended
  (replies → notifications, interactive tools refused) until some action reset
  it. Fix: `close_session` clears `active_session_key` when it names the closed
  session. Regression test:
  `tests/test_user_isolation.py::test_close_session_clears_dangling_active_session_key`.

### Observations (not bugs, worth a design decision)

- **Uninstall form offers un-removable packages.** The `package_id` step lists
  *every* installed package, including ones another installed package depends on.
  Selecting such a package makes `form()` call `build_uninstall_plan`, which
  raises `PackageError("Cannot uninstall …; required by …")`. It does **not**
  crash — `Action.enact` catches it and surfaces the message — but the UX would
  be cleaner if the step filtered to standalone-removable packages (or showed
  the dependents up front). Found by `fuzz_packages_cleanup`.
- **`litellm` is not a kernel requirement.** It's commented out of
  `requirements.txt` (kernel-minimal), so `_safe_pip_removals` does not protect
  it — uninstalling `service-litellm` will pip-remove `litellm`. Defensible (it's
  one of several possible backends), but since the kernel can't run without *a*
  backend, you may want a "protected pip" list independent of `requirements.txt`.

### Structural net: the write-path backstop

Rather than only point-fixing each mutator, `runtime.handle_action` now calls
`_reconcile_session_binding(session)` at entry: before any action writes against
`session.conversation_id`, it verifies the row still exists and is still owned by
the session's user, detaching to `None` otherwise. This catches desyncs **no
individual mutator remembered to reconcile — including ones not yet written** —
turning a FOREIGN KEY crash / cross-user write into a clean re-route. The point
fixes above remain (cheaper, and they preserve context); this is the net under
them. The fuzzer's `raw_delete_then_drive` rule exercises it on every run by
deleting through the raw `db` path that bypasses the runtime's own detach.
