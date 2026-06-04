# `stress/` — kernel stress-testing harness

QA scaffolding for the lite kernel, modelled on how real kernels are tested
(layered: selftests → coverage-guided fuzzing → sanitizers + CI). Nothing here
is imported by the kernel; it only imports *from* it.

Install dev deps once: `pip install -r requirements-dev.txt`

## The three layers

| Layer | File | Kernel analog | What it catches |
|---|---|---|---|
| **Sanitizer** | `invariants.py` | KASAN/KMSAN | Latent corruption made loud: broken DB integrity, orphaned messages, **cross-user ownership**, leaked phase frames / threads, registry drift. |
| **Fuzzer** | `fuzz_runtime.py` | syzkaller/syzbot | Bugs in *orderings* of lifecycle ops against the runtime (the 940-line dispatcher). Random valid sequences, oracle = `invariants` after every step. |
| **Driver** | `driver.py` | (no kernel analog) | Semantic/UX bugs a blind fuzzer can't reach — an intelligent agent (MiniMax) as the brain, a human/Claude as the adversarial user. |

Supporting: `boot.py` (headless full-kernel boot against a throwaway or
persistent data dir, pluggable LLM) and `fake_llm.py` (network-free
`ScriptedLLM` + seeded `MonkeyLLM` fuzzing brain).

## Running

```bash
# Fuzz the runtime (deterministic, no network). Every failure shrinks to a
# minimal reproducing op-sequence — turn it into a tests/ regression test.
pytest stress/fuzz_runtime.py -q
pytest stress/fuzz_runtime.py -q --hypothesis-seed=random   # deeper search

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
