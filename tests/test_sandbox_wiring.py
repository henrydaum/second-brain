"""The things a green suite was not checking.

Every claim here was false in a tree where all 1132 tests passed, which is the
only reason this file exists. They share a shape: each one is a *wire* between
two halves of the sandbox that were individually well tested and not tested
together — a service and the context that answers it, a chain and the callee
that should descend from it, an escort and the model name it is shown, a
Request and the ledger that is supposed to record it.

The lesson worth keeping is that unit tests on both ends of a wire say nothing
about the wire.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox, provenance
from sandbox.bridge import adapt, configure
from sandbox.console import Console
from sandbox.guest.requests import Request, Result
from sandbox.interpreter import Execution, Interpreter
from sandbox.policy import SAFE, UNSAFE, Chain, Decision, classify

# ──────────────────────────────────────────────────────────────────────
# A resident service must have something to answer Requests from.
# ──────────────────────────────────────────────────────────────────────

SERVICE = '''
"""A service that persists a setting it owns."""

from guest.bases import BaseService


class Keeper(BaseService):
    """Reads and writes its own config."""

    name = "keeper"
    exports = ["remember", "recall"]
    requests = ["config.read", "config.write"]

    def start(self, sdk):
        """Nothing to open."""
        return True

    def remember(self, sdk, value):
        """Persist through the service-owned setting."""
        sdk.config.write("keeper_note", value, scope="plugin")
        return True

    def recall(self, sdk):
        """Read it back."""
        return sdk.config.read("keeper_note")
'''


@pytest.fixture
def box():
    """A sandbox the bridge routes migrated plugins through."""
    made = Sandbox()
    configure(made)
    yield made
    configure(None)
    made.shutdown()


@pytest.fixture(autouse=True)
def clean_boxes():
    """Boxes are module caches; a leak hides staleness."""
    yield
    for name in ("service_keeper",):
        unload_box(name)


def _keeper(tmp_path, sandbox_):
    """Build and load the migrated service the way discovery would."""
    path = tmp_path / "service_keeper.py"
    path.write_text(SERVICE, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    return module.build_services({})["keeper"]


def test_a_resident_service_can_reach_config(tmp_path, box):
    """A service is loaded before any session exists, so nothing handed it a
    context — and a handler with no context answers from nothing.

    ``config.read`` is the probe because it is classified SAFE and therefore
    actually reaches a handler. It is also where the damage was worst: it
    returned None for every key, which is indistinguishable from unset, so the
    timekeeper read back an empty job list and carried on.
    """
    store = {"keeper_note": "already on disk"}
    box.bind_context(lambda session_key=None: SimpleNamespace(
        config=store, db=None, services={}, runtime=None, user_id=1,
        session_key=session_key))

    service = _keeper(tmp_path, box)
    assert service.load() is True
    try:
        assert service.recall() == "already on disk"
    finally:
        service.unload()


def test_without_a_context_a_service_reads_nothing(tmp_path, box):
    """The regression, stated as the bug: no context, no answer.

    Note what this does *not* do — raise. That is the whole reason it survived
    a green suite for so long: an unwired service looks exactly like a
    correctly wired one whose setting happens to be unset.
    """
    service = _keeper(tmp_path, box)
    assert service.load() is True
    try:
        assert service.recall() is None
    finally:
        service.unload()


def test_a_service_write_outside_its_own_settings_is_refused(tmp_path, box):
    """Reaching the handler is not the same as being allowed to.

    A plugin persisting a setting the registry says it owns is safe; anything
    else is a config change and asks. This service is synthetic and owns
    nothing, so it is refused at the gate — which is the correct answer and
    confirms the context did not quietly widen anything.
    """
    from sandbox.bridge import ServiceCallFailed

    box.bind_context(lambda session_key=None: SimpleNamespace(
        config={}, db=None, services={}, runtime=None, user_id=1,
        session_key=session_key))
    service = _keeper(tmp_path, box)
    assert service.load() is True
    try:
        with pytest.raises(ServiceCallFailed) as caught:
            service.remember("not mine to write")
        assert "denied" in str(caught.value)
    finally:
        service.unload()


# ──────────────────────────────────────────────────────────────────────
# Provenance has to survive a Request that re-enters the sandbox.
# ──────────────────────────────────────────────────────────────────────

def test_a_callee_descends_from_its_caller():
    """``Chain.push`` was only ever called at the outermost run, so every
    chain was one link deep and the callee started a fresh one beside its
    caller rather than below it."""
    outer = Chain(root="user").push("tool_a")
    with provenance.serving(outer, None):
        caller = provenance.current()
        assert caller is not None
        inner = caller.chain.push("service_b")

    assert inner.render() == "user -> tool_a -> service_b"
    assert inner.depth == 2


def test_a_tool_reaching_itself_is_refused():
    """The cycle detector could never fire while chains were one link deep,
    so a tool calling itself recursed until a pool ran dry."""
    chain = Chain(root="user").push("tool_a")
    with provenance.serving(chain, None):
        recursive = provenance.current().chain.push("tool_a")

    assert recursive.cyclic
    verdict = classify(Request("fs.read", {"path": "x"}), recursive)
    assert verdict.level == UNSAFE
    assert "cycle" in verdict.reason


def test_the_ambient_caller_is_cleared_even_when_a_handler_raises():
    """A pool worker keeps its context between tasks, so a value left behind
    would be believed by the next, unrelated Request that landed on it."""
    assert provenance.current() is None
    with pytest.raises(RuntimeError):
        with provenance.serving(Chain(root="user"), None):
            raise RuntimeError("handler blew up")
    assert provenance.current() is None


def test_a_grant_is_not_re_derived_by_a_callee():
    """``push`` copies the caller's grant down unchanged. A callee that read
    its *own* declaration instead would widen the set the user answered."""
    approved = Chain(root="user", approved=frozenset({"net.http"})).push("cmd")
    inner = approved.push("tool_b")

    assert inner.approved == frozenset({"net.http"})
    assert classify(Request("net.http", {"url": "https://x"}), inner).level == SAFE
    # And nothing outside the grant rides in on it.
    assert classify(Request("proc.run", {"argv": ["ls"]}), inner).level == UNSAFE


# ──────────────────────────────────────────────────────────────────────
# A deadline must measure the guest, not the clock.
# ──────────────────────────────────────────────────────────────────────

def _execution():
    """A bare execution to account time against."""
    return Execution(name="probe", chain=Chain())


def test_waiting_on_the_kernel_is_not_charged_to_the_guest():
    """An escort placing a model call, or a service inside sdk.ui.ask, is
    waiting for something the kernel itself started. Charging that to its
    deadline killed the healthy case — and made escorts unusable."""
    execution = _execution()
    started = time.monotonic()
    execution.entered()
    time.sleep(0.4)
    execution.left()

    assert execution.running_for(started) < 0.2


def test_a_runaway_that_spins_on_requests_still_runs_out():
    """The tempting simplification — "time since the guest last did
    something" — makes ``while True: sdk.fs.list('.')`` immortal, because it
    is never idle for more than a millisecond. Blocked time is subtracted
    instead, which takes one long wait off in full and a thousand short ones
    off to nearly nothing."""
    execution = _execution()
    started = time.monotonic()
    end = time.time() + 0.4
    while time.time() < end:
        execution.entered()
        execution.left()

    assert execution.running_for(started) > 0.15


def test_pure_compute_is_charged_in_full():
    """Nothing is discounted for code that never asks the kernel anything."""
    execution = _execution()
    started = time.monotonic()
    time.sleep(0.3)
    assert execution.running_for(started) >= 0.3


def test_nothing_outlives_the_hard_ceiling():
    """Blocked time is discounted, so a runaway that hides inside long
    Requests would otherwise never be overdue at all."""
    from sandbox.watchdog import HARD_CEILING, overdue

    execution = _execution()
    execution.entered()          # blocked, and staying blocked
    started = time.monotonic()
    # Far under the running deadline, far over the wall-clock ceiling.
    assert overdue(execution, started, deadline=1e9,
                   now=started + HARD_CEILING + 1)


# ──────────────────────────────────────────────────────────────────────
# The console is lent, not surrendered.
# ──────────────────────────────────────────────────────────────────────

class _Feed:
    """A source that yields slowly, so a reader is still blocked in it."""

    def __init__(self, lines, pause=0.2):
        self._lines = list(lines)
        self._pause = pause
        self._at = 0

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(self._pause)
        if self._at >= len(self._lines):
            raise StopIteration
        self._at += 1
        return self._lines[self._at - 1]


def test_reclaiming_the_console_does_not_start_a_second_reader():
    """Release used to stop the reader, which cleared the handle while that
    thread was still blocked in ``readline`` — so the liveness guard could not
    see it and the next claim started another. Two readers split a person's
    keystrokes, which presents as the machine dropping characters."""
    console = Console()
    feed = _Feed(["one\n", "two\n", "three\n"])
    before = _readers()

    console.claim("first", source=feed)
    time.sleep(0.05)
    console.release("first")
    console.claim("second", source=feed)
    time.sleep(0.05)

    assert _readers() - before == 1
    console.stop()


def test_a_superseded_reader_cannot_close_the_console_under_its_successor():
    """The orphan's ``finally`` set ``_closed``, so a frontend that had just
    claimed the console successfully got EOFError on its next read and stopped
    itself. A frontend restart therefore killed the terminal."""
    console = Console()
    console.claim("first", source=_Feed(["a\n"], pause=0.15))
    time.sleep(0.02)
    console.release("first")
    console.claim("second", source=_Feed(["b\n"] * 5, pause=0.15))

    time.sleep(0.4)
    console.read_line()          # must not raise
    console.stop()


def test_release_does_not_hand_the_next_claimant_stale_keystrokes():
    """What was typed belonged to the frontend that has gone away.

    Replaying it into a fresh session would answer a prompt nobody had seen
    with a stranger's keystrokes.
    """
    console = Console()
    console.claim("first", source=_Feed(["secret\n"] * 5, pause=0.02))
    time.sleep(0.15)
    assert console._lines, "nothing was buffered, so the test proves nothing"

    console.release("first")
    assert not console._lines
    console.stop()


def _readers() -> int:
    """How many console reader threads are alive right now."""
    return sum(1 for t in threading.enumerate() if t.name == "console-reader")


# ──────────────────────────────────────────────────────────────────────
# The flight recorder was not recording.
# ──────────────────────────────────────────────────────────────────────

def _sink(tmp_path):
    """A real database and the sandbox sink that writes to it."""
    from pipeline.database import Database
    from runtime.ledger import sandbox_sink

    db = Database(str(tmp_path / "ledger.db"))
    return db, sandbox_sink(db)


def test_an_effect_is_recorded_with_its_whole_chain(tmp_path):
    """Nothing wired ``record=``, so no Request a plugin ever made reached the
    ledger — the one place unattended work is meant to be reconstructable
    from."""
    from runtime.ledger import SANDBOX_ORIGIN

    db, record = _sink(tmp_path)
    chain = Chain(root="cron:nightly").push("task_index").push("service_web")
    record(chain, Request("fs.write", {"path": "/tmp/x"}),
           Decision(SAFE, "scratch"), Result(data=True))

    [row] = db.get_ledger_rows(origin=SANDBOX_ORIGIN)
    assert row["action_type"] == "fs.write"
    assert row["ok"] == 1
    assert "cron:nightly -> task_index -> service_web" in row["data_json"]


def test_polling_reads_are_not_recorded_but_denials_are(tmp_path):
    """A console frontend reads every poll — twenty rows a second, forever,
    burying everything worth reading. But a *denied* read is a real event."""
    from runtime.ledger import SANDBOX_ORIGIN

    db, record = _sink(tmp_path)
    chain = Chain(root="user").push("frontend_repl")

    record(chain, Request("console.read", {}), Decision(SAFE, "ok"),
           Result(data="hi"))
    record(chain, Request("fs.read", {"path": "/etc/x"}),
           Decision(UNSAFE, "no"), Result.refusal("nope"))

    rows = db.get_ledger_rows(origin=SANDBOX_ORIGIN)
    assert [r["action_type"] for r in rows] == ["fs.read"]
    assert rows[0]["error_code"] == "denied"


# ──────────────────────────────────────────────────────────────────────
# The escort's view of the model.
# ──────────────────────────────────────────────────────────────────────

def test_an_escort_is_shown_the_profile_name(monkeypatch):
    """``ModelRequest.llm`` is a name. Reaching for ``.model_name`` on a string
    yields nothing, so every escort ever built was shown ""."""
    from runtime.hooks import ModelRequest
    from sandbox.hooks import project_model_request

    request = ModelRequest(llm="fast-profile", messages=[])
    assert project_model_request(request)["llm"] == "fast-profile"


def test_an_escort_swaps_by_name_and_unknown_names_are_ignored(monkeypatch):
    """Backends stopped being services when the LLM became kernel routing, so
    looking a profile up in ``runtime.services`` found nothing and no
    sandboxed escort could swap a model at all."""
    import llm as llm_registry
    from runtime.hooks import ModelRequest
    from sandbox.hooks import apply_model_request

    monkeypatch.setattr(llm_registry, "brain",
                        lambda name: object() if name == "big" else None)
    runtime = SimpleNamespace(services={}, config={})
    request = ModelRequest(llm="small", messages=[])

    apply_model_request(request, {"llm": "big"}, runtime)
    assert request.llm == "big"          # the name, never a brain object

    apply_model_request(request, {"llm": "does-not-exist"}, runtime)
    assert request.llm == "big"          # silently retargeting is the worst case
