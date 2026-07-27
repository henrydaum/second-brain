"""The persistent/ephemeral divide, and how a box ends.

The claims under test:

- ephemeral boxes end at their result; resident ones survive calls and keep
  state
- persistence is decided *before* the run, so code cannot drift into it by
  refusing to finish
- calls are serialized per box
- ending is the kernel's decision, escalating ask -> starve -> kill
"""

import threading
import time
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Interpreter, run_in_process
from sandbox.boxes import BoxError, open_box
from sandbox.runner_subprocess import run_in_subprocess

FIXTURES = Path(__file__).parent / "fixtures"
SERVICE = FIXTURES / "service_counter.py"
SCRATCHPAD = FIXTURES / "scratchpad_server.py"
SCRIPT = FIXTURES / "scratch_script.py"


@pytest.fixture
def interp():
    """An interpreter that refuses everything unsafe."""
    it = Interpreter()
    yield it
    it.shutdown()


@pytest.fixture
def boxes():
    """Open boxes, guaranteed stopped even if a test fails."""
    opened = []

    def _open(*args, **kwargs):
        """Open and track."""
        box = open_box(*args, **kwargs)
        opened.append(box)
        return box

    yield _open
    for box in opened:
        try:
            box.stop()
        except Exception:
            pass
        unload_box(box.name)


@pytest.fixture(params=[False, True], ids=["in_process", "subprocess"])
def isolated(request):
    """Every lifecycle claim is made of both runners."""
    return request.param


def test_host_managed_resident_defers_lifecycle(interp, boxes, isolated):
    """Frontends bind host authority before start and explicitly stop once."""
    box = boxes(
        interp, SERVICE, "Counter", name="counter",
        isolated=isolated, manage_lifecycle=False,
    )

    before = box.call("total")
    assert not before.ok
    assert box.call("start").ok
    assert box.call("total").data == 0
    assert box.call("stop").ok
    assert box.stop().ok


# ──────────────────────────────────────────────────────────────────────
# Ephemeral: the lifetime is the call.
# ──────────────────────────────────────────────────────────────────────

def test_an_ephemeral_run_keeps_nothing(interp, tmp_path):
    """Two runs of the same script share no state."""
    from guest.loader import load_entry

    entry = load_entry(SCRIPT, "pure_math", box_name="scratch_script")
    first = run_in_process(interp, entry, name="m", kwargs={"values": [2]})
    second = run_in_process(interp, entry, name="m", kwargs={"values": [8]})
    assert first.data == 2 and second.data == 8
    unload_box("scratch_script")


def test_an_ephemeral_subprocess_exits_after_its_result(interp, tmp_path):
    """The process is gone once the answer is in."""
    target = tmp_path / "f.txt"
    target.write_text("a b", encoding="utf-8")
    result = run_in_subprocess(interp, str(SCRIPT), "summarize", name="s",
                               box="scratch_script",
                               kwargs={"path": str(target)}, timeout=30)
    assert result.ok
    assert result.data["words"] == 2


def test_never_returning_is_a_hang_not_a_promotion(interp):
    """Code cannot reach unlimited lifetime by refusing to finish.

    This is the invariant the whole distinction rests on: persistence is
    granted by the kernel in advance, never taken by behaving a certain way.
    """
    started = time.perf_counter()
    result = run_in_subprocess(interp, str(FIXTURES / "sandbox_plugin.py"),
                               "spins", name="runaway", timeout=1.0)
    elapsed = time.perf_counter() - started
    assert not result.ok
    assert "timed out" in result.error
    assert elapsed < 15.0


# ──────────────────────────────────────────────────────────────────────
# Resident: state survives calls.
# ──────────────────────────────────────────────────────────────────────

def test_a_service_keeps_state_across_calls(interp, boxes, isolated):
    """The point of a resident box."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    assert box.alive
    assert box.call("add", n=3).data == 3
    assert box.call("add", n=4).data == 7
    assert box.call("total").data == 7


def test_start_runs_before_any_call(interp, boxes, isolated):
    """A box is loaded, not merely spawned."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    assert box.call("total").data == 0    # start() set running = 0


def test_a_resident_box_can_make_requests_mid_call(interp, boxes, isolated,
                                                   tmp_path):
    """Calls and Requests interleave on one channel."""
    target = tmp_path / "note.txt"
    target.write_text("resident", encoding="utf-8")
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    assert box.call("read_file", path=str(target)).data == "resident"


def test_one_bad_call_does_not_kill_the_service(interp, boxes, isolated):
    """A raising method fails that call and nothing else."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    box.call("add", n=5)
    bad = box.call("explode")
    assert not bad.ok
    assert "bad call" in bad.error
    assert box.alive
    assert box.call("total").data == 5


def test_an_unknown_method_is_a_failure_not_a_crash(interp, boxes, isolated):
    """Calling something that is not there is ordinary."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    missing = box.call("no_such_method")
    assert not missing.ok
    assert box.alive


def test_a_bare_script_can_be_a_persistent_server(interp, boxes, isolated):
    """No base class, no contract: module globals are the state."""
    box = boxes(interp, SCRATCHPAD, "", name="scratchpad", isolated=isolated)
    assert box.call("remember", key="a", value=1).data == 1
    assert box.call("remember", key="b", value=2).data == 2
    assert box.call("recall", key="a").data == 1
    assert box.call("stats").data == {"notes": 2, "calls": 3}


# ──────────────────────────────────────────────────────────────────────
# Serialization.
# ──────────────────────────────────────────────────────────────────────

def test_calls_are_serialized_per_box(interp, boxes, isolated):
    """One call at a time - which is what lets the wire skip message ids."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    threads = [threading.Thread(target=box.call, args=("add",), kwargs={"n": 1})
               for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Serialized increments cannot interleave, so none are lost.
    assert box.call("total").data == 20


# ──────────────────────────────────────────────────────────────────────
# Ending it: ask, starve, kill.
# ──────────────────────────────────────────────────────────────────────

def test_ask_stops_a_box_gracefully(interp, boxes, isolated):
    """Tier one: stop() runs, the box goes away."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    assert box.stop().ok
    assert not box.alive


def test_calling_a_stopped_box_fails_cleanly(interp, boxes, isolated):
    """Use after stop is an ordinary failure."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    box.stop()
    result = box.call("total")
    assert not result.ok
    assert "not running" in result.error


def test_stopping_twice_is_harmless(interp, boxes, isolated):
    """Teardown has to be idempotent - the watcher will double-fire."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=isolated)
    assert box.stop().data is True
    assert box.stop().data is False


def test_a_hung_call_is_starved_and_the_box_survives_in_process(interp, boxes):
    """Tier two, in-process: no kill available, so starve the call."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=False, call_timeout=0.3)
    started = time.perf_counter()
    result = box.call("hang")
    assert not result.ok
    assert "timed out" in result.error
    assert time.perf_counter() - started < 5.0


def test_a_hung_call_is_starved_in_a_subprocess(interp, boxes):
    """Tier two, subprocess: the same escalation, a harder boundary."""
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=True, call_timeout=0.5)
    started = time.perf_counter()
    result = box.call("hang")
    assert not result.ok
    assert time.perf_counter() - started < 15.0


def test_kill_is_available_only_to_the_subprocess_runner(interp, boxes):
    """The honest asymmetry: a process can be killed, a thread cannot."""
    box = boxes(interp, SERVICE, "Counter", name="counter", isolated=True)
    proc = box.proc
    box.stop()
    assert proc.poll() is not None       # actually gone


def test_a_box_that_will_not_start_raises(interp):
    """A handle to a box that never loaded is not worth handing back."""
    with pytest.raises(BoxError):
        open_box(interp, SERVICE, "NoSuchClass", name="broken",
                 isolated=True)


# ──────────────────────────────────────────────────────────────────────
# Provenance and the ledger reach resident boxes too.
# ──────────────────────────────────────────────────────────────────────

def test_requests_from_a_resident_box_carry_its_chain(tmp_path):
    """A service's Requests are gated like anyone else's."""
    seen = {}
    it = Interpreter(record=lambda chain, req, dec, res: seen.setdefault(
        "chain", chain.render()))
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    try:
        box = open_box(it, SERVICE, "Counter", name="counter", isolated=True)
        box.call("read_file", path=str(target))
        box.stop()
    finally:
        it.shutdown()
        unload_box("counter")

    assert seen["chain"].endswith("counter")
