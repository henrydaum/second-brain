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


def test_persistent_returns_obey_the_plain_data_boundary(interp, boxes,
                                                         isolated):
    box = boxes(interp, SERVICE, "Counter", name="counter_values",
                isolated=isolated)
    assert box.call("canonical").data == {"pair": [1, 2], "blob": b"ab"}
    assert box.call("reserved").data == {"__bytes__": "AA=="}
    refused = box.call("live")
    assert not refused.ok
    assert "live or non-serializable" in refused.error
    assert box.alive
    assert box.call("total").data == 0


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


def test_a_hung_call_starves_and_ends_the_in_process_box(interp, boxes):
    """Tier two, in-process: no kill available, so starve the call.

    Starving ends the *box*, not just the call. Cancellation is per-execution
    and a resident box has exactly one for its whole life, so there is no way
    back: every later call would answer ``Terminated`` at once, and the
    starved worker is still alive, so reusing the box would put two threads on
    one execution. A box that reported itself healthy after this is how a
    starved REPL went silently deaf instead of stopping.
    """
    box = boxes(interp, SERVICE, "Counter", name="counter",
                isolated=False, call_timeout=0.3)
    started = time.perf_counter()
    result = box.call("hang")
    assert not result.ok
    assert "timed out" in result.error
    assert time.perf_counter() - started < 5.0
    assert not box.alive
    assert not box.call("bump").ok


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


# ────────────────────────────────────────────────────────────────────
# The facade over the same lifecycle (was test_sandbox_facade.py)
# ────────────────────────────────────────────────────────────────────

from sandbox import BoxError, Sandbox
from sandbox.guest.box import SUBPROCESS


# FIXTURES / SCRIPT / SERVICE / SCRATCHPAD are already defined at the top of
# this file, with the same values; only TOOL is new here.
TOOL = FIXTURES / "tool_wordcount.py"


@pytest.fixture
def sb():
    """A sandbox that refuses everything unsafe."""
    box = Sandbox()
    yield box
    box.shutdown()


@pytest.fixture(autouse=True)
def clean_boxes():
    """Module caches are per-box; leaking one hides staleness."""
    yield
    for name in ("scratch_script", "wordcount", "counter", "scratchpad",
                 "slow", "declares", "loud", "broken", "spinner"):
        unload_box(name)


@pytest.fixture
def note(tmp_path):
    """A small file to read."""
    target = tmp_path / "note.txt"
    target.write_text("one two three", encoding="utf-8")
    return str(target)


# ──────────────────────────────────────────────────────────────────────
# One call does the whole sequence.
# ──────────────────────────────────────────────────────────────────────

def test_run_needs_nothing_but_a_file(sb, note):
    """No runner choice, no box name, no timeout - it is all declared."""
    result = sb.run(SCRIPT, "summarize", kwargs={"path": note})
    assert result.ok
    assert result.data["words"] == 3


def test_declarations_are_read_without_importing(sb):
    """The file says what it wants; the kernel reads it with ast."""
    report, spec = sb.inspect(TOOL)
    assert report.ok
    assert report.declarations["name"] == "word_count"
    assert report.declarations["requests"] == ["fs.read"]
    assert spec.name == "wordcount"


def test_a_declared_isolation_picks_the_runner(sb, tmp_path, note):
    """isolation = subprocess means a subprocess, without the caller asking."""
    script = tmp_path / "declares.py"
    script.write_text(
        'isolation = "subprocess"\n\n'
        "def go(sdk, path):\n"
        "    return sdk.fs.read(path)\n", encoding="utf-8")
    report, spec = sb.inspect(script)
    assert spec.isolation == SUBPROCESS
    assert sb.run(script, "go", kwargs={"path": note}).data == "one two three"


def test_an_invalid_file_is_never_executed(sb, tmp_path):
    """Validation is a gate on running, not just advice."""
    marker = tmp_path / "ran.txt"
    script = tmp_path / "broken.py"
    script.write_text(
        "def go(sdk):\n"
        f"    open({str(marker)!r}, 'w').write('x')\n"
        "    return sdk.ok(1)\n", encoding="utf-8")

    with pytest.raises(BoxError) as exc:
        sb.run(script, "go")
    assert "sdk.fs" in str(exc.value)
    assert not marker.exists()


def test_a_disclaimed_file_still_runs(sb, tmp_path, caplog):
    """A foreign library warns; it does not block."""
    script = tmp_path / "loud.py"
    script.write_text(
        "import json\n\ndef go(sdk):\n    return sdk.ok(1)\n",
        encoding="utf-8")
    assert sb.run(script, "go").data == 1


def test_a_persistent_declaration_refuses_to_be_run(sb):
    """Lifetime is decided before the run, and run() is the wrong door."""
    with pytest.raises(BoxError) as exc:
        sb.run(SERVICE, "Counter")
    assert "resident" in str(exc.value)


# ──────────────────────────────────────────────────────────────────────
# Blocking and non-blocking: the wait=True / wait=False shapes.
# ──────────────────────────────────────────────────────────────────────

def test_start_returns_before_the_work_finishes(sb, tmp_path):
    """wait=False: the caller keeps going while the work continues."""
    script = tmp_path / "slow.py"
    script.write_text(
        "def go(sdk):\n"
        "    total = 0\n"
        "    for _ in range(40):\n"
        "        sdk.fs.list('.')\n"
        "        total += 1\n"
        "    return sdk.ok(total)\n", encoding="utf-8")

    started = time.perf_counter()
    run = sb.start(script, "go")
    handed_back = time.perf_counter() - started

    result = run.wait(timeout=30)
    assert result.ok and result.data == 40
    assert handed_back < 0.5      # returned long before the work was done
    assert run.done


def test_wait_can_be_called_more_than_once(sb, note):
    """A handle is not consumed by looking at it."""
    run = sb.start(SCRIPT, "summarize", kwargs={"path": note})
    first = run.wait(timeout=30)
    second = run.wait(timeout=30)
    assert first.data == second.data


def test_waiting_with_a_short_timeout_reports_still_running(sb, tmp_path):
    """Polling must not lie about having a result."""
    script = tmp_path / "slow.py"
    script.write_text(
        "def go(sdk):\n"
        "    for _ in range(200):\n"
        "        sdk.fs.list('.')\n"
        "    return sdk.ok(1)\n", encoding="utf-8")
    run = sb.start(script, "go")
    early = run.wait(timeout=0.01)
    if not early.ok:
        assert "still running" in early.error
        assert early.retryable
    assert run.wait(timeout=30).ok


def test_on_done_fires_with_the_result(sb, note):
    """How a spawner queues a completion notice back to its session."""
    landed = threading.Event()
    box = {}

    def finished(result):
        """Receive the result."""
        box["result"] = result
        landed.set()

    sb.start(SCRIPT, "summarize", kwargs={"path": note}, on_done=finished)
    assert landed.wait(timeout=30)
    assert box["result"].data["words"] == 3


def test_a_failing_on_done_does_not_break_the_run(sb, note):
    """A bad callback is the caller's problem, not the sandbox's."""
    def explode(result):
        """Misbehave."""
        raise RuntimeError("callback is broken")

    run = sb.start(SCRIPT, "summarize", kwargs={"path": note},
                   on_done=explode)
    assert run.wait(timeout=30).ok


def test_background_runs_can_be_cancelled(sb, tmp_path):
    """The counterpart to starting one: stopping it."""
    script = tmp_path / "spinner.py"
    script.write_text(
        'isolation = "subprocess"\n\n'
        "def go(sdk):\n"
        "    while True:\n"
        "        sdk.fs.list('.')\n", encoding="utf-8")

    run = sb.start(script, "go")
    time.sleep(0.3)
    run.cancel()
    result = run.wait(timeout=30)
    assert run.cancelled
    assert not result.ok


def test_run_is_start_plus_wait(sb, note):
    """The blocking call is sugar, not a second implementation."""
    direct = sb.run(SCRIPT, "summarize", kwargs={"path": note})
    staged = sb.start(SCRIPT, "summarize", kwargs={"path": note}).wait(30)
    assert direct.data == staged.data


# ──────────────────────────────────────────────────────────────────────
# Resident boxes are tracked.
# ──────────────────────────────────────────────────────────────────────

def test_open_tracks_the_box(sb):
    """An untracked resident box is an orphan after a restart."""
    box = sb.open(SERVICE, "Counter")
    assert sb.box("counter") is box
    assert box in sb.boxes()


def test_opening_twice_reuses_the_live_box(sb):
    """Loading a service twice would double its memory and split its state."""
    first = sb.open(SERVICE, "Counter")
    first.call("add", n=5)
    second = sb.open(SERVICE, "Counter")
    assert second is first
    assert second.call("total").data == 5


def test_close_stops_and_forgets(sb):
    """After closing, the name is free and the box is gone."""
    sb.open(SERVICE, "Counter")
    assert sb.close("counter").ok
    assert sb.box("counter") is None
    assert sb.boxes() == []


def test_closing_something_that_is_not_open_is_harmless(sb):
    """Teardown paths double-fire; they must be idempotent."""
    assert sb.close("counter").data is False


def test_a_bare_script_opens_as_a_resident_server(sb):
    """No class, no contract: module globals are the state."""
    box = sb.open(SCRATCHPAD, "", name="scratchpad")
    box.call("remember", key="a", value=1)
    box.call("remember", key="b", value=2)
    assert box.call("stats").data["notes"] == 2


def test_shutdown_closes_everything(sb):
    """The reason the sandbox tracks what it opened."""
    sb.open(SERVICE, "Counter")
    sb.open(SCRATCHPAD, "", name="scratchpad")
    assert len(sb.boxes()) == 2
    sb.shutdown()
    assert sb.boxes() == []


def test_shutdown_cancels_background_runs(tmp_path):
    """A restart must not leave work running behind it."""
    box = Sandbox()
    script = tmp_path / "spinner.py"
    script.write_text(
        'isolation = "subprocess"\n\n'
        "def go(sdk):\n"
        "    while True:\n"
        "        sdk.fs.list('.')\n", encoding="utf-8")
    run = box.start(script, "go")
    time.sleep(0.3)
    box.shutdown()
    assert run.wait(timeout=30) is not None
    assert run.cancelled


def test_a_family_default_is_visible_without_importing(sb):
    """Inherited defaults are invisible to ast, so the family supplies them.

    ``class Counter(BaseService)`` never writes ``lifetime`` in the file, but
    a service is persistent by definition and the base class *is* named in the
    source.
    """
    report, spec = sb.inspect(SERVICE)
    assert report.declarations["family"] == "service"
    assert spec.persistent


def test_the_file_overrides_its_family_default(sb, tmp_path):
    """Defaults fill gaps; they never overrule what was written."""
    script = tmp_path / "service_transient.py"
    script.write_text(
        "from guest.bases import BaseService\n\n\n"
        "class Transient(BaseService):\n"
        '    """Declines to be resident."""\n'
        '    name = "transient"\n'
        '    lifetime = "ephemeral"\n\n'
        "    def start(self, sdk):\n"
        '        """Start."""\n'
        "        return True\n", encoding="utf-8")
    report, spec = sb.inspect(script)
    assert report.ok, report.render()
    assert not spec.persistent


from types import SimpleNamespace
from sandbox import Sandbox, provenance
from sandbox.bridge import adapt, configure
from sandbox.console import Console
from sandbox.guest.requests import Request, Result
from sandbox.interpreter import Execution, Interpreter
from sandbox.policy import SAFE, UNSAFE, Chain, Decision, classify

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
# The slowest wait in the system is a person reading a dialog.
# ──────────────────────────────────────────────────────────────────────

def test_a_slow_approval_is_not_charged_to_the_guest():
    """Reading the dialog is waiting on the kernel like any other wait.

    The accounting above was applied to handlers and not to the approval leg,
    so the dialog — which may sit for DIALOG_TIMEOUT, ten times the default
    deadline — burned the guest's whole budget. Every unsafe Request a person
    thought about failed with a timeout blaming the plugin, and the tool the
    user *had* just approved reported that it had not run.
    """
    asked = []

    def approve(chain, request, decision):
        """A person who takes longer than the whole deadline to answer."""
        asked.append(request.type)
        time.sleep(0.5)
        return True

    it = Interpreter(approve=approve)
    try:
        def plugin(sdk):
            """The Request the bug was reported against, answered slowly."""
            try:
                sdk.agent.schedule("do the thing", "0 9 * * *")
            except sdk.Failed:
                pass         # no timekeeper here; the answer is not the point
            return "finished"

        result = run_in_process(it, plugin, name="patient", timeout=0.3)
    finally:
        it.shutdown()

    assert asked == ["agent.schedule"], "the dialog has to have been reached"
    assert result.ok, result.error
    assert result.data == "finished"


def test_an_approval_that_lands_after_cancellation_does_not_take_effect(
        monkeypatch):
    """A yes given to a caller that has already been told "no" must not run.

    Cancelling only set a flag the gate reads on the way *in*, and an approved
    Request is long past the gate — so the handler ran anyway. Observed as a
    scheduled subagent that timed out, was created regardless, and whose retry
    reported that the job already existed.
    """
    from sandbox import interpreter as interpreter_module

    ran = []
    monkeypatch.setitem(
        interpreter_module.HANDLERS, "config.write",
        lambda ctx, args: ran.append(args) or Result(data=True))

    execution = Execution(name="abandoned", chain=Chain())
    answered = threading.Event()

    def approve(chain, request, decision):
        """Answer yes, but only once the caller has given up waiting."""
        answered.wait(3.0)
        return True

    it = Interpreter(approve=approve)
    try:
        def plugin(sdk):
            """One unsafe Request that will be abandoned mid-dialog."""
            try:
                return sdk.config.write("probe_setting", 1)
            except BaseException:
                return "unwound"

        def cancel_soon():
            """Play the deadline expiring while the dialog is still up."""
            time.sleep(0.3)
            it.cancel(execution)
            answered.set()

        threading.Thread(target=cancel_soon, daemon=True).start()
        run_in_process(it, plugin, name="abandoned", timeout=5,
                       execution=execution)
    finally:
        it.shutdown()

    assert ran == [], "the handler ran for a caller that had already given up"
