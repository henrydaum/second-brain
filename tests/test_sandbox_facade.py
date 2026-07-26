"""The front door: one API, declaration-driven, blocking or not."""

import threading
import time
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import BoxError, Sandbox
from sandbox.guest.box import SUBPROCESS

FIXTURES = Path(__file__).parent / "fixtures"
SCRIPT = FIXTURES / "scratch_script.py"
TOOL = FIXTURES / "tool_wordcount.py"
SERVICE = FIXTURES / "service_counter.py"
SCRATCHPAD = FIXTURES / "scratchpad_server.py"


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
        "    return sdk.ok(sdk.fs.read(path).data)\n", encoding="utf-8")
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
