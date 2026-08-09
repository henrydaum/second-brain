"""Subprocess execution, and the property the whole design rests on: the same
plugin file behaves identically under both runners.

The security contract's claim is that *the code functions the same either way,
just with different levels of security*. These tests are what makes that a
fact rather than an intention.
"""

import importlib.util
import time
from pathlib import Path

import pytest

from sandbox import Chain, Interpreter, Result, run_in_process
from sandbox.runner_subprocess import run_in_subprocess

FIXTURE = Path(__file__).parent / "fixtures" / "sandbox_plugin.py"


def _load_fixture():
    """Import the fixture in-process so both runners get the same code."""
    spec = importlib.util.spec_from_file_location("sandbox_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def interp():
    """An interpreter that refuses everything unsafe."""
    it = Interpreter()
    yield it
    it.shutdown()


@pytest.fixture
def both(interp):
    """Run one fixture function under each runner and return both Results."""
    module = _load_fixture()

    def run(func_name, **kwargs):
        """Execute under in-process and subprocess runners."""
        in_proc = run_in_process(
            interp, getattr(module, func_name),
            name=func_name, kwargs=kwargs, timeout=30)
        sub = run_in_subprocess(
            interp, str(FIXTURE), func_name,
            name=func_name, kwargs=kwargs, timeout=30)
        return in_proc, sub

    return run


def _same(a: Result, b: Result):
    """Assert two Results are equivalent in every observable way."""
    assert a.ok == b.ok, f"ok differs: {a} vs {b}"
    assert a.data == b.data, f"data differs: {a.data!r} vs {b.data!r}"
    assert a.denied == b.denied, f"denied differs: {a} vs {b}"


# ──────────────────────────────────────────────────────────────────────
# The core claim.
# ──────────────────────────────────────────────────────────────────────

def test_read_is_identical(both, tmp_path):
    """A successful read behaves the same on both runners."""
    target = tmp_path / "notes.md"
    target.write_text("hello sandbox", encoding="utf-8")
    a, b = both("read_and_truncate", path=str(target), limit=5)
    _same(a, b)
    assert a.data == "he..."


def test_helper_requests_are_identical(both, tmp_path):
    """A plain helper making Requests works the same on both."""
    target = tmp_path / "a.txt"
    target.write_text("A", encoding="utf-8")
    a, b = both("via_helper", path=str(target))
    _same(a, b)
    assert a.data == "A"


def test_egress_is_refused_on_both(both):
    """Policy is enforced by the one gate, so both runners refuse alike."""
    a, b = both("attempt_egress")
    _same(a, b)
    assert a.data["denied"] is True


def test_denial_is_survivable_on_both(both, tmp_path):
    """A refusal is an ordinary failure in both runners - not a kill."""
    target = tmp_path / "ok.txt"
    target.write_text("still here", encoding="utf-8")
    a, b = both("survives_denial", path=str(target))
    _same(a, b)
    assert a.data == "still here"


def test_respond_terminates_on_both(both):
    """Asking to end actually ends it, identically."""
    a, b = both("responds_early")
    _same(a, b)
    assert a.data == "early"


def test_unhandled_errors_become_failures_on_both(both):
    """A plugin that breaks yields a failure Result, never a crash."""
    a, b = both("raises")
    assert not a.ok and not b.ok
    assert "something went wrong" in a.error
    assert "something went wrong" in b.error


def test_missing_file_is_reported_the_same(both):
    """The failure path matches too, not just the happy path."""
    a, b = both("read_and_truncate", path="does/not/exist.txt")
    _same(a, b)
    assert not a.ok


def test_canonical_values_are_identical_on_both(both):
    a, b = both("returns_canonical_values")
    _same(a, b)
    assert a.data == {"pair": [1, 2], "blob": b"ab"}


def test_reserved_byte_tag_is_data_on_both(both):
    a, b = both("returns_reserved_tag_dict")
    _same(a, b)
    assert a.data == {"__bytes__": "AA=="}


@pytest.mark.parametrize("function", ["returns_live_object", "sends_live_object"])
def test_live_objects_fail_on_both_sides_of_the_boundary(both, function):
    a, b = both(function)
    assert not a.ok and not b.ok
    assert "live or non-serializable" in a.error
    assert "live or non-serializable" in b.error


@pytest.mark.parametrize("function", ["returns_oversized_value",
                                      "sends_oversized_request"])
def test_oversized_payloads_fail_on_both(both, function):
    a, b = both(function)
    assert not a.ok and not b.ok
    assert "exceeds" in a.error
    assert "exceeds" in b.error


# ──────────────────────────────────────────────────────────────────────
# Properties specific to the subprocess boundary.
# ──────────────────────────────────────────────────────────────────────

def test_plugin_stdout_does_not_corrupt_the_wire(interp):
    """Plugin prints go to stderr so the protocol stream stays clean."""
    result = run_in_subprocess(interp, str(FIXTURE), "prints_to_stdout",
                               name="printer", timeout=30)
    assert result.ok
    assert result.data == "survived"


def test_a_runaway_is_actually_killed(interp):
    """The real win over threads: a process can be stopped."""
    started = time.perf_counter()
    result = run_in_subprocess(interp, str(FIXTURE), "spins",
                               name="runaway", timeout=1.0)
    elapsed = time.perf_counter() - started
    assert not result.ok
    assert "timed out" in result.error
    assert elapsed < 15.0


def test_waiting_on_the_kernel_is_not_charged_to_the_guest():
    """A deadline measures *guest* execution, and this runner was the last
    place still measuring wall clock.

    The child here does nothing slow: it makes one unsafe Request and waits
    while a person reads the dialog. That wait belongs to the kernel — it is
    the kernel that has not answered yet — so charging it killed every
    subprocess plugin that asked anything, at its declared timeout, and the
    report blamed the plugin. The identical code in-process was never killed,
    which is the asymmetry ``watchdog.overdue`` exists to remove.

    Its twin is ``test_a_runaway_is_actually_killed`` above: code that is
    genuinely spinning still runs out its deadline."""
    def slow_approver(chain, request, decision):
        """A dialog somebody takes their time over."""
        time.sleep(2.5)
        return False

    interp = Interpreter(approve=slow_approver)
    try:
        result = run_in_subprocess(interp, str(FIXTURE), "attempt_egress",
                                   name="patient", timeout=1.0)
    finally:
        interp.shutdown()

    assert result.ok, result.error
    assert result.data["denied"] is True


def test_missing_function_faults_cleanly(interp):
    """A bad entry point is reported, not hung on."""
    result = run_in_subprocess(interp, str(FIXTURE), "no_such_function",
                               name="missing", timeout=30)
    assert not result.ok
    assert "no_such_function" in result.error


def test_missing_file_faults_cleanly(interp):
    """So is a bad module path."""
    result = run_in_subprocess(interp, "does/not/exist.py", "anything",
                               name="missing", timeout=30)
    assert not result.ok


def test_provenance_reaches_the_dialog_from_a_subprocess(tmp_path):
    """The chain is kernel-side, so it survives the process boundary."""
    seen = {}

    def approve(chain, request, decision):
        """Record what the user would have been shown."""
        seen["chain"] = chain.render()
        return False

    it = Interpreter(approve=approve)
    try:
        run_in_subprocess(it, str(FIXTURE), "attempt_egress",
                          name="fetcher", timeout=30,
                          chain=Chain(root="cron:nightly_index"))
    finally:
        it.shutdown()

    assert seen["chain"] == "cron:nightly_index -> fetcher"


def test_requests_are_ledgered_from_a_subprocess(tmp_path):
    """Both runners feed the same ledger sink."""
    target = tmp_path / "x.txt"
    target.write_text("x", encoding="utf-8")
    rows = []

    it = Interpreter(record=lambda chain, req, dec, res, ctx=None: rows.append(
        (chain.render(), req.type, dec.level, res.ok)))
    try:
        run_in_subprocess(it, str(FIXTURE), "read_and_truncate",
                          name="reader", timeout=30,
                          kwargs={"path": str(target)})
    finally:
        it.shutdown()

    assert rows
    chain, req_type, level, ok = rows[0]
    assert req_type == "fs.read"
    assert chain.endswith("reader")
    assert ok


# ──────────────────────────────────────────────────────────────────────
# Library logging inside a child.
#
# Plugin code may not import ``logging`` — ``sdk.log`` is the route — but the
# libraries it imports have no such rule, and a child inherits stderr rather
# than piping it. A transport library retrying a dead network logs a full
# traceback every cycle, so without this an overnight outage costs the whole
# terminal.
# ──────────────────────────────────────────────────────────────────────

def _record(message: str, line: int = 10, level=None):
    """One error record from a fixed call site."""
    import logging

    return logging.LogRecord("lib.updater", level or logging.ERROR, "lib.py",
                             line, message, None, None)


def test_repeated_errors_from_one_site_collapse():
    from sandbox.guest.child import _CollapseRepeats

    throttle = _CollapseRepeats(cooldown=1000.0)

    assert throttle.filter(_record("boom")) is True
    assert [throttle.filter(_record("boom")) for _ in range(20)] == [False] * 20


def test_the_backlog_is_counted_when_the_cooldown_lapses():
    """Silence that never says what it swallowed is indistinguishable from
    the problem having stopped."""
    from sandbox.guest.child import _CollapseRepeats

    throttle = _CollapseRepeats(cooldown=0.05)
    throttle.filter(_record("boom"))
    for _ in range(7):
        assert throttle.filter(_record("boom")) is False

    time.sleep(0.06)
    record = _record("boom")
    assert throttle.filter(record) is True
    assert "+7 more like it" in record.getMessage()


def test_the_traceback_is_dropped_but_the_message_is_not():
    """The hundredth copy of a stack is not more informative than the first."""
    from sandbox.guest.child import _CollapseRepeats

    record = _record("boom")
    try:
        raise OSError("getaddrinfo failed")
    except OSError:
        import sys

        record.exc_info = sys.exc_info()

    assert _CollapseRepeats().filter(record) is True
    assert record.exc_info is None
    assert record.getMessage() == "boom"


def test_different_sites_do_not_silence_each_other():
    from sandbox.guest.child import _CollapseRepeats

    throttle = _CollapseRepeats(cooldown=1000.0)

    assert throttle.filter(_record("boom", line=10)) is True
    assert throttle.filter(_record("boom", line=99)) is True
    assert throttle.filter(_record("different", line=10)) is True


def test_a_varying_message_still_collapses():
    """Keyed on the template, so a retry counter in the args does not defeat it."""
    import logging

    from sandbox.guest.child import _CollapseRepeats

    throttle = _CollapseRepeats(cooldown=1000.0)

    def attempt(n):
        return logging.LogRecord("lib.updater", logging.ERROR, "lib.py", 10,
                                 "retry %d failed", (n,), None)

    assert throttle.filter(attempt(1)) is True
    assert throttle.filter(attempt(2)) is False


def test_warnings_are_never_collapsed():
    """A library at WARNING is not looping, and one at DEBUG was asked for."""
    import logging

    from sandbox.guest.child import _CollapseRepeats

    throttle = _CollapseRepeats(cooldown=1000.0)
    for _ in range(5):
        assert throttle.filter(
            _record("noisy", level=logging.WARNING)) is True


def test_configuring_the_child_logger_is_idempotent_and_deferential():
    import logging

    from sandbox.guest.child import _CollapseRepeats, _tame_library_logging

    root = logging.getLogger()
    saved, saved_level = list(root.handlers), root.level
    try:
        root.handlers = []
        _tame_library_logging()
        _tame_library_logging()
        assert len(root.handlers) == 1
        assert any(isinstance(f, _CollapseRepeats)
                   for f in root.handlers[0].filters)

        # A child that already has handlers keeps them: this configures a
        # bare process, it does not take over one somebody else set up.
        existing = logging.NullHandler()
        root.handlers = [existing]
        _tame_library_logging()
        assert root.handlers == [existing]
    finally:
        root.handlers, root.level = saved, saved_level


# ────────────────────────────────────────────────────────────────────
# base64 is invisible to the plugin (was test_sandbox_bytes.py)
# ────────────────────────────────────────────────────────────────────

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from sandbox import Sandbox
from sandbox.guest import requests as R
from sandbox.policy import Chain, classify


PAYLOAD = bytes([0x89, 0x50, 0x4E, 0x47, 0x00, 0xFF, 0xFE, 0x0D, 0x0A, 0x1A])

ROUND_TRIP = '''
"""Reads bytes, writes them back, and reports what it saw."""

ISOLATION

from guest.bases import BaseTool


class Copy(BaseTool):
    """Copy a file byte for byte."""

    name = "copy"
    description = "Round-trip a file through read_bytes/write_bytes."

    def run(self, sdk, src, dst):
        """Copy and report the length and the first byte."""
        data = sdk.fs.read_bytes(src)
        sdk.fs.write_bytes(dst, data)
        return {"length": len(data), "first": data[0], "is_bytes": isinstance(data, bytes)}
'''


@pytest.fixture
def box():
    """A sandbox torn down even if a test fails."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def tool(tmp_path, request):
    """The round-tripping tool, optionally subprocess-isolated."""
    isolation = getattr(request, "param", "")
    source = ROUND_TRIP.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    path = tmp_path / "tool_copy.py"
    path.write_text(source, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────
# The round trip.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tool", ["", "subprocess"], indirect=True)
def test_bytes_survive_the_crossing_unchanged(box, tool, tmp_path):
    """The claim: what went in is what came out, in either runner."""
    import trees

    src = tmp_path / "in.png"
    # Reads are safe anywhere; writes are judged by where, so the copy has to
    # land in scratch or the run is (correctly) denied.
    dst = trees.tree("workspace").path / "temp" / "sb-test-copy.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    src.write_bytes(PAYLOAD)

    result = box.run(tool, "Copy", kwargs={"src": str(src), "dst": str(dst)})

    assert result.ok, result.error
    assert result.data["length"] == len(PAYLOAD)
    assert result.data["first"] == PAYLOAD[0]
    # The guest sees bytes, never the base64 the wire carried.
    assert result.data["is_bytes"] is True
    assert dst.read_bytes() == PAYLOAD


def test_text_read_would_have_corrupted_it(tmp_path):
    """Why the Request exists at all, stated as a test rather than a comment."""
    path = tmp_path / "in.png"
    path.write_bytes(PAYLOAD)
    mangled = path.read_text(encoding="utf-8", errors="replace")
    assert mangled.encode("utf-8", errors="replace") != PAYLOAD


# ──────────────────────────────────────────────────────────────────────
# Policy: the encoding must not be a way around the rule.
# ──────────────────────────────────────────────────────────────────────

def test_reading_bytes_is_as_safe_as_reading_text():
    """Both are reads, and reads are safe because egress is not."""
    request = R.Request(R.FS_READ_BYTES, {"path": "x"})
    assert classify(request, Chain()).safe
    assert request.read_only


def test_writing_bytes_is_judged_by_where_exactly_like_text(tmp_path):
    """Same act, different encoding — so it must get the same answer."""
    import tempfile
    import trees
    scratch = trees.tree("workspace").path / "temp" / "sb-test-bytes.bin"
    assert classify(R.Request(R.FS_WRITE_BYTES, {"path": str(scratch), "data": "eA=="}),
                    Chain()).safe
    foreign_temp = Path(tempfile.gettempdir()) / "another-program.bin"
    assert not classify(
        R.Request(R.FS_WRITE_BYTES, {"path": str(foreign_temp), "data": "eA=="}),
        Chain()).safe
    assert not classify(
        R.Request(R.FS_WRITE_BYTES, {"path": "main.pyw", "data": "eA=="}),
        Chain()).safe


# ──────────────────────────────────────────────────────────────────────
# Handler edges.
# ──────────────────────────────────────────────────────────────────────

def test_junk_base64_is_refused_rather_than_truncated(tmp_path):
    """Silently dropping invalid characters would write a short file."""
    from sandbox.handlers.fs_net import _fs_write_bytes

    target = tmp_path / "out.bin"
    result = _fs_write_bytes(None, {"path": str(target), "data": "not!base64!"})

    assert not result.ok
    assert "base64" in result.error
    assert not target.exists()


def test_an_oversized_file_is_refused(tmp_path, monkeypatch):
    """The cap bounds one wire frame, and base64 inflates by 4/3."""
    from sandbox.handlers import fs_net

    monkeypatch.setattr(fs_net, "MAX_READ_BINARY", 4)
    path = tmp_path / "big.bin"
    path.write_bytes(b"12345")

    result = fs_net._fs_read_bytes(None, {"path": str(path)})

    assert not result.ok
    # The refusal has to name the way out, or the only route to a file bigger
    # than one frame is undiscoverable from the error that blocks it.
    assert "offset=" in result.error and "length=" in result.error


def test_the_binary_cap_fits_inside_one_wire_message(tmp_path):
    """Derived, not guessed.

    It was 32 MB against a 16 MB frame, so every read between the two limits
    passed the check and then died in ``protocol.encode`` as an unsendable
    result — a fault where the caller had earned an ordinary failure.
    """
    from sandbox.guest import protocol
    from sandbox.handlers.fs_net import MAX_READ_BINARY

    encoded = (MAX_READ_BINARY + 2) // 3 * 4
    assert encoded < protocol.MAX_MESSAGE_BYTES


def test_a_window_reads_exactly_what_it_asked_for(tmp_path):
    """``offset``/``length`` are how a file larger than a frame is read at all."""
    import base64

    from sandbox.handlers.fs_net import _fs_read_bytes

    path = tmp_path / "payload.bin"
    payload = bytes(range(256))
    path.write_bytes(payload)

    result = _fs_read_bytes(None, {"path": str(path), "offset": 10,
                                   "length": 5})

    assert result.ok
    assert base64.b64decode(result.data) == payload[10:15]


def test_windows_reassemble_a_file_bigger_than_the_cap(tmp_path, monkeypatch):
    """The loop a plugin writes, against a cap small enough to force it."""
    import base64

    from sandbox.handlers import fs_net

    monkeypatch.setattr(fs_net, "MAX_READ_BINARY", 16)
    path = tmp_path / "long.bin"
    payload = bytes(range(256)) * 3
    path.write_bytes(payload)

    chunks, offset = [], 0
    while True:
        result = fs_net._fs_read_bytes(None, {"path": str(path),
                                              "offset": offset, "length": 16})
        assert result.ok, result.error
        chunk = base64.b64decode(result.data)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)

    assert b"".join(chunks) == payload


def test_a_window_past_the_end_answers_empty(tmp_path):
    """What ends the loop above. Returning the whole file here would not."""
    from sandbox.handlers.fs_net import _fs_read_bytes

    path = tmp_path / "short.bin"
    path.write_bytes(b"abc")

    result = _fs_read_bytes(None, {"path": str(path), "offset": 99})

    assert result.ok
    assert result.data == ""


def test_a_window_over_the_cap_is_refused_like_a_whole_file(tmp_path,
                                                            monkeypatch):
    """The cap is on one answer, so asking for a big window is the same ask."""
    from sandbox.handlers import fs_net

    monkeypatch.setattr(fs_net, "MAX_READ_BINARY", 8)
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * 100)

    result = fs_net._fs_read_bytes(None, {"path": str(path), "length": 64})

    assert not result.ok
    assert "offset=" in result.error


def test_append_mode_adds_rather_than_replaces(tmp_path):
    """``mode="append"`` opens 'ab', so a second write extends the file."""
    from sandbox.handlers.fs_net import _fs_write_bytes
    import base64

    target = tmp_path / "log.bin"
    for chunk in (b"\x00\x01", b"\x02\x03"):
        _fs_write_bytes(None, {"path": str(target),
                               "data": base64.b64encode(chunk).decode(),
                               "mode": "append"})

    assert target.read_bytes() == b"\x00\x01\x02\x03"


def test_every_result_extra_survives_the_child(both):
    """A field forgotten in ``to_dict`` is lost here and nowhere else.

    The in-process runner hands the Result back by reference, so it cannot
    notice a serialization gap. This is the only runner that can.
    """
    in_proc, sub = both("returns_every_extra")
    assert in_proc.ok and sub.ok
    assert sub.data == {"n": 1}
    assert sub.llm_summary == in_proc.llm_summary == "a summary"
    assert sub.attachment_paths == in_proc.attachment_paths == ["/tmp/a.png"]
    assert sub.also_contains == in_proc.also_contains == ["nested"]
    assert sub.discovered_paths == in_proc.discovered_paths == ["/tmp/found"]
