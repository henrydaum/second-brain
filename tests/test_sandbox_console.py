"""The console: the kernel reads, the guest drains.

``input()`` is refused for three compounding reasons, and this suite is the
argument for each. It blocks — and a box takes one call at a time, so a
frontend blocked on input holds its own box and cannot render. A subprocess
box's stdin *is* the wire protocol, so reading it would eat the frames the box
talks over. And a rule that worked in-process and corrupted the protocol under
isolation is the worst kind, because nothing fails until someone sets
``isolation``.

Inverting it fixes all three at once, and buys testability: the reader takes
any iterator of lines, so none of this needs a terminal.
"""

import io
import threading
import time

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.console import CONSOLE, MAX_BUFFERED, Console
from sandbox.frontends import park, unpark

CONSOLE_FRONTEND = '''
"""A migrated console frontend."""

from guest.bases import BaseFrontend


class Term(BaseFrontend):
    """Reads the machine's console."""

    name = "term"
    uses_console = True
    ISOLATION

    def start(self, sdk):
        """Nothing to open — the kernel owns the console."""
        self._seen = []
        return True

    def poll(self, sdk):
        """Drain one line, if one arrived."""
        line = sdk.console.read_line()
        if line is None:
            return False
        self._seen.append(line)
        return True

    def render(self, sdk, session_key, kind, payload):
        """Echo messages to the console."""
        if kind == "messages":
            for text in payload or []:
                sdk.console.write(text)

    def seen(self, sdk):
        """What was read. For the tests."""
        return list(self._seen)

    def read_once(self, sdk):
        """One read, surfaced directly. For the tests."""
        return sdk.console.read_line()
'''


@pytest.fixture
def box():
    """A sandbox torn down even if a test fails."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def console():
    """A console fed from a string, never from a terminal."""
    made = Console()
    yield made
    made.stop()


def _fed(console, text: str, writer=None):
    """Claim a console with scripted input. Returns the token."""
    token = park(object())
    console.claim(token, source=io.StringIO(text), writer=writer)
    return token


# ──────────────────────────────────────────────────────────────────────
# Reading, without blocking.
# ──────────────────────────────────────────────────────────────────────

def test_lines_arrive_in_order(console):
    """The basic promise."""
    token = _fed(console, "first\nsecond\nthird\n")
    try:
        _settle(console)
        assert console.read_line() == "first"
        assert console.read_line() == "second"
        assert console.read_line() == "third"
    finally:
        unpark(token)


def test_reading_never_blocks(console):
    """The whole reason this is not ``input()``.

    A blocking read would hold the calling box, and the frontend could not
    render until the person pressed return — so agent output would appear only
    *after* the next thing they typed.
    """
    token = park(object())
    # A source that never yields: the reader thread waits, this must not.
    console.claim(token, source=_never())
    try:
        started = time.time()
        assert console.read_line() is None
        assert time.time() - started < 0.1
    finally:
        unpark(token)


def test_line_endings_are_stripped(console):
    """Both kinds, because Windows exists."""
    token = _fed(console, "unix\nwindows\r\n")
    try:
        _settle(console)
        assert console.read_line() == "unix"
        assert console.read_line() == "windows"
    finally:
        unpark(token)


def test_a_closed_console_raises_once_drained(console):
    """End of input has to be distinguishable from "nothing yet".

    Otherwise a frontend on a piped stdin idles forever instead of stopping.
    Buffered lines come out first — closing must not lose what was typed.
    """
    token = _fed(console, "last\n")
    try:
        _settle(console)
        assert console.read_line() == "last"
        with pytest.raises(EOFError):
            console.read_line()
    finally:
        unpark(token)


def test_the_buffer_is_bounded(console):
    """A pasted novel must not grow the buffer without limit."""
    token = _fed(console, "".join(f"line{i}\n" for i in range(MAX_BUFFERED + 50)))
    try:
        _settle(console)
        drained = []
        while True:
            try:
                line = console.read_line()
            except EOFError:
                break
            if line is None:
                break
            drained.append(line)
        assert len(drained) == MAX_BUFFERED
        # The oldest are dropped, so what a person typed most recently wins.
        assert drained[-1] == f"line{MAX_BUFFERED + 49}"
    finally:
        unpark(token)


# ──────────────────────────────────────────────────────────────────────
# One claimant.
# ──────────────────────────────────────────────────────────────────────

def test_a_second_frontend_cannot_take_the_console(console):
    """Two readers would split a person's keystrokes non-deterministically —
    a bug that reads as the machine dropping characters."""
    first, second = park(object()), park(object())
    try:
        assert console.claim(first, source=io.StringIO("x\n")) is True
        assert console.claim(second, source=io.StringIO("y\n")) is False
        assert console.owner == first
    finally:
        unpark(first)
        unpark(second)


def test_reclaiming_with_the_same_token_works(console):
    """A frontend must not be locked out by its own previous claim."""
    token = park(object())
    try:
        assert console.claim(token, source=io.StringIO("")) is True
        assert console.claim(token, source=io.StringIO("")) is True
    finally:
        unpark(token)


def test_releasing_frees_it_for_the_next_frontend(console):
    """Stopping a frontend has to hand the console back."""
    first, second = park(object()), park(object())
    try:
        console.claim(first, source=io.StringIO(""))
        console.release(first)
        assert console.owner == ""
        assert console.claim(second, source=io.StringIO("")) is True
    finally:
        unpark(first)
        unpark(second)


def test_a_stale_token_cannot_revoke_the_current_claim(console):
    """A frontend that already lost the console must not be able to take it
    from whoever holds it now."""
    stale, current = park(object()), park(object())
    try:
        console.claim(stale, source=io.StringIO(""))
        console.release(stale)
        console.claim(current, source=io.StringIO(""))

        console.release(stale)                 # the stale frontend stopping
        assert console.owner == current
    finally:
        unpark(stale)
        unpark(current)


# ──────────────────────────────────────────────────────────────────────
# Writing.
# ──────────────────────────────────────────────────────────────────────

def test_writing_goes_to_the_injected_writer(console):
    """Injectable, so a test never has to capture a real stdout."""
    written = []
    token = _fed(console, "", writer=written.append)
    try:
        console.write("hello")
        console.write("no newline", end="")
        assert written == ["hello\n", "no newline"]
    finally:
        unpark(token)


# ──────────────────────────────────────────────────────────────────────
# Through the Requests, from inside a real box.
# ──────────────────────────────────────────────────────────────────────

def _open(box, tmp_path, isolation=""):
    """Open a console frontend box directly."""
    source = CONSOLE_FRONTEND.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    path = tmp_path / "frontend_term.py"
    path.write_text(source, encoding="utf-8")
    return box.open(path, "Term", name="frontend_term")


@pytest.mark.parametrize("isolation", ["", "subprocess"])
def test_a_box_reads_the_console_through_a_request(box, tmp_path, isolation):
    """Both runners — and the subprocess case is the point.

    A child never opens stdin, because the *kernel* is what reads it. So a
    console frontend can be isolated, which ``input()`` could never allow: a
    child's stdin is the wire protocol it talks over.
    """
    token = park(object())
    CONSOLE.claim(token, source=io.StringIO("typed by a person\n"))
    opened = _open(box, tmp_path, isolation)
    try:
        assert opened.call("__bind__", token=token).ok
        _settle(CONSOLE)
        assert opened.call("poll").data is True
        assert opened.call("seen").data == ["typed by a person"]
    finally:
        CONSOLE.release(token)
        unpark(token)
        box.close("frontend_term")
        unload_box("frontend_term")


def test_a_box_without_the_console_is_refused(box, tmp_path):
    """A tool that imported the namespace reaches nothing."""
    token = park(object())
    opened = _open(box, tmp_path)
    try:
        assert opened.call("__bind__", token=token).ok
        result = opened.call("read_once")
        assert not result.ok
        assert "belongs to another frontend, or to none" in result.error
    finally:
        unpark(token)
        box.close("frontend_term")
        unload_box("frontend_term")


def test_writing_from_a_box_reaches_the_console(box, tmp_path):
    """The render half, end to end."""
    written = []
    token = park(object())
    CONSOLE.claim(token, source=io.StringIO(""), writer=written.append)
    opened = _open(box, tmp_path)
    try:
        assert opened.call("__bind__", token=token).ok
        assert opened.call("render", session_key="default", kind="messages",
                           payload=["from the box"]).ok
        assert written == ["from the box\n"]
    finally:
        CONSOLE.release(token)
        unpark(token)
        box.close("frontend_term")
        unload_box("frontend_term")


@pytest.mark.parametrize("text", [
    "| a | bbbb |\n| --- | --- |\n| 1 | 2 |",
    "text\n\n| Name | Value |\n|---|---|\n| x | 1 |\n| yy | 22 |\n\nafter",
    "```python\ncode here\n```",
    "plain text only",
    r"| a \| b | c |" + "\n|---|---|\n| 1 | 2 |",
    "",
])
def test_the_guest_renderer_matches_the_kernels(text):
    """``sdk.md.plain`` is the guest's copy of the kernel's ``render_plain``.

    A sandboxed frontend cannot import kernel helpers, so the code had to be
    duplicated — and duplicated code drifts. Pinning them against each other is
    what makes the copy safe: the REPL's output must not change shape because
    it moved into a box.
    """
    from guest.sdk import SDK
    from plugins.frontends.helpers.formatters import render_plain

    assert SDK(None).md.plain(text) == render_plain(text)


def test_input_is_still_refused_by_the_validator():
    """Providing a route must not quietly open the one that blocks."""
    from sandbox.validator import validate

    report = validate("def go(sdk):\n    return input()\n", filename="s.py")
    assert not report.ok
    assert "sdk.console.read_line" in report.render()


# ──────────────────────────────────────────────────────────────────────

def _never():
    """A source that yields nothing and does not end."""
    stop = threading.Event()
    while not stop.wait(0.05):
        yield  # pragma: no cover - never reached; the test does not wait


def _settle(console, timeout=1.0):
    """Wait for the reader thread to have drained its source.

    The reader is a thread, so 'the line has arrived' is not instant. Polling
    for it beats sleeping a guessed interval.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        with console._lock:
            if console._lines or console._closed:
                return
        time.sleep(0.005)


from sandbox.console import Console

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
