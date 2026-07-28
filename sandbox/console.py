"""The machine's console, owned by the kernel and lent to one frontend.

Sandboxed code cannot read a terminal, and the reasons compound. ``input()``
blocks, and a box takes one call at a time — so a frontend blocked on input
holds its own box and cannot render, which means agent output would only appear
*after* the next thing you typed. Worse, a subprocess box's stdin **is** the
wire protocol: reading it would eat the frames the box is talking over. A rule
that worked in-process and corrupted the protocol under isolation is the worst
kind, because nothing fails until someone sets ``isolation``.

So the console inverts exactly like the poll loop did: **the host reads, the
guest drains.** One thread here pulls lines off stdin into a buffer;
``console.read_line`` takes what has arrived and returns immediately, so a
frontend never blocks and renders keep landing between polls. The child process
never opens stdin at all, which means a sandboxed REPL can run *more* isolated
than the native one rather than less.

**One claimant.** Two frontends reading one stdin would split a person's
keystrokes between them non-deterministically — a bug that reads as the machine
dropping characters. A frontend declares ``uses_console = True`` and the kernel
refuses a second claim, the same way an operating system does not hand the same
tty to two foreground processes.

**Releasing does not stop the reader**, and that is the fix for the way the
above used to fail in practice. Releasing *did* stop it, which meant clearing
``_reader`` while that thread was still blocked in ``readline`` — so the
liveness guard could no longer see it, the next claim started a *second*
reader, and both then appended to one buffer. Worse, whichever orphan woke
first ran its ``finally`` and set ``_closed``, so the frontend that had just
successfully claimed the console got ``EOFError`` on its next read and stopped
itself. A frontend restart therefore killed the terminal.

There is one stdin per process and the reader is a daemon filling a bounded
buffer, so stopping it on release bought nothing. Release now drops ownership
and empties the buffer; the reader keeps running. The one case that genuinely
needs a new reader — a test injecting a different source — is handled by
generation, so a superseded reader can neither append nor declare the console
closed.

**The source is injectable.** ``start(source=...)`` takes any iterator of
lines, so tests drive a console without a terminal and a future frontend could
be fed from somewhere other than stdin. That is most of why this is a Request
rather than a builtin: ``input()`` can only ever mean the real keyboard.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque

logger = logging.getLogger("Sandbox")

# How many unread lines to hold before dropping the oldest. Someone pasting a
# large block should not be able to grow this without bound, and a frontend
# that has stopped draining is already broken.
MAX_BUFFERED = 1000


class Console:
    """One console: a reader thread, a buffer, and at most one claimant."""

    def __init__(self):
        self._lines: deque = deque()
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._stopping = threading.Event()
        self._closed = False
        self._owner: str = ""
        self._writer = None
        # Which reader is the live one. A thread blocked in ``readline``
        # cannot be woken, so a superseded reader stays alive until its next
        # line arrives; it checks this on the way back and, finding itself
        # out of date, neither appends nor reports the console closed.
        self._generation = 0
        self._source = None

    # ── claiming ───────────────────────────────────────────────────

    @property
    def owner(self) -> str:
        """The token holding the console, or "" if nobody does."""
        with self._lock:
            return self._owner

    def claim(self, token: str, source=None, writer=None) -> bool:
        """Take the console for one frontend. False if somebody else has it.

        Re-claiming with the same token succeeds, so a frontend that restarts
        is not locked out by its own previous claim.
        """
        if not token:
            return False
        with self._lock:
            if self._owner and self._owner != token:
                return False
            self._owner = token
            self._writer = writer
        self.start(source)
        return True

    def release(self, token: str) -> None:
        """Give the console back. Only the holder can, so a stale token from a
        frontend that already stopped cannot revoke its successor's claim.

        The reader is left running. Whoever claims next inherits it, which is
        what stops a restart from putting two readers on one stdin.
        """
        with self._lock:
            if not token or self._owner != token:
                return
            self._owner = ""
            self._writer = None
            # Whatever arrived was typed at the frontend that has now gone
            # away, and handing it to the next one would replay a stranger's
            # keystrokes into a fresh session.
            self._lines.clear()

    # ── the reader ─────────────────────────────────────────────────

    def start(self, source=None) -> None:
        """Begin reading. Idempotent for the same source.

        A second claim reuses the running reader rather than starting another,
        which is what keeps one person's keystrokes going to one place. A
        *different* source supersedes: the old reader is retired by generation
        and cannot touch the buffer again.
        """
        with self._lock:
            live = self._reader is not None and self._reader.is_alive()
            same_source = source is None or source is self._source
            if live and same_source:
                return
            self._generation += 1
            generation = self._generation
            self._stopping.clear()
            self._closed = False
            self._lines.clear()
            self._source = source
            self._reader = threading.Thread(
                target=self._read, args=(source, generation), daemon=True,
                name="console-reader")
            self._reader.start()

    def _read(self, source=None, generation: int = 0) -> None:
        """Pull lines until the source ends. Runs on its own thread.

        Daemon, and deliberately not interruptible: a thread blocked in
        ``readline`` cannot be woken, so retirement is something this checks
        after each line rather than something done *to* it. The process
        exiting is what finally ends it.
        """
        stream = source if source is not None else sys.stdin
        try:
            for line in stream:
                with self._lock:
                    if self._stopping.is_set() or generation != self._generation:
                        return
                    self._lines.append(line.rstrip("\n").rstrip("\r"))
                    while len(self._lines) > MAX_BUFFERED:
                        self._lines.popleft()
        except Exception as exc:
            logger.debug("console reader ended: %s", exc)
        finally:
            with self._lock:
                # Only the live reader may declare the console closed. A
                # retired one saying so would hand ``EOFError`` to a frontend
                # whose own reader is alive and well.
                if generation == self._generation:
                    self._closed = True

    def stop(self) -> None:
        """Retire the reader and forget what was buffered.

        For process teardown and tests. Ordinary release does *not* come
        through here — see :meth:`release`.
        """
        self._stopping.set()
        with self._lock:
            self._generation += 1
            self._lines.clear()
            self._reader = None
            self._source = None

    # ── what the Requests reach ────────────────────────────────────

    def read_line(self) -> str | None:
        """The next line, or None if none has arrived. Never blocks.

        Raises when the console is closed *and* drained, so a frontend reading
        a console that has gone away finds out rather than idling forever. On
        a piped stdin that is what stops the frontend at end of input.
        """
        with self._lock:
            if self._lines:
                return self._lines.popleft()
            if self._closed:
                raise EOFError("the console is closed")
        return None

    def write(self, text: str, end: str = "\n") -> None:
        """Put text on the console."""
        writer = self._writer
        if writer is not None:
            writer(f"{text}{end}")
            return
        sys.stdout.write(f"{text}{end}")
        sys.stdout.flush()


# One console, because one process has one. Tests build their own rather than
# reaching for this.
CONSOLE = Console()
