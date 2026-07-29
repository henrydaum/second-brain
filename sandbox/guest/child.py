"""The child half of a subprocessed sandbox — ``python -m guest.child``.

Reads a ``start`` message, imports the target file, runs the function with an
SDK bound to the pipe, and reports a ``done``. It holds no interpreter, no
queues, and no threads: it writes a Request and blocks on a pipe read, which
is why the plugin code above it cannot tell which runner it is on.

**What this boundary does and does not buy.** A separate process gives
killability (a runaway can actually be stopped, unlike a thread), crash
containment, and — on POSIX — memory and CPU ceilings via ``setrlimit``. It
does *not* restrict which files the child may open: it runs as the same user
with the same permissions. Genuine filesystem isolation needs a container, an
AppContainer, or a separate account with ACLs, and that is a later step. Do
not describe this mode as full isolation until that lands.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback

from . import protocol
from .channel import PipeChannel, Terminated
from .loader import load_entry
from .requests import RequestFailed, Result
from .sdk import SDK

#: How long an error record silences its own repeats. Five minutes is long
#: enough that an outage lasting all night costs a screenful rather than a
#: gigabyte, and short enough that a problem which is still happening keeps
#: saying so.
LOG_REPEAT_COOLDOWN = 300.0


class _CollapseRepeats(logging.Filter):
    """Let one of each recurring error through per cooldown, and count the rest.

    Plugin code may not import ``logging`` — ``sdk.log`` is the route, and it
    goes down the wire to the kernel's sink. The *libraries* plugin code imports
    have no such rule, and nothing here can give them one. Their records land on
    ``logging.lastResort``, which writes to stderr, which a subprocess box
    inherits from the parent rather than piping (a pipe nobody drains is a
    child that blocks when it fills).

    That arrangement is fine until something retries forever. A chat frontend's
    long-poll loop meeting a DNS failure logs a full traceback every cycle, all
    night, and the useful contents of the terminal are gone by morning. So the
    first occurrence is allowed through — without its traceback, since the
    hundredth copy of a stack is not more informative than the first —
    identical ones are dropped for a cooldown, and the next one through says how
    many were swallowed.

    Keyed on the record's *template* (module, line, unformatted message) rather
    than its rendered text, so a message that varies only by a retry counter or
    a timestamp still collapses. Errors only: a library at WARNING is not
    looping, and one at DEBUG was asked for.
    """

    def __init__(self, cooldown: float = LOG_REPEAT_COOLDOWN):
        super().__init__()
        self.cooldown = cooldown
        self._seen: dict = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Whether this record is the one that gets to speak."""
        if record.levelno < logging.ERROR:
            return True
        key = (record.name, record.module, record.lineno, str(record.msg))
        now = time.monotonic()
        until, suppressed = self._seen.get(key, (0.0, 0))
        if now < until:
            self._seen[key] = (until, suppressed + 1)
            return False
        record.exc_info = None
        record.exc_text = None
        if suppressed:
            record.msg = (f"{record.getMessage()} (+{suppressed} more like it "
                          f"in the last {int(self.cooldown)}s)")
            record.args = ()
        self._seen[key] = (now + self.cooldown, 0)
        return True


def _tame_library_logging() -> None:
    """Give the child's root logger a stderr handler that collapses repeats.

    Configured rather than left to ``lastResort`` because a filter needs a
    handler to hang on: ``lastResort`` is shared process-wide state we would be
    mutating for everyone, and it has no formatter worth keeping. Idempotent,
    and it defers to a child that somehow already has handlers.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "[box] %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_CollapseRepeats())
    root.addHandler(handler)
    root.setLevel(logging.WARNING)


def _redirect_stdout_to_stderr():
    """Take the wire, then send everything else to stderr.

    Plugin code prints, and so do libraries it imports at import time. If any
    of that reached the protocol stream it would corrupt the channel in a way
    that looks like a parser bug, so the real stdout is claimed for the wire
    before anything else runs and ``sys.stdout`` is pointed at stderr.
    """
    wire = sys.stdout.buffer
    sys.stdout = sys.stderr
    return wire


def _apply_limits(memory_mb: int | None, cpu_seconds: int | None):
    """Apply OS resource ceilings where the platform provides them.

    POSIX only. On Windows this is a no-op and the parent's wall-clock timeout
    plus ``kill()`` is the whole story — stated plainly rather than papered
    over, because a limit you believe in but do not have is worse than none.
    """
    try:
        import resource
    except ImportError:
        return False
    if memory_mb:
        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    if cpu_seconds:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    return True


def _fault(wire_out, exc) -> int:
    """Report a failure the code could not express as a Result."""
    protocol.write_message(wire_out, {
        "kind": protocol.FAULT,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    })
    return 1


def _send_result(wire_out, kind: str, result: Result) -> bool:
    """Send a Result, or fault if it will not fit down the wire."""
    try:
        protocol.write_message(wire_out, {"kind": kind,
                                          "result": result.to_dict()})
        return True
    except protocol.ProtocolError as exc:
        protocol.write_message(wire_out, {
            "kind": protocol.FAULT, "error": f"unsendable result: {exc}",
        })
        return False


def _run_ephemeral(wire_out, sdk, target, kwargs) -> int:
    """Call once, answer once, exit. The lifetime is the call."""
    try:
        raw = target(sdk, **kwargs)
        result = raw if isinstance(raw, Result) else Result(data=raw)
    except Terminated as stop:
        result = Result(data=stop.value)
    except RequestFailed as failed:
        # An uncaught Request failure is that failure, not a mystery.
        result = failed.result
    except Exception as exc:
        return _fault(wire_out, exc)
    return 0 if _send_result(wire_out, protocol.DONE, result) else 1


def _serve_persistent(wire_in, wire_out, sdk, instance,
                      manage_lifecycle: bool = True) -> int:
    """Load, announce, then answer calls until told to stop.

    Between calls this blocks on a read and executes nothing: a resident box
    costs a process and its memory, not CPU. It never decides to exit — the
    kernel owns the box, and the only ways out are a ``stop``, a channel that
    closes, or being killed.
    """
    start_fn = getattr(instance, "start", None)
    if manage_lifecycle and callable(start_fn):
        try:
            start_fn(sdk)
        except Terminated:
            return 0
        except Exception as exc:
            return _fault(wire_out, exc)

    protocol.write_message(wire_out, {"kind": protocol.READY})

    while True:
        try:
            message = protocol.read_message(wire_in)
        except protocol.ProtocolError as exc:
            return _fault(wire_out, exc)

        # The parent hung up. Nothing will answer our Requests, so stop.
        if message is None:
            break

        kind = message.get("kind")

        if kind == protocol.STOP:
            stop_fn = getattr(instance, "stop", None)
            if manage_lifecycle and callable(stop_fn):
                try:
                    stop_fn(sdk)
                except Exception:
                    pass    # a failed teardown must not block the shutdown
            break

        if kind != protocol.CALL:
            return _fault(wire_out, ValueError(f"unexpected message: {kind}"))

        method = getattr(instance, message.get("method", ""), None)
        if not callable(method):
            _send_result(wire_out, protocol.RETURN, Result.failure(
                f"no such method: {message.get('method')!r}"))
            continue

        try:
            raw = method(
                sdk,
                *(message.get("args") or []),
                **(message.get("kwargs") or {}),
            )
            result = raw if isinstance(raw, Result) else Result(data=raw)
        except Terminated:
            # Requests are being refused: the kernel is tearing us down and a
            # further answer would go nowhere.
            break
        except RequestFailed as failed:
            result = failed.result
        except Exception as exc:
            # One bad call must not take the service down with it.
            result = Result.failure(f"{type(exc).__name__}: {exc}")

        if not _send_result(wire_out, protocol.RETURN, result):
            return 1

    return 0


def main() -> int:
    """Run one box, ephemeral or resident."""
    wire_out = _redirect_stdout_to_stderr()
    wire_in = sys.stdin.buffer
    # Before the target is imported: a library can log at import time, and a
    # loop that floods is one that started early.
    _tame_library_logging()

    try:
        start = protocol.read_message(wire_in)
    except protocol.ProtocolError as exc:
        sys.stderr.write(f"sandbox child: bad start message: {exc}\n")
        return 2
    if start is None or start.get("kind") != protocol.START:
        sys.stderr.write("sandbox child: expected a start message\n")
        return 2

    _apply_limits(start.get("memory_mb"), start.get("cpu_seconds"))
    sdk = SDK(PipeChannel(wire_in, wire_out))

    try:
        target = load_entry(start["module"], start["func"],
                            box_name=start.get("box") or "",
                            root=start.get("root") or None,
                            extra_roots=start.get("extra_roots") or (),
                            bound=not start.get("persistent"),
                            method=start.get("method") or "run",
                            digest=start.get("digest") or "")
    except Exception as exc:
        return _fault(wire_out, exc)

    if start.get("persistent"):
        return _serve_persistent(
            wire_in, wire_out, sdk, target,
            manage_lifecycle=bool(start.get("manage_lifecycle", True)),
        )
    return _run_ephemeral(wire_out, sdk, target, start.get("kwargs") or {})


if __name__ == "__main__":
    sys.exit(main())
