"""Runners — where sandboxed code actually executes.

Two modes, one calling convention, exactly as the security contract requires:
*the code functions the same either way, just with different levels of
security.*

- **in-process** (this module): the code runs on a worker thread with the SDK
  bound to a queue. Fast, no setup cost, and the validation script is the only
  thing standing between the code and the machine — so this mode is for code
  that is trusted, linted, and merely careless.
- **subprocess** (next): the same code, the same SDK, the same Requests, with
  the boundary enforced by the operating system rather than by a linter.

The in-process runner cannot kill a runaway thread — Python provides no such
primitive. It does not need to: every effect passes through the interpreter, so
cancelling an execution starves it of effects at its next Request. Code stuck
in a pure compute loop still burns a core, which is precisely the honest limit
of in-process execution and the reason subprocess mode exists.
"""

from __future__ import annotations

import logging
import threading
import time

from .guest import protocol
from .guest.channel import Terminated
from .guest.codes import ERROR_GUEST_FAULT, ERROR_INVALID_ARGUMENT
from .guest.faults import guest_traceback
from .guest.requests import RequestFailed, Result
from .guest.sdk import SDK
from .interpreter import Execution, Interpreter, clamp_timeout
from .policy import Chain
from .watchdog import TICK, overdue

logger = logging.getLogger("Sandbox")


class TimedOut(Exception):
    """Raised inside the worker when the kernel's clamped deadline passes."""


def run_in_process(interpreter: Interpreter, fn, *, name: str,
                   chain: Chain | None = None, timeout: float | None = None,
                   kwargs: dict | None = None,
                   execution: Execution | None = None) -> Result:
    """Run ``fn(sdk, **kwargs)`` under the sandbox and return its Result.

    ``timeout`` is the plugin's *declared* value. It is clamped here: a plugin
    may ask for a longer leash, it does not get to grant itself one.
    """
    # An execution may be supplied so a caller holding it can cancel a run
    # that is still going.
    if execution is None:
        execution = Execution(name=name, chain=(chain or Chain()).push(name))
    try:
        call_kwargs = protocol.normalize(kwargs or {})
        protocol.encode({"kind": protocol.CALL, "kwargs":
                         protocol.pack_simple(call_kwargs)})
    except protocol.ProtocolError as exc:
        return Result.failure(f"unusable call arguments: {exc}",
                              code=ERROR_INVALID_ARGUMENT)
    sdk = SDK(interpreter.channel(execution))
    deadline = clamp_timeout(timeout)
    box: dict = {}

    def _worker():
        """Run the plugin body and capture whatever it produces."""
        try:
            box["result"] = fn(sdk, **call_kwargs)
        except Terminated as stop:
            box["result"] = Result(data=stop.value)
        except RequestFailed as failed:
            # An uncaught Request failure is that failure, not a mystery. A
            # refusal the plugin chose not to handle still reads as a refusal.
            box["result"] = failed.result
        except Exception as exc:
            # Debug, not exception: the stack rides the Result to the caller
            # and the model now, and StreamHandlers sit at WARNING — so
            # logging it at ERROR would dump every plugin bug on the user's
            # terminal, and only for the in-process runner, which is exactly
            # the asymmetry this reporting exists to remove.
            logger.debug("sandboxed code raised: %s", name, exc_info=True)
            box["result"] = Result.failure(
                f"{type(exc).__name__}: {exc}", code=ERROR_GUEST_FAULT,
                traceback=guest_traceback(exc, drop=(__file__,)))
        finally:
            execution.finished.set()

    worker = threading.Thread(target=_worker, daemon=True,
                              name=f"sandbox-{name}")
    started = time.perf_counter()
    worker.start()

    # Wait on the guest, not on the clock — the same measure a resident box
    # uses, through the same helper, which exists precisely "so the in-process
    # loop and the watchdog thread cannot drift" (``watchdog.overdue``). They
    # had drifted: this was a bare ``wait(timeout=deadline)``, so an ephemeral
    # command was charged for time the *kernel* spent answering it. A command
    # that asked something slow — or waited on a service the kernel was busy
    # with — died at thirty seconds having done nothing wrong, and the report
    # blamed the plugin. ``HARD_CEILING`` still backs it up, so a runaway
    # hiding inside long Requests is not immortal.
    monotonic_start = time.monotonic()
    completed = True
    while not execution.finished.wait(timeout=TICK):
        if overdue(execution, monotonic_start, deadline):
            completed = False
            break

    for level, message in execution.logs:
        logger.log(getattr(logging, level.upper(), logging.INFO),
                   "[%s] %s", name, message)

    if not completed:
        # Starve it rather than pretend we killed it: the thread survives, but
        # its next Request is refused, so it can no longer affect anything.
        interpreter.cancel(execution)
        return Result.failure(
            f"timed out after {deadline:.1f}s (declared {timeout})",
            retryable=True)

    elapsed = time.perf_counter() - started
    logger.debug("%s finished in %.1fms", name, elapsed * 1000)

    if execution.response is not None:
        result = execution.response
    else:
        result = box.get("result")
        if not isinstance(result, Result):
            result = Result(data=result)
    try:
        return result.crossing()
    except protocol.ProtocolError as exc:
        return Result.failure(
            f"unsendable result: {exc}", code=ERROR_GUEST_FAULT)
