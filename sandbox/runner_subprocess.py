"""Subprocess execution — the same code, the same SDK, a harder boundary.

The parent pumps messages: read a Request, hand it to the *same*
:class:`~sandbox.interpreter.Interpreter` the in-process runner uses, write
the Result back. One gate, one policy function, one ledger, two runners.

What differs from in-process is only enforcement. A thread cannot be killed,
so in-process cancellation works by starvation; a process can, so here a
runaway is actually stopped. See :mod:`sandbox.child` for what this boundary
does and does not buy — notably, it is not filesystem isolation yet.

Entry is by *file path and function name*, not a callable: a function defined
in a local scope cannot cross a process boundary. Real plugins are files, so
this is the production shape anyway.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

from .guest import protocol
from .guest.faults import clamp
from .interpreter import Execution, Interpreter, clamp_timeout
from .policy import Chain
from .guest.codes import (ERROR_GUEST_EXITED, ERROR_GUEST_FAULT,
                         ERROR_INVALID_ARGUMENT, ERROR_TIMEOUT)
from .guest.requests import Request, Result

logger = logging.getLogger("Sandbox")

# The child runs ``guest`` as a *top-level* package, with the sandbox
# directory as its working directory. Nothing above ``guest/`` is on its
# import path, so the gate, the policy function and the handlers are not
# merely unused by the child — they are unreachable from it.
#
# This is also the container shape: an image copies ``guest/`` alone and runs
# the identical command.
CHILD_MODULE = "guest.child"
GUEST_ROOT = Path(__file__).resolve().parent


def run_in_subprocess(interpreter: Interpreter, module_path: str,
                      func_name: str, *, name: str,
                      chain: Chain | None = None, timeout: float | None = None,
                      kwargs: dict | None = None,
                      memory_mb: int | None = None,
                      box: str = "", box_root: str | None = None,
                      extra_roots: list | None = None,
                      root_dir: str | None = None,
                      execution: Execution | None = None,
                      on_proc=None, method: str = "run",
                      digest: str = "") -> Result:
    """Run ``func_name`` from ``module_path`` in a child process.

    ``timeout`` and ``memory_mb`` are the plugin's *declared* values and are
    clamped by the kernel — a plugin may ask for a longer leash, it does not
    get to grant itself one.
    """
    if execution is None:
        execution = Execution(name=name, chain=(chain or Chain()).push(name))
    deadline = clamp_timeout(timeout)
    root = root_dir or str(GUEST_ROOT)

    proc = subprocess_for(root)
    if on_proc is not None:
        # Hand the process out so a caller can kill a background run. A
        # cancelled execution stops being *serviced*, but only closing the
        # pipe stops a loop that ignores its Results.
        on_proc(proc)

    killed = threading.Event()

    def _watchdog():
        """Stop a runaway. A process, unlike a thread, can actually be killed."""
        killed.set()
        interpreter.cancel(execution)
        try:
            proc.kill()
        except OSError:
            pass

    timer = threading.Timer(deadline, _watchdog)
    timer.daemon = True
    timer.start()
    started = time.perf_counter()

    try:
        return _pump(interpreter, execution, proc, killed, deadline, {
            "kind": protocol.START,
            "module": str(module_path),
            "func": func_name,
            "method": method,
            "kwargs": kwargs or {},
            "box": box,
            "root": box_root,
            "extra_roots": list(extra_roots or []),
            "memory_mb": memory_mb,
            "cpu_seconds": int(deadline) + 1,
            "digest": digest,
        })
    finally:
        timer.cancel()
        _reap(proc)
        logger.debug("%s (subprocess) finished in %.1fms", name,
                     (time.perf_counter() - started) * 1000)


def subprocess_for(root: str):
    """Spawn a guest child rooted at ``sandbox/``.

    stderr is deliberately inherited rather than piped: a pipe nobody drains
    fills its buffer and blocks the child forever. The child's own prints go
    there, so they land in the parent's console as ordinary output.
    """
    return subprocess.Popen(
        [sys.executable, "-m", CHILD_MODULE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        cwd=root,
    )


def send(proc, message: dict) -> bool:
    """Write one message to the child; False if the channel is gone."""
    try:
        protocol.write_message(proc.stdin, message)
        return True
    except (OSError, ValueError):
        return False


def service_until(interpreter: Interpreter, execution: Execution, proc,
                  expected: set):
    """Answer the child's Requests until it says one of ``expected``.

    Shared by both lifetimes, so an ephemeral run and a call into a resident
    box are serviced by identical code — same gate, same provenance, same
    ledger. Returns the awaited message, or ``None`` if the child's stream
    closed first.
    """
    while True:
        try:
            message = protocol.read_message(proc.stdout)
        except protocol.ProtocolError as exc:
            return {"kind": protocol.FAULT, "error": f"protocol error: {exc}"}

        if message is None:
            return None

        kind = message.get("kind")
        if kind in expected:
            return message

        if kind == protocol.REQUEST:
            # A child naming a Request type that does not exist raises out of
            # ``Request.from_dict``. Letting that escape left the box marked
            # alive with the child still waiting for a RESULT that would never
            # come — a wedge, from a typo. Answer it instead.
            try:
                request = Request.from_dict(message["request"])
            except (KeyError, TypeError, ValueError) as exc:
                if not send(proc, {"kind": protocol.RESULT,
                                   "result": Result.failure(
                                       f"unusable request: {exc}",
                                       code=ERROR_INVALID_ARGUMENT).to_dict()}):
                    return None
                continue
            # The same gate the in-process runner uses: one classification
            # path, one provenance stack, one ledger.
            result = interpreter.submit(execution, request)
            if not send(proc, {"kind": protocol.RESULT,
                               "result": result.to_dict()}):
                return None

        elif kind == protocol.NOTICE:
            # Same gate, same provenance, same ledger — the only difference
            # is that nothing is written back, because nobody is waiting. A
            # malformed one is dropped for the same reason: there is no reply
            # channel to report it down.
            try:
                notice = Request.from_dict(message["request"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("[%s] discarded an unusable notice: %s",
                               execution.name, exc)
                continue
            interpreter.submit(execution, notice)

        elif kind == protocol.LOG:
            level = str(message.get("level", "info")).upper()
            logger.log(getattr(logging, level, logging.INFO),
                       "[%s] %s", execution.name, message.get("message", ""))

        else:
            return {"kind": protocol.FAULT,
                    "error": f"unexpected message kind: {kind}"}


def fault_result(message: dict, name: str) -> Result:
    """Turn a fault message into a failure Result.

    The child's stack used to go to ``logger.debug``, which is below the level
    the app configures — so the one site that captured a real traceback was the
    one site that threw it away. It rides on the Result now, which is the only
    channel reaching the caller, the frontend and the model.

    Re-clamped on receipt: the child is the far side of a trust boundary, and a
    cap it applies to itself is not a cap.
    """
    return Result.failure(message.get("error", "sandboxed code faulted"),
                          code=ERROR_GUEST_FAULT,
                          traceback=clamp(str(message.get("traceback") or "")))


def _pump(interpreter: Interpreter, execution: Execution, proc,
          killed: threading.Event, deadline: float, start: dict) -> Result:
    """Drive an ephemeral child until it finishes, faults, or is killed."""
    if not send(proc, start):
        return Result.failure("child exited before it could be started",
                              code=ERROR_GUEST_EXITED)

    message = service_until(interpreter, execution, proc,
                            {protocol.DONE, protocol.FAULT})

    # The watchdog firing is the answer, whatever the child managed to say on
    # its way out. Since a refused Request now unwinds the plugin promptly, a
    # starved runaway can return a tidy "denied" a moment before the kill
    # lands — and reporting that would hide the fact that it ran too long.
    if killed.is_set():
        return Result.failure(f"timed out after {deadline:.1f}s",
                              retryable=True, code=ERROR_TIMEOUT)
    if message is None:
        return Result.failure(
            f"child exited without a result (code {proc.poll()})",
            code=ERROR_GUEST_EXITED)
    if message["kind"] == protocol.FAULT:
        return fault_result(message, execution.name)
    return Result.from_dict(message["result"])


def _reap(proc):
    """Close pipes and make sure the child is gone."""
    for stream in (proc.stdin, proc.stdout):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("sandbox child would not die")
