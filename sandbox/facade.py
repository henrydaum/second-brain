"""The front door — one API for running anything under the sandbox.

Underneath there are three mechanisms (a thread runner, a process runner, and
resident boxes) with three different shapes. Callers should not have to know
which one they are using: whether code is isolated, how long it may take, and
whether it stays resident are all *declarations in the file*, resolved here.

So one call does the whole sequence:

    validate -> read declarations -> resolve the box -> clamp -> pick a runner

If a file will not pass validation it is never executed, and the bytes that
were validated are the bytes that run — the report carries them, so a file
that changes in between was never checked.

Two lifetimes, three entry points:

- :meth:`Sandbox.run` — ephemeral, blocking. Returns a Result.
- :meth:`Sandbox.start` — ephemeral, non-blocking. Returns a :class:`Run` to
  wait on, poll, or cancel later. This is the ``wait=False`` shape: a tool
  that spawns background work returns immediately while the work continues.
- :meth:`Sandbox.open` — resident. Returns a box to call into until stopped.

Every box the sandbox opens is tracked, so :meth:`Sandbox.shutdown` can close
them all. An untracked resident box is an orphaned process after a restart.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .approval import build_approver
from .boxes import DEFAULT_START_TIMEOUT, BoxError, PersistentBox, open_box
from .guest.box import PERSISTENT, SUBPROCESS, Membership, resolve
from .guest.loader import load_entry, unload_box
from .guest.requests import Result
from .interpreter import Execution, Interpreter
from .policy import Chain
from .runner import run_in_process
from .runner_subprocess import run_in_subprocess
from .validator import validate_file

logger = logging.getLogger("Sandbox")


class Run:
    """An ephemeral execution in flight.

    Returned by :meth:`Sandbox.start`. The work is already running; this is
    the handle for finding out how it went, or for stopping it.
    """

    def __init__(self, name: str, chain: Chain, execution: Execution,
                 interpreter: Interpreter):
        self.name = name
        self.chain = chain
        self._execution = execution
        self._interpreter = interpreter
        self._future = None
        self._proc = None
        self._cancelled = False

    def _attach(self, future):
        """Bind the future once the work has been submitted."""
        self._future = future
        return self

    def _attach_proc(self, proc):
        """Remember the child process so cancelling can reach it."""
        self._proc = proc

    @property
    def done(self) -> bool:
        """Whether the work has finished, one way or another."""
        return self._future is not None and self._future.done()

    @property
    def cancelled(self) -> bool:
        """Whether this run was cancelled."""
        return self._cancelled

    def wait(self, timeout: float | None = None) -> Result:
        """Block for the result. Safe to call more than once."""
        if self._future is None:
            return Result.failure("run was never started")
        try:
            return self._future.result(timeout=timeout)
        except TimeoutError:
            return Result.failure("still running", retryable=True)
        except Exception as exc:
            return Result.failure(f"run failed: {exc}")

    def cancel(self) -> None:
        """Stop it: starve the Requests, then close the pipe.

        Starvation alone only reaches code that propagates its failures; a
        loop that ignores Results keeps going. Killing the child is what ends
        both sides. In-process there is no kill, so a compute loop with no
        Requests survives — the honest limit of that runner.
        """
        self._cancelled = True
        self._interpreter.cancel(self._execution)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


class Sandbox:
    """The kernel's handle on sandboxed execution."""

    def __init__(self, interpreter: Interpreter | None = None, *,
                 context=None, approve=None, record=None, runtime=None,
                 session_key=None, max_background: int = 4):
        """
        context:
            The kernel's context object, passed to every handler — Second
            Brain's ``SecondBrainContext``. Lives on the host side and never
            crosses into the guest.
        runtime:
            The conversation runtime. Supplying it wires approval to the
            kernel's own doorway — ``vet_permission`` hooks, the user's
            trusted list, then a dialog. Without one, and without an explicit
            ``approve``, everything unsafe is refused.
        """
        if approve is None and runtime is not None:
            approve = build_approver(runtime, session_key)
        self.interpreter = interpreter or Interpreter(
            approve=approve, record=record, context=context)
        self._pool = ThreadPoolExecutor(max_workers=max_background,
                                        thread_name_prefix="sandbox-bg")
        self._boxes: dict[str, PersistentBox] = {}
        self._runs: list[Run] = []
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────
    # Resolving what a file asked for.
    # ──────────────────────────────────────────────────────────────

    def inspect(self, source):
        """Validate a file and resolve the box it would run in.

        Returns ``(report, spec)``. Does not import or execute anything.
        """
        report = validate_file(source)
        declared = report.declarations
        membership = Membership(
            source=Path(source).stem,
            box=str(declared.get("box") or ""),
            isolation=str(declared.get("isolation") or ""),
            lifetime=str(declared.get("lifetime") or ""),
            timeout=float(declared.get("timeout") or 0),
            memory_mb=int(declared.get("memory_mb") or 0),
        )
        spec = resolve([membership])[membership.box_name]
        return report, spec

    def _prepare(self, source, *, isolated=None, timeout=None, name=None):
        """Validate and resolve, or explain why the file will not run."""
        report, spec = self.inspect(source)
        if not report.ok:
            raise BoxError(report.render())
        if report.disclaimed:
            logger.warning("%s runs with a disclaimer:\n%s",
                           report.filename, report.render())
        return report, spec, {
            "name": name or spec.name,
            "isolated": spec.isolation == SUBPROCESS if isolated is None
            else isolated,
            "timeout": timeout if timeout is not None else (
                spec.timeout or None),
            "memory_mb": spec.memory_mb or None,
        }

    # ──────────────────────────────────────────────────────────────
    # Ephemeral.
    # ──────────────────────────────────────────────────────────────

    def run(self, source, entry: str = "", *, kwargs: dict | None = None,
            chain: Chain | None = None, name: str | None = None,
            isolated: bool | None = None, timeout: float | None = None,
            context=None) -> Result:
        """Run once and wait for the answer. The ``wait=True`` shape."""
        run = self.start(source, entry, kwargs=kwargs, chain=chain, name=name,
                         isolated=isolated, timeout=timeout, context=context)
        return run.wait()

    def start(self, source, entry: str = "", *, kwargs: dict | None = None,
              chain: Chain | None = None, name: str | None = None,
              isolated: bool | None = None, timeout: float | None = None,
              on_done=None, context=None) -> Run:
        """Run without waiting. The ``wait=False`` shape.

        The work begins immediately on a background thread and the caller
        keeps going. ``on_done`` fires with the Result when it finishes, which
        is how a spawner queues a completion notice back to its session.
        """
        report, spec, opts = self._prepare(source, isolated=isolated,
                                           timeout=timeout, name=name)
        if spec.lifetime == PERSISTENT:
            raise BoxError(
                f"{opts['name']} declares a persistent lifetime; open it as a "
                f"resident box rather than running it")

        run_name = opts["name"]
        execution = Execution(name=run_name,
                              chain=(chain or Chain()).push(run_name),
                              context=context)
        run = Run(run_name, execution.chain, execution, self.interpreter)

        def _work() -> Result:
            """Do the run on a background thread."""
            try:
                if opts["isolated"]:
                    return run_in_subprocess(
                        self.interpreter, str(source), entry, name=run_name,
                        kwargs=kwargs, timeout=opts["timeout"],
                        memory_mb=opts["memory_mb"], box=spec.name,
                        box_root=str(Path(source).parent),
                        execution=execution, on_proc=run._attach_proc)
                target = load_entry(source, entry, box_name=spec.name)
                return run_in_process(
                    self.interpreter, target, name=run_name, kwargs=kwargs,
                    timeout=opts["timeout"], execution=execution)
            finally:
                if not opts["isolated"]:
                    unload_box(spec.name)

        def _finish(future):
            """Notify the caller, without letting a bad callback matter."""
            with self._lock:
                if run in self._runs:
                    self._runs.remove(run)

            if on_done is None:
                return
            try:
                on_done(future.result())
            except Exception:
                logger.exception("on_done callback failed for %s", run_name)

        with self._lock:
            self._runs.append(run)
        future = self._pool.submit(_work)
        future.add_done_callback(_finish)
        return run._attach(future)

    # ──────────────────────────────────────────────────────────────
    # Resident.
    # ──────────────────────────────────────────────────────────────

    def open(self, source, entry: str = "", *, name: str | None = None,
             chain: Chain | None = None, isolated: bool | None = None,
             call_timeout: float | None = None,
             start_timeout: float = DEFAULT_START_TIMEOUT) -> PersistentBox:
        """Load a resident box and keep a handle on it."""
        report, spec, opts = self._prepare(source, isolated=isolated,
                                           timeout=call_timeout, name=name)
        box_name = opts["name"]
        with self._lock:
            existing = self._boxes.get(box_name)
            if existing is not None and existing.alive:
                return existing

        box = open_box(self.interpreter, source, entry, name=box_name,
                       isolated=opts["isolated"], chain=chain,
                       call_timeout=opts["timeout"],
                       start_timeout=start_timeout,
                       box_root=str(Path(source).parent),
                       memory_mb=opts["memory_mb"])
        with self._lock:
            self._boxes[box_name] = box
        return box

    def box(self, name: str) -> PersistentBox | None:
        """The resident box under this name, if one is loaded."""
        box = self._boxes.get(name)
        return box if box is not None and box.alive else None

    def boxes(self) -> list:
        """Every resident box currently loaded."""
        return [b for b in self._boxes.values() if b.alive]

    def close(self, name: str) -> Result:
        """Stop one resident box."""
        with self._lock:
            box = self._boxes.pop(name, None)
        if box is None:
            return Result(data=False)
        outcome = box.stop()
        unload_box(name)
        return outcome

    # ──────────────────────────────────────────────────────────────
    # Teardown.
    # ──────────────────────────────────────────────────────────────

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel background runs and close every resident box.

        Without this a restart leaves orphaned processes behind, which is the
        whole reason the sandbox tracks what it opened.
        """
        with self._lock:
            runs, self._runs = list(self._runs), []
            names = list(self._boxes)

        # Cancel first, then give the runs a moment to unwind *before* the
        # gate goes away. A run still draining when the interpreter stops
        # would otherwise wait on an answer that can no longer come.
        for run in runs:
            if not run.done:
                run.cancel()
        deadline = time.monotonic() + timeout
        for run in runs:
            run.wait(timeout=max(0.0, deadline - time.monotonic()))

        for name in names:
            try:
                self.close(name)
            except Exception:
                logger.exception("failed to close box %s", name)
        self._pool.shutdown(wait=False)
        self.interpreter.shutdown()
