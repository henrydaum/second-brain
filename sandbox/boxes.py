"""Resident boxes — sandboxed code the kernel calls into repeatedly.

An ephemeral box *is* its call: run, answer, gone. A resident one is loaded
once and then waits, holding whatever state it acquired — a model, a
connection, a scratchpad's globals — while the kernel calls in.

**Persistence is decided before the code runs.** It is a property of the box,
resolved from declarations, never something code can drift into by refusing to
finish. A script that never returns in an ephemeral box is not persistent; it
is a runaway, and it times out. Nothing can promote itself to unlimited
lifetime by hanging.

**One call at a time.** Every box serializes: a lock per box, one call in
flight. That is what lets the wire protocol work without message ids, and it
matches how a loaded service behaves anyway.

**Ending a box is the kernel's decision**, escalating through three tiers:

===========  =========================================  ==========  ==========
Tier         Mechanism                                  in-process  subprocess
===========  =========================================  ==========  ==========
``ask``      ``stop`` message; ``stop()`` runs; exit    yes         yes
``starve``   refuse every Request; code unwinds at its  yes         yes
             next one
``kill``     terminate the process                      no          yes
===========  =========================================  ==========  ==========

Starvation is the tier that works everywhere, and it works *because* every
effect goes through the gate: code that cannot act cannot resist being shut
down. The honest gap is a pure compute loop that makes no Requests — nothing
short of ``kill`` reaches that, and in-process there is no ``kill``.
"""

from __future__ import annotations

import logging
import threading

from .guest import protocol
from .guest.loader import load_entry, unload_box
from .guest.requests import Result
from .guest.sdk import SDK
from .interpreter import Execution, Interpreter, clamp_timeout
from .policy import Chain
from .runner_subprocess import (GUEST_ROOT, fault_result, send, service_until,
                                subprocess_for)

logger = logging.getLogger("Sandbox")

# Loading a service may be slow — a model off disk, a connection handshake —
# so starting a box gets its own, longer ceiling than any single call to it.
# This is the role the old ``BaseService.load_timeout`` played.
DEFAULT_START_TIMEOUT = 600.0


class BoxError(RuntimeError):
    """A box could not be started, or was used after it stopped."""


class PersistentBox:
    """A loaded box the kernel can call into. Base for both runners."""

    def __init__(self, name: str, chain: Chain | None = None,
                 call_timeout: float | None = None):
        self.name = name
        self.execution = Execution(name=name,
                                   chain=(chain or Chain()).push(name))
        # For a resident box the declared timeout bounds one *call*: there is
        # no total lifetime to bound.
        self.call_timeout = clamp_timeout(call_timeout)
        self._lock = threading.Lock()
        self._alive = False

    @property
    def alive(self) -> bool:
        """Whether this box can still take calls."""
        return self._alive

    def call(self, method: str, **kwargs) -> Result:
        """Invoke a method and wait for its answer. Serialized per box."""
        if not self._alive:
            return Result.failure(f"box {self.name!r} is not running")
        with self._lock:
            if not self._alive:
                return Result.failure(f"box {self.name!r} is not running")
            return self._call(method, kwargs)

    def stop(self, timeout: float = 10.0) -> Result:
        """Shut down: ask, then starve, then kill."""
        raise NotImplementedError

    def _call(self, method: str, kwargs: dict) -> Result:
        """Runner-specific call."""
        raise NotImplementedError


class InProcessBox(PersistentBox):
    """A resident box on this side of the boundary.

    The instance simply stays in memory, so a call is a direct invocation —
    no serve loop needed, because there is no boundary to cross. Each call
    still runs on its own worker thread so the per-call deadline and the
    starvation path behave exactly as they do for a subprocess.
    """

    def __init__(self, target, name: str, chain=None, call_timeout=None):
        super().__init__(name, chain, call_timeout)
        self.target = target
        self.sdk = SDK(None)   # channel attached at start()

    def start(self, interpreter: Interpreter,
              timeout: float = DEFAULT_START_TIMEOUT) -> Result:
        """Run the target's ``start`` if it has one, then stand by."""
        self.sdk = SDK(interpreter.channel(self.execution))
        self._interpreter = interpreter
        start_fn = getattr(self.target, "start", None)
        if callable(start_fn):
            outcome = self._invoke(start_fn, {}, clamp_timeout(timeout))
            if not outcome.ok:
                return outcome
        self._alive = True
        return Result(data=True)

    def _invoke(self, fn, kwargs: dict, deadline: float) -> Result:
        """Run one callable on a worker thread under a deadline."""
        from .guest.channel import Terminated
        from .guest.requests import RequestFailed

        box: dict = {}
        done = threading.Event()

        def _worker():
            """Call it and capture whatever comes back."""
            try:
                raw = fn(self.sdk, **kwargs)
                box["result"] = raw if isinstance(raw, Result) else Result(
                    data=raw)
            except Terminated as stop:
                box["result"] = Result(data=stop.value)
            except RequestFailed as failed:
                box["result"] = failed.result
            except Exception as exc:
                box["result"] = Result.failure(f"{type(exc).__name__}: {exc}")
            finally:
                done.set()

        threading.Thread(target=_worker, daemon=True,
                         name=f"box-{self.name}").start()
        if not done.wait(timeout=deadline):
            # Starve it: the thread survives, but its next Request is refused
            # so it can no longer affect anything.
            self._interpreter.cancel(self.execution)
            return Result.failure(f"timed out after {deadline:.1f}s",
                                  retryable=True)
        return box.get("result", Result(data=None))

    def _call(self, method: str, kwargs: dict) -> Result:
        """Invoke a method on the resident instance."""
        fn = getattr(self.target, method, None)
        if not callable(fn):
            return Result.failure(f"no such method: {method!r}")
        return self._invoke(fn, kwargs, self.call_timeout)

    def stop(self, timeout: float = 10.0) -> Result:
        """Ask it to stop, then starve it. There is no kill for a thread."""
        if not self._alive:
            return Result(data=False)
        self._alive = False
        stop_fn = getattr(self.target, "stop", None)
        if callable(stop_fn):
            self._invoke(stop_fn, {}, clamp_timeout(timeout))
        self._interpreter.cancel(self.execution)
        unload_box(self.name)
        return Result(data=True)


class SubprocessBox(PersistentBox):
    """A resident box behind a process boundary."""

    def __init__(self, module_path: str, entry: str, name: str,
                 chain=None, call_timeout=None, box_root=None,
                 memory_mb: int | None = None, extra_roots=()):
        super().__init__(name, chain, call_timeout)
        self.module_path = str(module_path)
        self.entry = entry
        self.box_root = box_root
        self.extra_roots = list(extra_roots or [])
        self.memory_mb = memory_mb
        self.proc = None
        self._interpreter = None

    def start(self, interpreter: Interpreter,
              timeout: float = DEFAULT_START_TIMEOUT) -> Result:
        """Spawn the child, load the target, and wait for it to stand by."""
        self._interpreter = interpreter
        self.proc = subprocess_for(str(GUEST_ROOT))
        deadline = clamp_timeout(timeout)

        timer = threading.Timer(deadline, self._kill)
        timer.daemon = True
        timer.start()
        try:
            ok = send(self.proc, {
                "kind": protocol.START,
                "module": self.module_path,
                "func": self.entry,
                "persistent": True,
                "box": self.name,
                "root": self.box_root,
                "extra_roots": self.extra_roots,
                "memory_mb": self.memory_mb,
                "cpu_seconds": None,
            })
            if not ok:
                return Result.failure("child exited before it could start")
            message = service_until(interpreter, self.execution, self.proc,
                                    {protocol.READY, protocol.FAULT})
        finally:
            timer.cancel()

        if message is None:
            self._reap()
            return Result.failure(f"box {self.name!r} died while starting")
        if message["kind"] == protocol.FAULT:
            self._reap()
            return fault_result(message, self.name)

        self._alive = True
        return Result(data=True)

    def _call(self, method: str, kwargs: dict) -> Result:
        """Send a call and service Requests until the answer comes back.

        The deadline escalates rather than merely starving. Refusing a
        Request only stops code that propagates the failure; a loop that
        ignores its Results keeps asking forever, and the parent would keep
        answering forever. Killing closes the pipe, which is what actually
        ends both sides.

        A consequence of serializing per box: the child has one thread of
        control, so a hung call cannot be cancelled on its own. Ending it ends
        the box.
        """
        timer = threading.Timer(self.call_timeout, self._kill)
        timer.daemon = True
        timer.start()
        try:
            if not send(self.proc, {"kind": protocol.CALL, "method": method,
                                    "kwargs": kwargs}):
                self._alive = False
                return Result.failure("box channel closed")
            message = service_until(self._interpreter, self.execution,
                                    self.proc,
                                    {protocol.RETURN, protocol.FAULT})
        finally:
            timer.cancel()

        if message is None:
            self._alive = False
            self._reap()
            return Result.failure(
                f"box {self.name!r} died during {method!r}", retryable=True)
        if message["kind"] == protocol.FAULT:
            return fault_result(message, self.name)
        return Result.from_dict(message["result"])

    def _starve(self):
        """Stop answering this box's Requests so a stuck call unwinds."""
        if self._interpreter is not None:
            self._interpreter.cancel(self.execution)

    def _kill(self):
        """Last resort, and the one thing a thread can never do."""
        self._starve()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass

    def stop(self, timeout: float = 10.0) -> Result:
        """Ask, then starve, then kill."""
        if not self._alive:
            return Result(data=False)
        self._alive = False
        send(self.proc, {"kind": protocol.STOP})
        try:
            self.proc.wait(timeout=timeout)
        except Exception:
            self._kill()
        self._reap()
        return Result(data=True)

    def _reap(self):
        """Close the pipes and make sure the child is gone."""
        self._alive = False
        proc, self.proc = self.proc, None
        if proc is None:
            return
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
        except Exception:
            logger.warning("box %s would not die", self.name)


def open_box(interpreter: Interpreter, module_path, entry: str = "", *,
             name: str, isolated: bool = False, chain: Chain | None = None,
             call_timeout: float | None = None,
             start_timeout: float = DEFAULT_START_TIMEOUT,
             box_root=None, memory_mb: int | None = None,
             extra_roots=()) -> PersistentBox:
    """Load a resident box and return a handle to call into.

    ``entry`` names a plugin class, or is empty for a bare script — in which
    case the module itself is the object, its functions are the methods, and
    its globals are the state that persists. That is the whole of a scratchpad
    server.

    Raises :class:`BoxError` if it will not start, since a handle to a box
    that never loaded is not a useful thing to hand back.
    """
    if isolated:
        box = SubprocessBox(module_path, entry, name, chain=chain,
                            call_timeout=call_timeout, box_root=box_root,
                            memory_mb=memory_mb, extra_roots=extra_roots)
    else:
        target = load_entry(module_path, entry, box_name=name, root=box_root,
                            bound=False, extra_roots=extra_roots)
        box = InProcessBox(target, name, chain=chain,
                           call_timeout=call_timeout)

    outcome = box.start(interpreter, timeout=start_timeout)
    if not outcome.ok:
        raise BoxError(f"{name}: {outcome.error}")
    return box
