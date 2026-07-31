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
import time

from .guest import protocol
from .guest.codes import ERROR_GUEST_FAULT
from .guest.faults import guest_traceback
from .guest.loader import (install_parsers, load_entries, load_entry,
                           unload_box)
from .guest.requests import Result
from .guest.sdk import SDK
from .interpreter import Execution, Interpreter, clamp_timeout
from .policy import Chain
from .runner_subprocess import (GUEST_ROOT, fault_result, send, service_until,
                                subprocess_for)
from .watchdog import WATCHDOG

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

    def call(self, method: str, *args, target: str = "", **kwargs) -> Result:
        """Invoke a method and wait for its answer. Serialized per box.

        If a caller is on this thread — a sandboxed tool that reached us
        through ``service.call``, say — the box answers *as* that caller for
        the duration: its chain descends from the caller's rather than
        starting beside it, and its Requests are answered from the caller's
        world rather than the kernel's. Serialization is what makes swapping
        live state safe here; one call is in flight at a time by construction.

        A call with no caller — a poll tick, a bus delivery, a hook — keeps
        the box's own chain and context, which is right: a service acting on
        its own initiative is not acting for anybody.

        ``target`` names which occupant, for a box holding several services;
        empty means the only one. It is keyword-only, and that is not a style
        choice: the first positional here is already ``method``, and
        ``sandbox/hooks.py`` documents being bitten once by exactly that
        collision — a ``chain=`` parameter that a caller passed positionally
        and silently invoked as a method name.
        """
        if not self._alive:
            return Result.failure(f"box {self.name!r} is not running")
        from . import provenance

        with self._lock:
            if not self._alive:
                return Result.failure(f"box {self.name!r} is not running")
            caller = provenance.current()
            if caller is None:
                return self._call(method, args, kwargs, target)
            base_chain = self.execution.chain
            base_context = self.execution.context
            self.execution.chain = caller.chain.push(self.name)
            if caller.context is not None:
                self.execution.context = caller.context
            try:
                return self._call(method, args, kwargs, target)
            finally:
                self.execution.chain = base_chain
                self.execution.context = base_context

    def stop(self, timeout: float = 10.0) -> Result:
        """Shut down: ask, then starve, then kill."""
        raise NotImplementedError

    def _call(self, method: str, args: tuple, kwargs: dict,
              target: str = "") -> Result:
        """Runner-specific call."""
        raise NotImplementedError

    def _receiver(self, targets: dict, target: str):
        """Resolve which occupant a call is for, or None.

        Shared by both runners because both hold the same question: a box with
        one occupant serves it whatever the call said, and a box with several
        serves only a name it recognises.
        """
        if target:
            return targets.get(target)
        return next(iter(targets.values())) if len(targets) == 1 else None


class InProcessBox(PersistentBox):
    """A resident box on this side of the boundary.

    The instance simply stays in memory, so a call is a direct invocation —
    no serve loop needed, because there is no boundary to cross. Each call
    still runs on its own worker thread so the per-call deadline and the
    starvation path behave exactly as they do for a subprocess.
    """

    def __init__(self, target, name: str, chain=None, call_timeout=None,
                 manage_lifecycle: bool = True):
        super().__init__(name, chain, call_timeout)
        # One occupant, or a ``{name: object}`` mapping when several services
        # share this box. ``target`` stays the single case so nothing that
        # reaches for it has to learn about the other.
        self.targets = target if isinstance(target, dict) else {"": target}
        self.target = next(iter(self.targets.values()))
        self.manage_lifecycle = manage_lifecycle
        self.sdk = SDK(None)   # channel attached at start()

    def start(self, interpreter: Interpreter,
              timeout: float = DEFAULT_START_TIMEOUT) -> Result:
        """Run each occupant's ``start`` if it has one, then stand by."""
        self.sdk = SDK(interpreter.channel(self.execution))
        self._interpreter = interpreter
        if self.manage_lifecycle:
            for occupant in self.targets.values():
                start_fn = getattr(occupant, "start", None)
                if callable(start_fn):
                    outcome = self._invoke(start_fn, (), {},
                                           clamp_timeout(timeout))
                    if not outcome.ok:
                        return outcome
        self._alive = True
        return Result(data=True)

    def _invoke(
        self,
        fn,
        args: tuple,
        kwargs: dict,
        deadline: float,
    ) -> Result:
        """Run one callable on a worker thread under a deadline."""
        from .guest.channel import Terminated
        from .guest.requests import RequestFailed

        from .watchdog import TICK, overdue as is_overdue

        box: dict = {}
        done = threading.Event()

        def _worker():
            """Call it and capture whatever comes back."""
            try:
                raw = fn(self.sdk, *args, **kwargs)
                box["result"] = raw if isinstance(raw, Result) else Result(
                    data=raw)
            except Terminated as stop:
                # A cancelled execution is the kernel tearing this box down,
                # not the code answering. Reporting it as ``ok`` with no data
                # is what let a starved REPL look healthy: ``_drive_polls``
                # read the success, reset its failure count, and spun on a
                # dead box forever while the user typed into nothing. The only
                # ``Terminated`` carrying a real value comes from
                # ``sdk.respond``, which a persistent box may not use.
                box["result"] = (
                    Result.failure(f"box {self.name!r} was cancelled")
                    if self.execution.cancelled
                    else Result(data=stop.value))
            except RequestFailed as failed:
                box["result"] = failed.result
            except Exception as exc:
                box["result"] = Result.failure(
                    f"{type(exc).__name__}: {exc}", code=ERROR_GUEST_FAULT,
                    traceback=guest_traceback(exc, drop=(__file__,)))
            finally:
                done.set()

        threading.Thread(target=_worker, daemon=True,
                         name=f"box-{self.name}").start()

        # Wait on the guest, not on the clock. ``running_for`` discounts the
        # time the kernel spent answering this execution's Requests, so a call
        # that spends two minutes inside sdk.ui.ask or an escorted model call
        # is never mistaken for a hung one — while a runaway that spins on
        # Requests still runs out its deadline, because it is never blocked
        # for long.
        started = time.monotonic()
        expired = False
        while not done.wait(timeout=TICK):
            if is_overdue(self.execution, started, deadline):
                expired = True
                break

        if expired:
            # Starve it: the thread survives, but its next Request is refused
            # so it can no longer affect anything. For a *resident* box that
            # also ends the box, because cancellation is per-execution and a
            # resident box has exactly one for its whole life — there is no
            # way back. Two further reasons the box cannot be reused: the
            # starved worker is still alive, so a later call would put two
            # threads on one Execution and two Requests in one inbox; and
            # every later call would answer ``Terminated`` immediately. Saying
            # the box is dead lets ``_drive_polls`` stop and report, instead
            # of a frontend that accepts keystrokes and does nothing.
            self._interpreter.cancel(self.execution)
            self._alive = False
            return Result.failure(
                f"timed out after {deadline:.1f}s of running")
        return box.get("result", Result(data=None))

    def _call(self, method: str, args: tuple, kwargs: dict,
              target: str = "") -> Result:
        """Invoke a method on the addressed resident instance."""
        receiver = self._receiver(self.targets, target)
        if receiver is None:
            return Result.failure(
                f"no such target: {target!r}; this box holds "
                f"{sorted(self.targets)}")
        fn = getattr(receiver, method, None)
        if not callable(fn):
            return Result.failure(f"no such method: {method!r}")
        return self._invoke(fn, args, kwargs, self.call_timeout)

    def stop(self, timeout: float = 10.0) -> Result:
        """Ask each occupant to stop, then starve. No kill for a thread."""
        if not self._alive:
            return Result(data=False)
        self._alive = False
        if self.manage_lifecycle:
            for occupant in self.targets.values():
                stop_fn = getattr(occupant, "stop", None)
                if callable(stop_fn):
                    self._invoke(stop_fn, (), {}, clamp_timeout(timeout))
        self._interpreter.cancel(self.execution)
        unload_box(self.name)
        return Result(data=True)


class SubprocessBox(PersistentBox):
    """A resident box behind a process boundary."""

    def __init__(self, module_path: str, entry: str, name: str,
                 chain=None, call_timeout=None, box_root=None,
                 memory_mb: int | None = None, extra_roots=(),
                 manage_lifecycle: bool = True, digest: str = "",
                 entries=(), parsers=()):
        super().__init__(name, chain, call_timeout)
        self.module_path = str(module_path)
        self.entry = entry
        # Several plugin classes to instantiate from one module import. Empty
        # for the ordinary single-occupant box, in which case ``entry`` alone
        # names what to load.
        self.entries = list(entries or [])
        self.box_root = box_root
        self.extra_roots = list(extra_roots or [])
        # Parser files the host resolved from this plugin's declared
        # ``parse_modalities``, imported into the box ahead of the entry.
        self.parsers = list(parsers or [])
        self.memory_mb = memory_mb
        self.manage_lifecycle = manage_lifecycle
        self.digest = digest
        self.proc = None
        self._interpreter = None

    def start(self, interpreter: Interpreter,
              timeout: float = DEFAULT_START_TIMEOUT) -> Result:
        """Spawn the child, load the target, and wait for it to stand by."""
        self._interpreter = interpreter
        self.proc = subprocess_for(str(GUEST_ROOT))
        deadline = clamp_timeout(timeout)

        # Same reasoning as ``_call``: a guest whose ``start`` is waiting on
        # the kernel to read a file or fetch a model is not a guest that has
        # hung, so the start deadline counts guest time too.
        ticket = WATCHDOG.watch(self.execution, deadline, self._kill)
        try:
            ok = send(self.proc, {
                "kind": protocol.START,
                "module": self.module_path,
                "func": self.entry,
                "entries": self.entries,
                "persistent": True,
                "box": self.name,
                "root": self.box_root,
                "extra_roots": self.extra_roots,
                "parsers": self.parsers,
                "memory_mb": self.memory_mb,
                "cpu_seconds": None,
                "manage_lifecycle": self.manage_lifecycle,
                "digest": self.digest,
            })
            if not ok:
                return Result.failure("child exited before it could start")
            message = service_until(interpreter, self.execution, self.proc,
                                    {protocol.READY, protocol.FAULT})
        finally:
            WATCHDOG.release(ticket)

        if message is None:
            self._reap()
            return Result.failure(f"box {self.name!r} died while starting")
        if message["kind"] == protocol.FAULT:
            self._reap()
            return fault_result(message, self.name)

        self._alive = True
        return Result(data=True)

    def _call(self, method: str, args: tuple, kwargs: dict,
              target: str = "") -> Result:
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
        # Watched rather than timed. This thread is about to block reading the
        # pipe, so it cannot check anything itself; and the deadline counts
        # *guest* time, so a call that waits two minutes on a model the kernel
        # is fetching for it is not overdue at all. A shared watchdog also
        # spares us a thread per call, which at a 50 ms poll was twenty a
        # second for the life of the process.
        ticket = WATCHDOG.watch(self.execution, self.call_timeout, self._kill)
        try:
            # Packed for the same reason ``Request.args`` is: a caller handing
            # a service raw bytes — an image to OCR, a vector to store — must
            # not depend on which side of a pipe the service happens to be on.
            # The answer needs nothing here; it travels as a Result, which
            # packs itself.
            if not send(self.proc, {
                "kind": protocol.CALL,
                "method": method,
                "target": target,
                "args": protocol.pack(list(args)),
                "kwargs": protocol.pack(kwargs),
            }):
                self._alive = False
                return Result.failure("box channel closed")
            message = service_until(self._interpreter, self.execution,
                                    self.proc,
                                    {protocol.RETURN, protocol.FAULT})
        finally:
            WATCHDOG.release(ticket)

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
        """Last resort, and the one thing a thread can never do.

        Marks the box dead as well as killing the process. Starvation is
        per-execution and a resident box has one execution for its whole life,
        so there is no way back from here — and a box that says it is alive
        while every Request it makes is refused is the shape of bug that had a
        frontend accepting keystrokes and doing nothing with them.
        """
        self._starve()
        self._alive = False
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
             extra_roots=(), manage_lifecycle: bool = True,
             digest: str = "", entries=(), parsers=()) -> PersistentBox:
    """Load a resident box and return a handle to call into.

    ``entry`` names a plugin class, or is empty for a bare script — in which
    case the module itself is the object, its functions are the methods, and
    its globals are the state that persists. That is the whole of a scratchpad
    server.

    ``entries`` names several plugin classes instead, all instantiated from
    one module import and addressed by ``call(..., target=name)``. Only
    services use it, and only because they share something expensive.

    Raises :class:`BoxError` if it will not start, since a handle to a box
    that never loaded is not a useful thing to hand back.
    """
    if isolated:
        box = SubprocessBox(module_path, entry, name, chain=chain,
                            call_timeout=call_timeout, box_root=box_root,
                            memory_mb=memory_mb, extra_roots=extra_roots,
                            manage_lifecycle=manage_lifecycle, digest=digest,
                            entries=entries, parsers=parsers)
    else:
        # In-process the box is this process, so the declared parsers are
        # imported here — before the entry, so a module-scope route lookup in
        # the plugin finds them already collected.
        install_parsers(parsers, box_name=name, root=box_root)
        if entries:
            target = load_entries(module_path, entries, box_name=name,
                                  root=box_root, extra_roots=extra_roots,
                                  digest=digest)
        else:
            target = load_entry(module_path, entry, box_name=name,
                                root=box_root, bound=False,
                                extra_roots=extra_roots, digest=digest)
        box = InProcessBox(target, name, chain=chain,
                           call_timeout=call_timeout,
                           manage_lifecycle=manage_lifecycle)

    outcome = box.start(interpreter, timeout=start_timeout)
    if not outcome.ok:
        # Carry the stack: this is the case it is worth the most in. "The box
        # did not start" and "line 41 of your service, KeyError" are the same
        # event, and only one of them can be acted on.
        raise BoxError(f"{name}: {outcome.error}"
                       + (f"\n{outcome.traceback}" if outcome.traceback else ""))
    return box
