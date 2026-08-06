"""The drive loop — the kernel side of the sandbox.

This is the whole architecture; everything else hangs off it. Sandboxed code
runs somewhere it cannot act (a worker thread now, a subprocess later) and
sends Requests here. The interpreter classifies each one, executes the allowed
ones, and hands the result back so the code resumes.

**The servicer is split in two, deliberately.** Classification is serial — one
gate thread, so provenance and policy have a single ordering point — but
execution is not. A plugin's 30-second HTTP call is dispatched to a pool, so it
never blocks anyone else's classification. Concurrency is cheap here;
*granularity* is what costs, which is why the SDK offers batch forms.

**Asking the user leaves the gate for the same reason, only more so.** A
dialog is drawn by a frontend, and a frontend reaches the kernel only through
Requests that arrive at this gate — so asking on the gate thread meant the
question could never be shown and the answer could never be read.

Cancellation works by starvation. A thread cannot be killed, but every effect
must pass through this loop, so refusing to service a cancelled execution's
Requests stops it doing anything at its very next Request.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import epoch, provenance
from .guest import protocol
from .handlers import HANDLERS
from .policy import Chain, Decision, classify
from .guest.channel import Terminated
from .guest.codes import (ERROR_APPROVAL_DECLINED, ERROR_CANCELLED,
                         ERROR_HANDLER_ERROR, ERROR_NO_HANDLER,
                         ERROR_SHUTTING_DOWN)
from .guest.requests import SELF_RESPOND, Request, Result

logger = logging.getLogger("Sandbox")

# A plugin may ask for a longer leash; it does not get to grant itself one.
# It declares intent, the kernel clamps. (See the security contract: a
# self-evolving agent may change what it can ask for, but not what it is
# authorized to affect.)
MAX_TIMEOUT_SECONDS = 600.0

#: What anything that declares nothing gets. Sixty rather than thirty because
#: this is a deadline on *running* time — waiting on the kernel is discounted —
#: so the only thing it ever cuts short is code genuinely computing, and thirty
#: seconds of computation is a low bar for the work scripts and tasks are
#: actually reached for. The cost of raising it is that a hung box takes twice
#: as long to report, which is bounded and visible; the cost of leaving it low
#: is honest work killed with the report blaming the plugin, which is the
#: failure this file's history is mostly about.
DEFAULT_TIMEOUT_SECONDS = 60.0

# Handlers block, and several of them block for a long time on purpose:
# ``ui.ask`` waits five minutes for a person, ``proc.run`` two for a command,
# ``net.http`` thirty seconds, and ``tool.call`` / ``service.call`` /
# ``command.call`` re-enter the sandbox and wait for whatever *that* does. Each
# one occupies a worker while it waits, so a small pool is not a throughput
# limit but a liveness one: four workers meant four simultaneous questions
# stopped the sandbox servicing anything at all, including the Requests the
# frontend needed to draw those questions.
#
# Threads waiting on I/O are cheap and the pool grows lazily, so the ceiling
# costs nothing until it is used.
DEFAULT_MAX_WORKERS = 16
# Asking is slower still and must never queue behind execution, so its pool is
# sized for concurrent dialogs rather than concurrent work.
DEFAULT_MAX_APPROVALS = 8


def _shutting_down() -> Result:
    """The answer to anything that arrives while the sandbox is closing."""
    return Result.refusal("sandbox is shutting down",
                          code=ERROR_SHUTTING_DOWN)


def _cancelled() -> Result:
    """The answer to a cancelled execution — answered, never serviced."""
    return Result.refusal("execution cancelled", code=ERROR_CANCELLED)


def clamp_timeout(declared: float | None) -> float:
    """Clamp a plugin's self-declared timeout to the kernel's ceiling."""
    if not declared or declared <= 0:
        return DEFAULT_TIMEOUT_SECONDS
    return min(float(declared), MAX_TIMEOUT_SECONDS)


@dataclass
class Execution:
    """One running piece of sandboxed code.

    Owns the two queues that carry Requests out and Results back. The chain is
    kernel-side state — the code inside never sees it and cannot alter it.
    """
    name: str
    chain: Chain
    outbox: queue.Queue = field(default_factory=queue.Queue)
    inbox: queue.Queue = field(default_factory=queue.Queue)
    logs: list = field(default_factory=list)
    cancelled: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    response: Result | None = None

    # The kernel context this execution's handlers answer from. Carried per
    # execution rather than held on the interpreter because contexts differ
    # by session and user, and two tool calls can be in flight at once — a
    # single shared context would answer one of them from the other's world.
    context: object = None

    # Time accounting for the deadline. A box waiting on a 90-second model
    # call is not hung — it is waiting for something the kernel itself
    # started — so that time belongs to the kernel and must not be charged to
    # the guest. But "time since the guest last did anything" is the wrong
    # correction: a runaway that hammers Requests in a loop is never idle for
    # long, and would be immortal. What is charged is therefore *elapsed time
    # minus time blocked on us*, which subtracts one long wait in full and a
    # spin loop's thousand short ones to nearly nothing.
    in_flight: int = 0
    blocked_total: float = 0.0
    _blocked_since: float = 0.0
    _progress_lock: threading.Lock = field(default_factory=threading.Lock)

    def entered(self):
        """One of this execution's Requests has reached a handler."""
        with self._progress_lock:
            if self.in_flight == 0:
                self._blocked_since = time.monotonic()
            self.in_flight += 1

    def left(self):
        """A handler has answered; the guest is about to run again."""
        with self._progress_lock:
            self.in_flight = max(0, self.in_flight - 1)
            if self.in_flight == 0 and self._blocked_since:
                self.blocked_total += time.monotonic() - self._blocked_since
                self._blocked_since = 0.0

    def blocked_for(self, now: float) -> float:
        """Total seconds the kernel has owed this execution an answer."""
        with self._progress_lock:
            pending = (now - self._blocked_since
                       if self.in_flight > 0 and self._blocked_since else 0.0)
            return self.blocked_total + pending

    def running_for(self, since: float, now: float | None = None) -> float:
        """Seconds of *guest* execution since ``since``.

        This is what a deadline measures. Waiting on the kernel does not
        count; running, spinning, and blocking on something the guest chose to
        do itself all do.
        """
        now = time.monotonic() if now is None else now
        return max(0.0, (now - since) - self.blocked_for(now))


class Interpreter:
    """Classifies and services Requests from running sandboxed code."""

    def __init__(self, approve=None, max_workers: int = DEFAULT_MAX_WORKERS,
                 record=None,
                 context=None, context_factory=None):
        """
        approve:
            callable(chain, request, decision) -> bool. Asks the user. When
            absent, unsafe Requests are refused — the safe default, and the
            same one the kernel uses when every permission hook abstains.
        record:
            callable(chain, request, decision, result) -> None. The ledger
            sink. Best-effort: the ledger observes the system, never breaks it.
        context:
            The kernel's context object, handed to every handler. This is
            Second Brain's ``SecondBrainContext`` — the same bag plugins used
            to receive, now sitting on the *host* side of the boundary where
            it answers Requests instead of being handed out. Never crosses
            into the guest.
        context_factory:
            callable(session_key) -> context, for executions that arrive
            without one. A resident box is the case that needs it: a service
            opens long before anything calls into it and has no session of its
            own, so nothing can hand it a context at open time and it would
            otherwise answer every Request from nothing at all.
        """
        self._approve = approve
        self._record = record
        self.context = context
        self._context_factory = context_factory
        self._approver_bound = approve is not None
        self._gate_queue: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="sandbox-exec")
        # Asking a human is the slowest thing this system does, and it gets its
        # own pool for two separate reasons. It is not the *gate*, because the
        # gate is the single ordering point for every Request in the process —
        # including the ones the frontend rendering the dialog has to make to
        # print it and to read the answer. And it is not the *execution* pool,
        # because a dialog that sits for the full DIALOG_TIMEOUT would occupy a
        # worker that running plugins need. More than one worker so an
        # unattended session's immediate refusal is never queued behind a
        # foreground dialog nobody has answered yet.
        self._approvals = ThreadPoolExecutor(
            max_workers=max(2, min(max_workers, DEFAULT_MAX_APPROVALS)),
            thread_name_prefix="sandbox-approve")
        self._gate = threading.Thread(target=self._gate_loop, daemon=True,
                                      name="sandbox-gate")
        self._running = True
        self._gate.start()

    # ──────────────────────────────────────────────────────────────
    # The gate: serial, cheap, the single ordering point for policy.
    # ──────────────────────────────────────────────────────────────

    def _gate_loop(self):
        """Classify Requests one at a time; dispatch the work elsewhere."""
        while self._running:
            item = self._gate_queue.get()
            if item is None:
                break
            execution, request = item
            try:
                self._gate_one(execution, request)
            except Exception:
                logger.exception("gate failed on %s", request.type)
                execution.inbox.put(Result.failure("kernel error"))
        self._drain()

    def _drain(self):
        """Answer whatever was queued when the gate stopped.

        A Request left unanswered is a thread blocked forever: the caller is
        sitting on ``inbox.get()`` and the gate that would have replied is
        gone. Shutdown has to answer its remaining mail.
        """
        while True:
            try:
                item = self._gate_queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                continue
            execution, _ = item
            execution.inbox.put(_shutting_down())

    def _gate_one(self, execution: Execution, request: Request):
        """Classify a single Request and route it."""
        # Starvation: a cancelled execution is answered, never serviced.
        if execution.cancelled:
            execution.inbox.put(_cancelled())
            return

        decision = classify(request, execution.chain)

        if not decision.safe:
            if self._approve is None:
                self._settle(execution, request, decision,
                             Result.refusal(decision.reason,
                                        code=ERROR_APPROVAL_DECLINED))
                return
            # Asking leaves the gate too, and for a sharper reason than slow
            # work does. The approver renders a dialog through a frontend, and
            # that frontend reaches the kernel only by making Requests of its
            # own — which arrive *here*. Asking on this thread meant the dialog
            # could never be drawn and the answer could never be read, so the
            # REPL froze until the wait expired. It also let the approver block
            # on a lock held by the very thread waiting for its answer.
            self._approvals.submit(self._ask_then_execute,
                                   execution, request, decision)
            return

        # Execution leaves the gate immediately so slow work never blocks
        # classification for anyone else.
        self._hand_to_pool(execution, request, decision)

    def _hand_to_pool(self, execution: Execution, request: Request,
                      decision: Decision):
        """Give a permitted Request to the execution pool.

        Guarded because shutdown closes that pool, and a Request that is never
        answered is a thread waiting on ``inbox.get()`` for as long as the
        process lives. Refusing is the one thing worse than executing and the
        one thing better than hanging.
        """
        try:
            self._pool.submit(self._execute, execution, request, decision)
        except RuntimeError:
            self._settle(execution, request, decision,
                         _shutting_down())

    def _ask_then_execute(self, execution: Execution, request: Request,
                          decision: Decision):
        """Put one unsafe Request to the user, then run it or refuse it."""
        # Re-checked because the wait for a worker is time the execution could
        # have been cancelled in, and a cancelled execution is answered rather
        # than serviced — the same rule the gate applies on the way in.
        if execution.cancelled or not self._running:
            execution.inbox.put(
                _cancelled() if execution.cancelled else _shutting_down())
            return
        # Bracketed in the same accounting a handler is, and for the same
        # reason: the guest is waiting on us. A dialog may sit for
        # DIALOG_TIMEOUT (300s), far longer than any default deadline, so charging
        # it to the guest killed every unsafe Request a person read carefully —
        # the tool reported a timeout blaming the plugin for a question the
        # user had not answered yet.
        execution.entered()
        try:
            allowed = self._approve(execution.chain, request, decision)
        except Exception:
            logger.exception("approval callback raised")
            allowed = False
        finally:
            execution.left()
        if not allowed:
            self._settle(execution, request, decision,
                         Result.refusal(decision.reason,
                                        code=ERROR_APPROVAL_DECLINED))
            return
        # Shutdown may have begun while the dialog was up, and so may
        # cancellation. The execution still has to be answered — an unanswered
        # Request is a thread blocked for the life of the process — but it does
        # not get to act on a yes given to a sandbox that is already going
        # away, nor on one whose caller has already been told it failed. That
        # second case is the worse of the two: the handler ran, the job was
        # created, and the model was told otherwise.
        if execution.cancelled:
            self._settle(execution, request, decision, _cancelled())
            return
        if not self._running:
            self._settle(execution, request, decision,
                         _shutting_down())
            return
        self._hand_to_pool(execution, request, decision)

    def _execute(self, execution: Execution, request: Request,
                 decision: Decision):
        """Run the handler off the gate thread and return the result.

        The handler runs marked as *serving this execution*, so anything it
        reaches that re-enters the sandbox — ``tool.call``, ``service.call``,
        ``command.call`` — can find out who is asking and descend from that
        chain rather than starting a fresh one beside it.
        """
        execution.entered()
        try:
            handler = HANDLERS.get(request.type)
            if handler is None:
                result = Result.failure(
                    f"no handler for {request.type}",
                    code=ERROR_NO_HANDLER)
            else:
                context = self._context_for(execution)
                try:
                    with provenance.serving(execution.chain, context,
                                            execution):
                        result = handler(context, request.args)
                except Exception as exc:
                    logger.exception("handler failed: %s", request.type)
                    result = Result.failure(f"handler error: {exc}",
                                            code=ERROR_HANDLER_ERROR)
        finally:
            execution.left()
        self._settle(execution, request, decision, result)

    def _context_for(self, execution: Execution):
        """The host object this execution's Requests are answered from.

        Resolved once and cached on the execution rather than rebuilt per
        Request: a resident box makes thousands, and they all belong to the
        same asker.
        """
        if execution.context is None and self._context_factory is not None:
            try:
                execution.context = self._context_factory(None)
            except Exception:
                logger.exception("could not build a context for %s",
                                 execution.name)
        return (execution.context if execution.context is not None
                else self.context)

    def context_for_session(self, session_key: str):
        """A host context belonging to one session, or ``None``.

        The counterpart to :meth:`_context_for`, which answers for a box acting
        on its own initiative and therefore builds the *kernel's* context. This
        answers for a box the kernel has called **on a session's behalf** — a
        hook standing at a doorway is the case — where the session is a fact
        the kernel holds and the box has no way to learn.

        It is deliberately only about the context and never about the chain.
        Which session's rows a call reads is a different question from who is
        asking for them, and only the second decides whether anything may be
        approved. See ``PersistentBox.call``.
        """
        if not session_key or self._context_factory is None:
            return None
        try:
            return self._context_factory(session_key)
        except Exception:
            logger.exception("could not build a context for session %s",
                             session_key)
            return None

    def set_context_factory(self, factory) -> None:
        """Install the context builder, once the kernel has parts to build from.

        Boot order forces this exactly as it forces ``set_approver``: services
        are discovered and loaded before the orchestrator, the tool registry
        and the runtime exist, so the factory is installed early and reads
        whatever has been wired by the time it is actually called.
        """
        self._context_factory = factory

    def set_record(self, record) -> None:
        """Install the ledger sink, once there is a database to write to."""
        self._record = record

    def _settle(self, execution: Execution, request: Request,
                decision: Decision, result: Result):
        """Record the Request, note whether it changed anything, and resume.

        The epoch bump sits here for the reason the ledger sink does: this is
        the one funnel every serviced Request passes through, so a single
        counter can speak for the whole sandbox. Its own guard, because an
        observer must never break a Request — ``epoch.bump`` cannot realistically
        raise, but neither could the ledger write, and the cost of being wrong
        is a plugin's effect failing for a bookkeeping reason.
        """
        if self._record is not None:
            try:
                self._record(execution.chain, request, decision, result)
            except Exception:
                logger.exception("ledger write failed")
        try:
            if epoch.counts(request, result):
                epoch.bump()
        except Exception:
            logger.exception("epoch bump failed")
        execution.inbox.put(result)

    # ──────────────────────────────────────────────────────────────
    # The channel sandboxed code talks through.
    # ──────────────────────────────────────────────────────────────

    def submit(self, execution: Execution, request: Request) -> Result:
        """Send a Request and block until the kernel answers.

        Called from the plugin's own thread. Blocking here is not new cost:
        code doing ``open(path).read()`` blocks its thread today. This moves
        *where* the wait happens, not whether there is one.
        """
        if request.type == SELF_RESPOND:
            execution.response = Result(data=request.args.get("value"))
            execution.finished.set()
            return Result(data=None)
        # Never queue onto a stopped gate: nothing would answer, and the
        # caller would wait on ``inbox.get()`` for as long as the process
        # lives.
        if not self._running:
            return _shutting_down()
        self._gate_queue.put((execution, request))
        return execution.inbox.get()

    def set_approver(self, approve) -> None:
        """Install the callable that asks the user, once there is one to ask.

        Boot order forces this: services and frontends are discovered and
        loaded before the conversation runtime exists, so a sandbox built
        early has nobody to put a dialog in front of. Without it every unsafe
        Request is refused outright for the life of the process — which is
        the correct default and the wrong permanent state.
        """
        self._approve = approve
        self._approver_bound = approve is not None

    @property
    def can_ask(self) -> bool:
        """Whether an unsafe Request can reach a human at all."""
        return self._approver_bound

    def cancel(self, execution: Execution):
        """Neutralize a running execution at its next Request."""
        execution.cancelled = True

    def channel(self, execution: Execution) -> "InterpreterChannel":
        """Build the transport an in-process SDK talks through."""
        return InterpreterChannel(self, execution)

    def shutdown(self):
        """Stop the gate, then close the execution pool.

        The gate may already have dequeued a Request when shutdown begins.
        Joining it before closing the pool lets that Request either dispatch
        normally or be refused by the drain, instead of racing a late
        ``submit`` against a closed executor.
        """
        self._running = False
        self._gate_queue.put(None)
        if self._gate is not threading.current_thread():
            self._gate.join()
        # Approvals close before the execution pool because an approved
        # Request is submitted *to* that pool from an approval worker. Neither
        # waits: a dialog nobody answered would otherwise hold shutdown open
        # for the full DIALOG_TIMEOUT. ``_running`` is already False, so a
        # worker coming back from a dialog refuses instead of executing.
        self._approvals.shutdown(wait=False)
        self._pool.shutdown(wait=False)


class InterpreterChannel:
    """The host-side transport: a queue hop to the gate thread.

    The guest's counterpart is ``guest.channel.PipeChannel``. Both satisfy the
    same two-method shape — ``send`` and ``log`` — which is the entire reason
    one SDK serves every runner.
    """

    def __init__(self, interpreter: Interpreter, execution: Execution):
        self._interpreter = interpreter
        self._execution = execution

    def send(self, request: Request) -> Result:
        """Send a Request and block for the kernel's answer.

        A cancelled execution unwinds rather than receiving a failure. The two
        look alike but are not: a *denial* is the user saying no, which code
        should handle and carry on from, while a *cancellation* is the kernel
        tearing the execution down, and code that carries on from that spins
        forever asking questions nobody will answer.

        This is the in-process counterpart of ``PipeChannel`` unwinding when
        its channel closes.
        """
        if self._execution.cancelled:
            raise Terminated(None)
        result = self._interpreter.submit(self._execution, request)
        if self._execution.cancelled:
            raise Terminated(None)
        try:
            return result.crossing()
        except protocol.ProtocolError as exc:
            return Result.failure(
                f"handler returned an unsendable result: {exc}",
                code=ERROR_HANDLER_ERROR)

    def notify(self, request: Request) -> None:
        """Send a Request without waiting for its answer.

        In-process there is no round trip to save, so this differs from
        ``send`` only in discarding the Result — which is the point: the two
        runners must agree on what a plugin can observe, or code written
        against one would misbehave under the other. Cancellation still
        unwinds, because a cancelled execution must stop making Requests
        however it is making them.
        """
        if self._execution.cancelled:
            raise Terminated(None)
        self._interpreter.submit(self._execution, request)
        if self._execution.cancelled:
            raise Terminated(None)

    def log(self, level: str, message: str) -> None:
        """Buffer a log line for the runner to emit."""
        self._execution.logs.append((level, message))
