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

Cancellation works by starvation. A thread cannot be killed, but every effect
must pass through this loop, so refusing to service a cancelled execution's
Requests stops it doing anything at its very next Request.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .handlers import HANDLERS
from .policy import Chain, Decision, classify
from .guest.channel import Terminated
from .guest.requests import SELF_RESPOND, Request, Result

logger = logging.getLogger("Sandbox")

# A plugin may ask for a longer leash; it does not get to grant itself one.
# It declares intent, the kernel clamps. (See the security contract: a
# self-evolving agent may change what it can ask for, but not what it is
# authorized to affect.)
MAX_TIMEOUT_SECONDS = 600.0
DEFAULT_TIMEOUT_SECONDS = 30.0


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


class Interpreter:
    """Classifies and services Requests from running sandboxed code."""

    def __init__(self, approve=None, max_workers: int = 4, record=None,
                 context=None):
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
        """
        self._approve = approve
        self._record = record
        self.context = context
        self._gate_queue: queue.Queue = queue.Queue()
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="sandbox-exec")
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
            execution.inbox.put(Result.refusal("sandbox is shutting down"))

    def _gate_one(self, execution: Execution, request: Request):
        """Classify a single Request and route it."""
        # Starvation: a cancelled execution is answered, never serviced.
        if execution.cancelled:
            execution.inbox.put(Result.refusal("execution cancelled"))
            return

        decision = classify(request, execution.chain)

        if not decision.safe:
            if self._approve is None:
                self._settle(execution, request, decision,
                             Result.refusal(decision.reason))
                return
            try:
                allowed = self._approve(execution.chain, request, decision)
            except Exception:
                logger.exception("approval callback raised")
                allowed = False
            if not allowed:
                self._settle(execution, request, decision,
                             Result.refusal(decision.reason))
                return

        # Execution leaves the gate immediately so slow work never blocks
        # classification for anyone else.
        self._pool.submit(self._execute, execution, request, decision)

    def _execute(self, execution: Execution, request: Request,
                 decision: Decision):
        """Run the handler off the gate thread and return the result."""
        handler = HANDLERS.get(request.type)
        if handler is None:
            result = Result.failure(f"no handler for {request.type}")
        else:
            try:
                context = (execution.context if execution.context is not None
                           else self.context)
                result = handler(context, request.args)
            except Exception as exc:
                logger.exception("handler failed: %s", request.type)
                result = Result.failure(f"handler error: {exc}")
        self._settle(execution, request, decision, result)

    def _settle(self, execution: Execution, request: Request,
                decision: Decision, result: Result):
        """Record the Request and resume the waiting code."""
        if self._record is not None:
            try:
                self._record(execution.chain, request, decision, result)
            except Exception:
                logger.exception("ledger write failed")
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
            return Result.refusal("sandbox is shutting down")
        self._gate_queue.put((execution, request))
        return execution.inbox.get()

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
        return result

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
