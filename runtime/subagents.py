"""Running an agent on a prompt, in the background, in its own conversation.

This is kernel routing and nothing else, which is why it lives in the kernel
rather than in a package. Spawning a subagent is ``runtime.iterate_agent_turn``
on a session key that is not the active one — every hard part is about *what
happens around* that call: which conversation it runs in, who is told when it
finishes, and what ends it when it will not end itself.

**A handle, not a table scan.** The store version located a running child by
matching ``payload_json LIKE '%"conversation_id": N,%'`` against ``task_runs``,
because a bus event has no identity to hand back. A spawn has one here, and two
whole mechanisms fall away with it: the side-set of "children the parent already
cancelled" (needed only to suppress a stale completion echo from a run that had
no state of its own) and the second copy of the same query in the barrier.

**One delivery, decided by whoever collects first.** A finished child's report
is stored on its handle and delivered exactly once — by an explicit
``collect``, or by the end-of-turn ``barrier`` for children nobody collected.
The worker never queues it directly, which is what makes ``wait=False`` usable
from a script (no turn to hold open) and from a turn (barrier holds it open)
without the two racing to report the same result twice.

**A deadline is a hard cutoff, never a silent drop.** A child still running at
its deadline is cancelled and reported as failed. The model always learns each
child's fate, because "no news" is indistinguishable from "still thinking" and
an agent that guesses will report findings nobody produced.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from events.event_bus import bus
from events.event_channels import SUBAGENT_SPAWN
from state_machine.serialization import save_state_marker

logger = logging.getLogger("Subagents")

# The session-key prefix that makes a child a child. It is never the active
# session key, which is what makes every subagent unattended: its Requests
# build a chain rooted here rather than at ``user``, so nothing can be approved
# on its behalf and every unsafe Request refuses. See sandbox/policy.py.
SESSION_PREFIX = "spawn_subagent:"

DEFAULT_TIMEOUT = 300
DEFAULT_CEILING = 4
# One level: a turn may spawn, a child may not. Nothing in the tree can answer
# a dialog, and a fan-out is multiplicative, so the depth someone is willing to
# run unattended is theirs to choose rather than a constant.
DEFAULT_DEPTH = 1
POLL_SECONDS = 0.25
BARRIER_POLL_SECONDS = 1.0

# How much of a child's report reaches the parent inline. The transcript is
# always complete in the child's own conversation; this is the preview.
NOTICE_CAP = 16000

RUNNING, DONE, FAILED, CANCELLED = "running", "done", "failed", "cancelled"
TERMINAL = {DONE, FAILED, CANCELLED}

# Appended when somebody is waiting for a report: the child's final message IS
# the deliverable. Without this a background agent writes as if a person will
# read the transcript and ask a follow-up, and nobody ever does.
REPORT_FRAMING = (
    "\n\n[Note: you are a background agent; nobody will reply to you. Your "
    "final message is the only thing delivered back to the requester — make "
    "it a complete, self-contained report of your findings.]"
)


def is_subagent_session(session_key: str | None) -> bool:
    """Whether this session key belongs to a spawned child."""
    return str(session_key or "").startswith(SESSION_PREFIX)


@dataclass
class Handle:
    """One spawn: what it is, how it is going, and who is owed the answer."""

    id: str
    conversation_id: int
    title: str
    # How long this child may *run*. The clock starts when a pool worker picks
    # it up, not when it was submitted: a fan-out wider than the pool queues,
    # and a deadline running while queued would cancel the tail of a large
    # fan-out before it ever spoke to a model. ``deadline`` is infinite until
    # then, so a queued child is never overdue.
    timeout: float
    deadline: float = float("inf")
    state: str = RUNNING
    text: str = ""
    error: str = ""
    # The session that started this and is owed the report. None for a
    # scheduled spawn, whose delivery surface is the user-facing push instead
    # — nobody is waiting on a return value, so there is nothing to collect.
    owner: str | None = None
    # Pins delivery to the conversation the spawn was made from. A session that
    # has since switched conversations must not receive the notice; the result
    # still lives in the child's own conversation.
    owner_conversation_id: int | None = None
    # Where this sits in the spawn tree. Depth 0 was started by a real turn;
    # depth 1 by a child of one. This cannot be read off the provenance chain:
    # a subagent's turn is not a nested sandbox call, so its Requests build a
    # *fresh* chain and ``Chain.depth`` is back to zero for every generation.
    # The lineage has to be carried here, which is also what lets a cancel
    # walk down it.
    depth: int = 0
    parent: str | None = None
    collected: bool = False
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def finished(self) -> bool:
        """Whether this child will do nothing further."""
        return self.state in TERMINAL

    def report(self) -> dict:
        """What crosses the sandbox boundary. Plain data, no live objects."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "title": self.title,
            "state": self.state,
            "ok": self.state == DONE,
            "text": self.text,
            "error": self.error,
        }

    def notice(self) -> str:
        """The report as one line of prose for the parent's message queue."""
        if self.state == CANCELLED:
            return (f"[Background agent '{self.title}' TIMED OUT and was "
                    f"cancelled — it delivered no result; do not report "
                    f"anything on its behalf. Partial transcript: "
                    f"conversation #{self.conversation_id}]")
        body = (self.text if self.state == DONE else self.error).strip()
        if len(body) > NOTICE_CAP:
            # Say what is missing, and how much. A preview the model mistakes
            # for the whole report is worse than an obvious truncation.
            body = (body[:NOTICE_CAP]
                    + f" … [report truncated: {len(body):,} chars total, "
                      f"first {NOTICE_CAP:,} shown]")
        state = "finished" if self.state == DONE else "FAILED"
        return (f"[Background agent '{self.title}' {state}] {body} "
                f"(full transcript: conversation #{self.conversation_id})")


class SubagentRegistry:
    """The kernel's live subagents: a pool, a handle each, and a barrier."""

    def __init__(self, runtime=None, config: dict | None = None):
        self.runtime = runtime
        self._config = config or {}
        self._handles: dict[str, Handle] = {}
        self._lock = threading.RLock()
        self._pool: ThreadPoolExecutor | None = None
        self._unsubscribe = None

    # --- lifecycle ----------------------------------------------------

    def bind(self, runtime=None, config: dict | None = None) -> None:
        """Receive the pieces the composition root builds after this one."""
        if runtime is not None:
            self.runtime = runtime
        if config is not None:
            self._config = config

    def start(self) -> None:
        """Begin serving scheduled spawns."""
        if self._unsubscribe is None:
            self._unsubscribe = bus.subscribe(SUBAGENT_SPAWN, self._on_event)

    def stop(self) -> None:
        """Stop serving and end every child still running.

        Cancelling rather than waiting, because the pool's threads are not
        daemons: a child mid-turn would otherwise hold the interpreter open at
        exit and ``/quit`` would appear to hang. Cancellation reaches the
        child's own turn through its session, which is the same path the user
        pressing Ctrl-C uses, so a child stops the way it always stops.
        """
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                logger.exception("could not unsubscribe from %s",
                                 SUBAGENT_SPAWN)
            self._unsubscribe = None
        self.cancel_all()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False)

    @property
    def ceiling(self) -> int:
        """How many children may run at once."""
        config = self._config or getattr(self.runtime, "config", None) or {}
        try:
            return max(1, int(config.get("max_concurrent_subagents")
                              or DEFAULT_CEILING))
        except (TypeError, ValueError):
            return DEFAULT_CEILING

    @property
    def timeout(self) -> int:
        """The ceiling on how long one child may run."""
        config = self._config or getattr(self.runtime, "config", None) or {}
        try:
            return max(1, int(config.get("subagent_timeout_seconds")
                              or DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT

    @property
    def max_depth(self) -> int:
        """How deep the spawn tree may go. 1 = a child may not spawn."""
        config = self._config or getattr(self.runtime, "config", None) or {}
        try:
            return max(1, int(config.get("max_subagent_depth")
                              or DEFAULT_DEPTH))
        except (TypeError, ValueError):
            return DEFAULT_DEPTH

    def _spawner(self, owner: str | None) -> Handle | None:
        """The handle whose child turn is doing this spawning, if any.

        A subagent session's key carries its conversation id, which is what
        connects a running child back to its own handle — the child cannot be
        asked, and would not be believed if it could.
        """
        if not is_subagent_session(owner):
            return None
        try:
            cid = int(str(owner).split(":", 1)[1])
        except (IndexError, ValueError):
            return None
        with self._lock:
            # A running handle first — that is the child actually spawning.
            # Falling back to the newest handle for the conversation matters
            # because a scheduled job reuses its conversation, so the id alone
            # can match several generations of the same job.
            matches = [h for h in self._handles.values()
                       if h.conversation_id == cid]
        running = [h for h in matches if not h.finished]
        return (running or matches or [None])[-1]

    def _executor(self) -> ThreadPoolExecutor:
        """The pool, sized on first use so a config edit before boot counts."""
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.ceiling, thread_name_prefix="Subagent")
            return self._pool

    # --- starting one -------------------------------------------------

    def spawn(
        self,
        prompt: str,
        *,
        title: str = "Subagent",
        attachments=None,
        timeout_seconds: int | None = None,
        conversation_id: int | None = None,
        owner: str | None = None,
        owner_conversation_id: int | None = None,
        user_id: int = 1,
        category: str = "Subagent",
        notification_mode: str | None = "off",
    ) -> Handle:
        """Start a child and return its handle. Raises on a refusal."""
        runtime = self.runtime
        if runtime is None:
            raise RuntimeError("the runtime is not available")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt is required")

        # How deep this would sit. A fan-out is multiplicative, and nobody in
        # the tree can answer a dialog, so the depth a person is willing to
        # run unattended is a setting rather than a constant.
        spawner = self._spawner(owner)
        if spawner is not None:
            depth = spawner.depth + 1
        elif is_subagent_session(owner):
            # A subagent whose own handle we can no longer identify — it was
            # collected, or the registry was restarted under it. Fail closed:
            # the one thing known for certain is that it *is* a child, so its
            # child is at least depth 1. Reading this as depth 0 would let a
            # forgotten handle buy unlimited nesting.
            depth = 1
        else:
            depth = 0
        if depth >= self.max_depth:
            raise PermissionError(
                f"subagents may not nest more than {self.max_depth} deep "
                f"(max_subagent_depth); this would be depth {depth + 1}")

        title = (title or "Subagent").strip() or "Subagent"
        paths = self._attachment_paths(attachments)
        ceiling = self.timeout
        try:
            timeout = min(int(timeout_seconds or ceiling), ceiling)
        except (TypeError, ValueError):
            timeout = ceiling

        cid = self._conversation_for(conversation_id, title, category, user_id,
                                     notification_mode)
        # Two guards worth keeping literal: a child must never drive the
        # conversation a person is looking at, and one conversation runs one
        # child at a time.
        if cid == getattr(runtime, "active_conversation_id", None):
            raise PermissionError(
                "a subagent cannot run in the active conversation — switch "
                "away or let it use its own")
        session = (getattr(runtime, "sessions", None) or {}).get(
            f"{SESSION_PREFIX}{cid}")
        if session is not None and getattr(session, "busy", False):
            raise PermissionError(
                f"a subagent is already running in conversation #{cid}")

        handle = Handle(
            id=uuid.uuid4().hex[:12],
            conversation_id=cid,
            title=title,
            timeout=timeout,
            owner=owner or None,
            owner_conversation_id=owner_conversation_id,
            depth=depth,
            parent=spawner.id if spawner is not None else None,
        )
        with self._lock:
            self._handles[handle.id] = handle
        logger.info("subagent %s spawned: %r in conversation #%s (owner=%s, "
                    "depth=%d, timeout=%ds)",
                    handle.id, title, cid, owner, depth, timeout)
        self._executor().submit(self._run, handle, prompt, paths)
        return handle

    def _conversation_for(self, conversation_id, title, category, user_id,
                          notification_mode) -> int:
        """Reuse a conversation when one was named, else open a fresh one."""
        runtime = self.runtime
        db = getattr(runtime, "db", None)
        if conversation_id is not None:
            try:
                cid = int(conversation_id)
            except (TypeError, ValueError):
                cid = None
            if cid is not None and db is not None and db.get_conversation(cid):
                return cid
        cid = runtime.create_conversation(title, kind="user",
                                          category=category, user_id=user_id)
        if cid is None:
            raise RuntimeError("could not create a conversation for the agent")
        marker = {"conversation_id": cid, "active_agent_profile": "default",
                  "profile_override": "default"}
        # notification_mode off for an interactive spawn: the child's output
        # belongs to the agent that asked for it, delivered through the report,
        # never pushed into the user's chat. A scheduled spawn keeps the
        # default on — the push is its only delivery surface.
        if notification_mode:
            marker["notification_mode"] = notification_mode
        if db is not None:
            save_state_marker(db, cid, marker)
        return cid

    @staticmethod
    def _attachment_paths(attachments) -> list[str]:
        """Validate attachment paths up front, while there is a caller to tell."""
        if isinstance(attachments, str):
            attachments = [attachments]
        out = []
        for raw in attachments or []:
            path = Path(str(raw)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"attachment not found: {path}")
            out.append(str(path))
        return out

    def _attachments(self, paths: list[str]) -> list:
        """Parse attachment paths into the bundle a turn takes."""
        if not paths:
            return []
        from attachments import parse_attachment
        services = getattr(self.runtime, "services", None) or {}
        return [parse_attachment(p, services=services,
                                 config={"max_chars": 4000}) for p in paths]

    # --- running one --------------------------------------------------

    def _run(self, handle: Handle, prompt: str, paths: list[str]) -> None:
        """Drive one child's turn. Never raises — the handle carries the news."""
        session_key = f"{SESSION_PREFIX}{handle.conversation_id}"
        runtime = self.runtime
        try:
            # Cancelled while queued behind the ceiling. Starting now would run
            # work whose failure has already been reported.
            if handle.state == CANCELLED:
                return
            # The clock starts here, not at submission: everything above this
            # line may have been spent queued behind the concurrency ceiling,
            # and charging a child for the queue cancels the tail of a large
            # fan-out before it has said a word.
            handle.deadline = time.time() + handle.timeout
            if handle.owner:
                prompt += REPORT_FRAMING
            runtime.open_session(session_key,
                                 conversation_id=handle.conversation_id)
            try:
                out = runtime.iterate_agent_turn(
                    session_key, prompt,
                    attachments=self._attachments(paths))
            finally:
                runtime.close_session(session_key)
        except Exception as exc:
            logger.exception("subagent %s failed", handle.id)
            self._finish(handle, FAILED, error=str(exc))
            return

        if not out.ok:
            error = ((out.error or {}).get("message")
                     or "\n".join(out.messages) or "the subagent failed")
            self._finish(handle, FAILED, error=error)
            return
        self._finish(handle, DONE, text="\n".join(out.messages))

    def _finish(self, handle: Handle, state: str, *, text: str = "",
                error: str = "") -> None:
        """Settle a handle, unless something already settled it."""
        with self._lock:
            # A cancellation that landed mid-turn wins: the caller was already
            # told this child produced nothing, and a late success would
            # contradict a failure the model has seen.
            if handle.state == CANCELLED:
                handle._done.set()
                return
            handle.state = state
            handle.text = text
            handle.error = error
        handle._done.set()
        if handle.owner is None and state == FAILED:
            # Nobody is going to collect this one, so a failure that is not
            # surfaced here is a failure nobody ever learns about.
            logger.error("scheduled subagent '%s' failed: %s",
                         handle.title, error)
            self._push(f"Scheduled agent '{handle.title}' failed: {error}")

    def _push(self, message: str) -> None:
        """Tell the user something, best-effort."""
        try:
            key = getattr(self.runtime, "active_session_key", None)
            self.runtime.push_message(key, message, source="subagents")
        except Exception:
            logger.exception("could not push a subagent notice")

    # --- collecting -----------------------------------------------------

    def get(self, handle_id: str) -> Handle | None:
        """One handle, or None."""
        with self._lock:
            return self._handles.get(str(handle_id or ""))

    def pending_for(self, owner: str | None) -> list[Handle]:
        """This owner's uncollected children, oldest first."""
        if not owner:
            return []
        with self._lock:
            return [h for h in self._handles.values()
                    if h.owner == owner and not h.collected]

    def collect(self, ids=None, *, owner: str | None = None,
                timeout: float | None = None) -> list[dict]:
        """Wait for children and take their reports.

        ``ids=None`` means every uncollected child this owner started.
        ``timeout=0`` polls without waiting; ``None`` waits until each child's
        own deadline. A child still running when the wait runs out is returned
        as it stands and stays uncollected, so a later call can pick it up.
        """
        if ids is None:
            wanted = self.pending_for(owner)
        else:
            if isinstance(ids, str):
                ids = [ids]
            wanted = [h for h in (self.get(i) for i in ids) if h is not None]
        if not wanted:
            return []

        limit = None if timeout is None else time.time() + max(0.0, float(timeout))
        for handle in wanted:
            self._wait_for(handle, limit)

        out = []
        with self._lock:
            for handle in wanted:
                if handle.finished:
                    handle.collected = True
                out.append(handle.report())
        return out

    def _wait_for(self, handle: Handle, limit: float | None) -> None:
        """Wait on one child, enforcing its deadline as a hard cutoff."""
        while not handle.finished:
            now = time.time()
            if handle.deadline <= now:
                self.cancel(handle.id)
                return
            # Whichever comes first: the caller's patience or the child's
            # deadline. Sliced so a cancelled child is noticed promptly.
            until = handle.deadline if limit is None else min(handle.deadline,
                                                              limit)
            if until <= now:
                return
            if handle._done.wait(min(POLL_SECONDS, until - now)):
                return

    def cancel(self, handle_id: str) -> bool:
        """End a child now, along with anything it started.

        Cancellation walks *down* the lineage. A child that spawned children
        of its own is about to stop making Requests, so nothing would ever
        reach its descendants — they would run to their own deadlines doing
        work for a parent that is gone. The walk is bounded by
        ``max_subagent_depth`` on the way in, so it terminates.
        """
        handle = self.get(handle_id)
        if handle is None or handle.finished:
            return False
        with self._lock:
            handle.state = CANCELLED
            handle.error = "cancelled"
            children = [h.id for h in self._handles.values()
                        if h.parent == handle.id and not h.finished]
        session = (getattr(self.runtime, "sessions", None) or {}).get(
            f"{SESSION_PREFIX}{handle.conversation_id}")
        event = getattr(session, "cancel_event", None)
        if event is not None:
            event.set()
        handle._done.set()
        for child in children:
            self.cancel(child)
        logger.info("subagent %s cancelled (%d descendant(s))",
                    handle.id, len(children))
        return True

    def cancel_for(self, owner: str) -> int:
        """Stop every child this session started. Returns how many.

        What ``/cancel`` reaches for: stopping a turn should stop the work
        that turn set going, or the agent is gone and its children carry on
        spending money on a question nobody is waiting for.
        """
        running = [h.id for h in self.pending_for(owner) if not h.finished]
        return sum(1 for h in running if self.cancel(h))

    def cancel_all(self) -> int:
        """The kill switch: stop every running child, whoever owns it."""
        with self._lock:
            running = [h.id for h in self._handles.values() if not h.finished]
        return sum(1 for h in running if self.cancel(h))

    def forget(self, owner: str) -> None:
        """Drop a finished owner's handles so the registry does not grow."""
        with self._lock:
            for key in [h.id for h in self._handles.values()
                        if h.owner == owner and h.finished and h.collected]:
                self._handles.pop(key, None)

    # --- the end-of-turn barrier ----------------------------------------

    def barrier(self, session) -> bool:
        """Hold an ending turn open until its children report.

        Standing at the exit — *before* the ``end_turn`` enact — is what keeps
        the agent's priority for the whole wait: there is no window in which a
        user message can land between the halves of one logical turn.

        Returns True when something was delivered, which is the loop's cue to
        re-drive so the model reads the reports before the turn really ends.
        Never raises: a barrier that breaks a turn is worse than one that
        misses a report.
        """
        try:
            return self._barrier(session)
        except Exception:
            logger.exception("the subagent barrier failed")
            return False

    def _barrier(self, session) -> bool:
        """The barrier proper. See :meth:`barrier`."""
        owner = str(getattr(session, "key", "") or "")
        pending = self.pending_for(owner)
        if not pending:
            return False

        cancel_event = getattr(session, "cancel_event", None)
        delivered = []
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                # The user stopped the turn. Take the children with it and let
                # the kernel's own cancel handling proceed.
                for handle in pending:
                    self.cancel(handle.id)
                    handle.collected = True
                return False
            now = time.time()
            for handle in list(pending):
                if handle.deadline <= now and not handle.finished:
                    self.cancel(handle.id)
                if handle.finished:
                    handle.collected = True
                    delivered.append(handle)
                    pending.remove(handle)
            if not pending:
                break
            time.sleep(BARRIER_POLL_SECONDS)

        queued = self._deliver(session, delivered)
        self.forget(owner)
        return queued

    def _deliver(self, session, handles: list[Handle]) -> bool:
        """Queue reports on the session's agent-facing message queue.

        Deliberately not ``runtime.push_message``: that is the *user*-facing
        bus push, and these reports are addressed to the model. The
        conversation guard is what keeps a notice from leaking into a
        conversation the session moved to after the spawn was made.
        """
        notices = [
            handle.notice() for handle in handles
            if handle.owner_conversation_id is None
            or getattr(session, "conversation_id", None)
            == handle.owner_conversation_id
        ]
        if not notices:
            return False
        try:
            with session.lock:
                session.pending_user_messages.extend(notices)
        except Exception:
            logger.exception("could not queue subagent reports")
            return False
        return True

    # --- scheduled spawns -----------------------------------------------

    def _on_event(self, payload) -> None:
        """A Timekeeper job fired. Start the child it describes."""
        payload = payload or {}
        try:
            handle = self.spawn(
                payload.get("prompt") or "",
                title=payload.get("title") or "Scheduled subagent",
                attachments=payload.get("attachments"),
                conversation_id=payload.get("conversation_id"),
                owner=(payload.get("report_session_key") or "").strip() or None,
                owner_conversation_id=payload.get("report_conversation_id"),
                category=self._scheduled_category(payload),
                # A scheduled child talks to the user directly; that push is
                # the only place its work would otherwise surface.
                notification_mode=None,
            )
        except Exception as exc:
            logger.error("scheduled subagent did not start: %s", exc)
            self._push(f"Scheduled agent did not start: {exc}")
            return
        # Off this thread, always. See ``_remember_conversation``.
        threading.Thread(
            target=self._remember_conversation,
            args=(payload, handle.conversation_id),
            daemon=True, name="subagent-pin-conversation",
        ).start()

    @staticmethod
    def _scheduled_category(payload) -> str:
        """Where a scheduled child's conversation files itself."""
        keeper = payload.get("_timekeeper") or {}
        return "Scheduled (one-time)" if keeper.get("one_time") else "Scheduled"

    def _remember_conversation(self, payload, cid: int) -> None:
        """Pin a recurring job's conversation so it accumulates one transcript.

        Without this every firing opens a new conversation and the job's
        history is scattered across dozens of them.

        **Runs on its own thread, and must.** This calls back into the
        timekeeper — the very service whose ``poll`` published the event that
        got us here — and that service is a resident box serializing under one
        lock which ``poll`` holds for its whole duration. Called inline it
        blocked on that lock forever, wedging the box until the hard ceiling
        killed it and taking the process's sandbox worker pool with it one
        stalled ``cron.*`` call at a time. ``sandbox/events.publish`` now keeps
        bus delivery off the publisher's thread, which fixes it at the other
        end too; this stays detached because a caller re-entering the box it
        was called from is unsafe on its own terms, whoever called it.

        Note the ``except`` below cannot cover that case: a deadlock is a
        block, not a raise. Nothing here has a return value anyone reads, so
        detaching costs the caller nothing.
        """
        job_name = (payload.get("_timekeeper") or {}).get("job_name")
        keeper = (getattr(self.runtime, "services", None) or {}).get(
            "timekeeper")
        if not job_name or keeper is None:
            return
        try:
            job = keeper.get_job(job_name)
            if job is not None:
                keeper.update_job(job_name, {"payload": {
                    **(job.get("payload") or {}), "conversation_id": cid}})
        except Exception:
            logger.exception("could not pin the conversation for job %s",
                             job_name)
