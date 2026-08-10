"""Per-conversation runtime state.

Two dataclasses live here, kept apart from the runtime so each can be read
on its own:

- :class:`RuntimeSession` is the in-memory bag of *everything* the runtime
  needs to reason about a single conversation: the state machine, the
  provider-shaped history list, persistence id, profile pinning, plugin-
  pinned tools, the per-session lock, and the cancel event.
- :class:`RuntimeResult` is the transport-neutral output the runtime hands
  back to a frontend after every action.
"""

from __future__ import annotations


import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from state_machine.conversation import ConversationState
from state_machine.errors import ActionResult
from runtime.notifications import DEFAULT_NOTIFICATION_MODE

_logger = logging.getLogger("Runtime")


@dataclass
class RuntimeResult:
    """Transport-neutral output for adapters to render."""

    ok: bool = True
    messages: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    buttons: list[dict[str, str]] = field(default_factory=list)
    form: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def add_action_result(self, result: ActionResult) -> "RuntimeResult":
        """Handle add action result."""
        self.ok = self.ok and result.ok
        self.events.extend(result.events)
        if result.message:
            self.messages.append(result.message)
        if result.error:
            self.error = result.error.to_dict()
            self.messages.append(result.error.message)
        return self


class _InterruptSlot:
    """One armable place in a session's interrupt registry.

    Occupies its slot for the life of the ``interruptible`` block whether or
    not anything was ever armed into it, so the registry stays a fixed set of
    "calls currently in flight" rather than a list that only grows when a
    caller happens to reach the arming point.
    """

    __slots__ = ("_session", "_stop")

    def __init__(self, session: "RuntimeSession"):
        self._session = session
        self._stop = None
        with session.lock:
            session._interrupts.append(self)

    def arm(self, stop) -> bool:
        """Say what would end this call. Returns whether it was accepted.

        Refused after a cancel, because the fire already happened: accepting
        here would park a stopper nobody is left to call and leave the caller
        blocking on exactly the thing it just asked to stop.
        """
        with self._session.lock:
            if self._session.cancel_event.is_set():
                return False
            self._stop = stop
            return True

    @property
    def armed(self) -> bool:
        """Whether a stopper is currently parked here."""
        return self._stop is not None

    def _disarm(self):
        """Vacate the slot.

        Removal is by identity rather than by position: calls nest (a
        compaction retry re-enters the model call), so an index would be
        invalidated by whichever slot happens to leave first.
        """
        with self._session.lock:
            self._stop = None
            try:
                self._session._interrupts.remove(self)
            except ValueError:
                pass


@dataclass
class RuntimeSession:
    """All mutable state for one frontend conversation/session."""

    key: str
    cs: ConversationState
    history: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: int | None = None
    busy: bool = False
    active_agent_profile: str = "default"
    # Subagent / specialist sessions pin a profile and can register extra tool
    # instances that are not part of the global tool_registry. When None /
    # empty, the session follows the runtime's active profile and registry.
    profile_override: str | None = None
    # The frontend transport that owns this session ("repl", "telegram", ...).
    # Set by BaseFrontend on first submit; lets the runtime apply that
    # frontend's profile (agent scope + command access). None for background
    # drivers, which follow the global active profile.
    frontend_name: str | None = None
    # The user whose data this session acts on. Ephemeral live binding set by the
    # frontend (like ``attended``) — None means the base user (DEFAULT_USER_ID).
    # Deliberately NOT persisted in to_marker(): ownership lives on the
    # conversation row, and persisting it here would let loading a conversation
    # silently rebind the session's identity. Identity flows frontend → session;
    # ownership flows conversation row → guard; the two never cross.
    user_id: int | None = None
    extra_tool_instances: list = field(default_factory=list)
    system_prompt_extras: dict[str, Any] = field(default_factory=dict)
    # Free-form per-plugin state bag, keyed by plugin name. The substrate for
    # on-demand plugins to stash session-scoped state without core-defined
    # fields. Persisted with the marker.
    plugin_state: dict[str, dict] = field(default_factory=dict)
    notification_mode: str = DEFAULT_NOTIFICATION_MODE
    # What ``recover_marker`` had to repair when this session was rebuilt, as
    # ``{title, body}``. Delivered as notifications at ``open_session``; kept
    # here as the session's own record of what it woke up to, which is the one
    # question the notification cannot answer later ("was *this* session the
    # one that crashed?"). Ephemeral — recovery already happened.
    restore_notices: list[dict] = field(default_factory=list)
    # Whether a human is present at this session right now (can answer an
    # interactive prompt and see output). None = defer to the kernel's global
    # single-active rule (REPL/Telegram, background drivers). True/False = the
    # owning frontend manages attendance explicitly (concurrent multi-user
    # frontends, e.g. a website setting it on socket connect/disconnect).
    # Ephemeral live state — deliberately NOT persisted in to_marker(), so it
    # resets to None (defer-to-global) across restarts.
    attended: bool | None = None
    # How this conversation answers approval dialogs: one of
    # ``runtime.security_modes.SECURITY_MODES``, or None for the default
    # ("ask"). Set by ``/mode`` through ``runtime.set_security_mode``.
    #
    # ``security_mode_conversation`` is what makes the mode **per
    # conversation** rather than per session, and it does it structurally: the
    # reader answers the default whenever this does not match the session's
    # live ``conversation_id``. That is deliberate over resetting the mode at
    # each of the places a conversation changes — ``load_conversation``,
    # ``/new``, ``/clear``, and the three paths that null the id out — because
    # a reset is a list to keep in step and this is a fact that cannot drift.
    # A mode cannot leak into the next conversation because there is nowhere
    # for it to leak from.
    #
    # Both are ephemeral — deliberately NOT persisted in to_marker(), so a
    # restart returns to "ask". A forgotten "yolo" that survives a restart is
    # the one failure this must not have.
    security_mode: str | None = None
    security_mode_conversation: int | None = None
    # A mode for the rest of the current agent turn, outranking the one above
    # and dropped by ``HookRegistry.finish_turn``. This is the grant scoped to
    # *time* that the three standing grant lists could never express — what
    # "allow, and stop asking for the rest of this turn" writes, and what an
    # approved plan hands the turn that follows it. Ephemeral for the same
    # reason and more so.
    turn_security_mode: str | None = None
    # User messages sent while the agent was mid-turn. The busy guard queues
    # them here instead of rejecting; ConversationLoop drains the list at each
    # loop boundary, and handle_action starts a fresh turn with any leftovers
    # once the current turn ends. Guarded by ``lock``. Deliberately NOT
    # persisted in to_marker(): crash recovery already tells the user their
    # in-flight message was not replayed, and a queue that silently survives
    # a restart would contradict that notice.
    pending_user_messages: list[str] = field(default_factory=list)
    # Agent actions queued by hooks/tools for the ConversationLoop to drain
    # at its next loop boundary (never mid tool-call batch): dicts of
    # ``{"name": tool_name, "args": {...}, "forced_by": <hook label>}``. The
    # agent-side mirror of ``pending_user_messages`` — how a turn_start hook
    # injects a tool call at the start of a turn, or a tool queues follow-up
    # work within the same turn. Guarded by ``lock``. Deliberately NOT
    # persisted in to_marker(): a queued action must not replay after a
    # restart the queuing hook never saw.
    pending_agent_actions: list = field(default_factory=list)
    # Attachments staged by hooks/tools for the session's next model call
    # (see ``HookRegistry.stage_attachment`` / ``drain_attachments``). Same
    # lifecycle rules as ``pending_agent_actions``: ephemeral, cleared at
    # turn end, deliberately NOT persisted in to_marker().
    staged_attachments: list = field(default_factory=list)
    # A tool (or hook) set this to end the current drive loop and have the
    # runtime immediately re-drive the turn — build_loop re-resolves the LLM,
    # registry, and prompt, so a plugin can swap any of them mid-turn (e.g.
    # escalation to a stronger model). The truncated drive keeps agent
    # priority and emits no end_turn: the re-driven loop finishes the logical
    # turn. This flag is the session-level spelling of the end_turn doorway's
    # ``Redrive`` verdict (runtime/hooks.py) — new code should return the
    # verdict from an end_turn hook; setting the flag directly remains
    # supported for tools that decide mid-turn. Ephemeral — deliberately NOT
    # persisted in to_marker(); a restart request must not survive a process
    # restart. Note a per-call ``llm_call`` escort can swap the brain
    # without any restart at all — prefer that when a re-drive was only ever
    # a vehicle for swapping the LLM.
    restart_turn: bool = False
    has_compaction_checkpoint: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # Stoppers for work currently in flight, armed by whoever is about to
    # block. ``cancel_event`` alone is a flag nobody watches: it is read
    # between actions, and everything slow lives *inside* one — so cancelling
    # was only ever as immediate as the current model or tool call. See
    # ``interruptible``. Holds :class:`_InterruptSlot` objects, one per call in
    # flight. Ephemeral by construction, and deliberately NOT in to_marker():
    # a stopper is a live callable.
    _interrupts: list = field(default_factory=list, repr=False)

    @contextlib.contextmanager
    def interruptible(self):
        """Arm a stopper for the duration of one blocking call.

        Yields a slot; whoever is about to block calls ``slot.arm(stop)`` with
        something that ends the wait, and ``interrupt`` fires it. A *slot*
        rather than a plain argument because the caller often does not know
        what to stop until it is already inside the call — the model path only
        learns which box is serving it once ``Brain.chat`` has leased one.

        **Arming is refused once ``cancel_event`` is set**, and that is the
        race this whole mechanism exists to close: a cancel landing between
        the loop's last check and the next call would otherwise register a
        stopper nobody is left to fire, and the turn would block anyway.
        ``interrupt`` cannot be given a second chance — it has already run.
        """
        slot = _InterruptSlot(self)
        try:
            yield slot
        finally:
            slot._disarm()

    def interrupt(self) -> int:
        """Fire every armed stopper. Returns how many. Never raises.

        Called from the *canceller's* thread while the work it is stopping
        blocks on another, so a stopper that raises must not take the cancel
        down with it — a half-cancelled turn is the worst outcome available.
        """
        with self.lock:
            stoppers = [slot._stop for slot in self._interrupts if slot.armed]
        for stop in stoppers:
            try:
                stop()
            except Exception:
                _logger.exception("interrupt stopper failed for %s", self.key)
        return len(stoppers)

    @contextlib.contextmanager
    def unlocked(self):
        """Fully release this thread's hold on ``lock`` for the duration.

        Installed on the session's :class:`ConversationState` so a command or
        tool *body* runs outside the lock that ``handle_action`` holds around
        dispatch, exactly as an agent turn already does. A body can block
        waiting for the user — an approval dialog, an interactive tool — and
        the answer arrives as a second action on another thread, which needs
        this same lock. Holding it through the call deadlocks that round trip;
        ``/packages install`` froze the whole app on precisely that cycle.

        ``lock`` is an RLock, so a nested ``handle_action`` on this thread may
        hold it several times over. Releasing once would leave it held and fix
        nothing, so the count is unwound and restored exactly.

        Concurrency is still guarded while parked: ``_run`` puts the state
        machine into a phase from ``BUSY_PHASES`` for the length of the call,
        and the busy guard in ``handle_action`` reads that phase — so only the
        ``answer_approval``/``cancel`` pair the dialog needs can get in.
        """
        depth = 0
        while True:
            try:
                self.lock.release()
            except RuntimeError:
                break
            depth += 1
        try:
            yield
        finally:
            for _ in range(depth):
                self.lock.acquire()

    def to_marker(self) -> dict[str, Any]:
        """Handle to marker."""
        state = self.cs.to_dict()
        state.update({
            "conversation_id": self.conversation_id,
            "active_agent_profile": self.active_agent_profile,
            "profile_override": self.profile_override,
            "frontend_name": self.frontend_name,
            "notification_mode": self.notification_mode,
            "system_prompt_extras": self.system_prompt_extras,
            "plugin_state": self.plugin_state,
            "busy": self.busy,
        })
        return state


class SessionConflict(RuntimeError):
    """Raised when a session_key is requested for a conversation_id that
    conflicts with an existing live session.

    Without this guard a second binding could silently stomp on the first;
    with it, the second bind fails loudly.
    """

    def __init__(self, session_key: str, existing_id: int | None, requested_id: int | None):
        """Initialize the session conflict."""
        super().__init__(
            f"Session '{session_key}' is already bound to conversation {existing_id}; "
            f"cannot rebind to {requested_id}."
        )
        self.session_key = session_key
        self.existing_id = existing_id
        self.requested_id = requested_id
