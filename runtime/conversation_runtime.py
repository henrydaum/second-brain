"""Adapter-facing conversation runtime.

This is the single dispatcher between a frontend transport (REPL, Telegram,
future event bus) and the state machine. It owns sessions, persistence,
and approvals, but every state-changing decision goes through one labeled
``cs.enact(...)`` site (see :meth:`ConversationRuntime._dispatch`). When
the user's action hands turn priority to the agent, the runtime hands off
to ``ConversationLoop.drive()``, which contains its own labeled
``cs.enact(...)`` site for the agent's moves.

That two-call-site shape mirrors PokerMonster's ``run_game``: one obvious
line where everything flows through, easy to find, easy to read.

How this file is organised
--------------------------

The runtime concerns are split across a small family of modules so each
one is its own readable unit:

- :mod:`state_machine.session` — the ``RuntimeSession`` + ``RuntimeResult``
  dataclasses and the ``SessionConflict`` error.
- :mod:`state_machine.runtime_persistence` — load/save/restore, the
  ``open_session`` unified entry point, and the ``iterate_agent_turn``
  external-driver path.
- :mod:`state_machine.runtime_approvals` — programmatic typed-input /
  approval requests.
- :mod:`state_machine.runtime_config` — per-session profile, scope, tool
  registry, system prompt, and ``ConversationLoop`` construction.
- :mod:`state_machine.runtime_dispatch` — small per-action helpers used
  inside ``_dispatch``.

This file keeps the **runtime story**: the constructor that wires
everything up, the ``handle_action`` entry point, the user-side dispatch
loop, the agent-turn driver, and the plugin-facing API surface. Read
top-to-bottom.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from events.event_bus import bus
from events.event_channels import (
    CHAT_MESSAGE_PUSHED,
    CONVERSATION_CHANGED,
    SESSION_AGENT_PROFILE_CHANGED,
    SESSION_CLOSED,
    SESSION_SECURITY_MODE_CHANGED,
    SYSTEM_PROMPT_EXTRA_CHANGED,
)

from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversation import CallableSpec
from state_machine.conversation_phases import BASE_PHASE, BUSY_PHASES, FORM_PHASES, PHASE_APPROVING_REQUEST
from state_machine.errors import ActionError
from runtime.security_modes import (
    CONVERSATION_SCOPE,
    DEFAULT_SECURITY_MODE,
    TURN_SCOPE,
    scope_name,
)
# Aliased: the reader below is *also* called ``security_mode``, and while a
# method never shadows a global inside its own body, two identical names one
# indent apart is a line nobody should have to think about twice.
from runtime.security_modes import security_mode as _normalize_mode
from runtime.session import RuntimeResult, RuntimeSession, SessionConflict

from runtime import runtime_approvals as _approvals
from runtime import runtime_config as _cfg
from runtime import dispatch as _disp
from runtime import ledger as _ledger
from runtime import notifications as _notifications
from runtime import persistence as _persist
from pipeline.database import DEFAULT_USER_ID

logger = logging.getLogger("Runtime")


class ConversationRuntime:
    """Owns sessions, persistence, commands/forms, approvals, and agent turns."""

    def __init__(
        self,
        db=None,
        services: dict | None = None,
        config: dict | None = None,
        tool_registry=None,
        system_prompt: str | Callable[[], str] = "",
        commands: dict[str, CallableSpec] | None = None,
        command_specs: dict[str, dict] | None = None,
        emit_event: Callable[[str, Any], None] | None = None,
        on_tool_start=None,
        on_tool_result=None,
        on_notice=None,
    ):
        """Initialize the conversation runtime."""
        self.db = db
        self.services = services or {}
        self.config = config or {}
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.commands = {**(commands or {}), **_cfg.command_specs_from_dicts(command_specs or {})}
        self.emit_event = emit_event
        self.on_tool_start = on_tool_start
        self.on_tool_result = on_tool_result
        self.on_notice = on_notice
        self.sessions: dict[str, RuntimeSession] = {}
        # Opt-in per-session extension points (permission gates, scope shapers).
        # Empty by default; plugins register into it. See runtime/hooks.py.
        from runtime.hooks import HookRegistry
        self.hooks = HookRegistry()
        # Background agents this runtime has started. Built here rather than
        # in the composition root because the end-of-turn barrier must exist
        # whether or not any plugin is installed — the same argument the
        # compaction layer makes one moment over. See runtime/subagents.py.
        from runtime.subagents import SubagentRegistry
        self.subagents = SubagentRegistry(self, self.config)
        self._approval_requests: dict[str, StateMachineApprovalRequest] = {}
        self._sessions_lock = threading.RLock()
        # Single global "active" session — the most recent user-driven
        # session_key. Automation paths explicitly opt out via
        # ``handle_action(..., user_driven=False)``.
        self.active_session_key: str | None = None
        # The first user-driven session after startup gets the previously
        # active conversation auto-restored, so the user lands back where
        # they left off. The id is persisted to config on every conversation
        # switch (see ``_persist_active_conversation``).
        self._restore_consumed_keys: set[str] = set()
        self._persisted_active_conv_by_user: dict[int, int | None] = {}

    # ──────────────────────────────────────────────────────────────────
    # Public entrypoint — every action a frontend can take ends up here.
    # ──────────────────────────────────────────────────────────────────

    def handle_action(self, session_key: str, action_type: str, payload: dict | str | None = None, *, user_driven: bool = True) -> RuntimeResult:
        """Route one frontend action through guards, dispatch, and optional agent follow-up."""
        session = self.get_session(session_key)
        self._reconcile_session_binding(session)
        if user_driven:
            self.active_session_key = session_key
            prior_conv = self._persisted_active_conv_by_user.get(self.session_user_id(session_key))

        # Cron-handoff guard: a non-user-driven send_text must never be
        # interpreted as form input. If the user is mid-form, refuse the turn.
        if (not user_driven
                and action_type == "send_text"
                and session.cs.phase in FORM_PHASES):
            return RuntimeResult(False, error={
                "code": "busy",
                "message": "Session is mid-form — handoff deferred."})

        # A notification, for the reason given at the busy branch below: the
        # usual way to cancel is a button, and pressing one invokes no callable.
        # Answering with nothing at all would be worse than either channel —
        # somebody pressed a control and is owed a reply, even when the reply is
        # that there was nothing to do.
        if action_type == "cancel" and not session.busy and session.cs.phase == BASE_PHASE:
            self.notify(title="Nothing to cancel",
                        body="No turn was running.",
                        source="runtime", session_key=session_key,
                        persist=False)
            return RuntimeResult(data={"cancelled": False})

        # Busy guard: if the session is mid-turn, only ``cancel`` and the
        # specific ``answer_approval`` for an active approval frame may
        # proceed. Everything else is told to wait or cancel first.
        if session.busy or session.cs.phase in BUSY_PHASES:
            if action_type == "cancel" and session.cs.phase != PHASE_APPROVING_REQUEST:
                # Cancel means "stop everything" — drop queued messages too,
                # and take the background agents this turn started with it
                # (and anything *they* started). Stopping the agent while its
                # children carry on is the worst of both: the work continues,
                # costs money, and reaches nobody.
                stopped = self.subagents.cancel_for(session_key)
                with session.lock:
                    # Nothing is queued in its place. ``pending_user_messages``
                    # is a *drive trigger*, not a mailbox: the closing-race
                    # drain below pops it and dispatches it as a fresh
                    # ``send_text``. Putting the "you were cancelled" notice
                    # here therefore started a whole new agent turn — which,
                    # since the finally has cleared the flag by then, was not
                    # cancelled and ran to completion. Telling the model is
                    # the loop's job, on its way out; see
                    # ``ConversationLoop._record_cancellation``.
                    session.pending_user_messages.clear()
                session.cancel_event.set()
                # The flag first, then the stoppers: everything that wakes up
                # must find the turn already cancelled, or it carries on doing
                # the work it was just interrupted out of.
                #
                # Setting the flag alone is what ``/cancel`` used to be, and
                # it is only read *between* actions — while everything slow
                # lives inside one. So cancelling was as immediate as the
                # current model or tool call, which for a streaming model that
                # had started repeating itself meant not immediate at all.
                self._interrupt_work(session)
                # Whether, not how many. ``cancel_for`` walks *down* the
                # lineage and stops descendants too, but counts only the
                # generation it iterated — ``pending_for`` matches on owner,
                # and a grandchild's owner is its parent's session key rather
                # than this one. So the number was only ever right while
                # ``max_subagent_depth`` is 1 (the default, where a child
                # cannot spawn at all) and would quietly under-report the
                # moment anyone raised it. A count nobody can trust at every
                # setting is worse than no count: this says the true thing at
                # all of them.
                # A notification, not ``callable_output``, and the distinction
                # is which *gesture* this was. Typing ``/cancel`` invokes a
                # callable by name; pressing a Cancel button invokes nothing —
                # it is a receipt about the turn, and a client with a command
                # panel had no command to put it against. The one that shipped
                # synthesized a phantom command run to hold it, which opened
                # the settings screen every time somebody stopped the agent.
                #
                # The twin case is two branches below and already knew this: a
                # message sent mid-turn raises a ``persist=False`` notification
                # and answers with ``data``. Same situation exactly — mid-turn,
                # user-initiated, worth seeing for a moment and worth nothing
                # afterwards — so it gets the same shape. The ``Cancel`` action
                # keeps ``callable_output``, because a settings form dismissing
                # itself really is that form reporting on its own navigation.
                #
                # Frontends with no notification surface flatten it back into
                # the chat, so nothing is lost on a terminal — though not
                # byte-identically: "Cancelled." becomes the title over the
                # body, which is how every other notification already reads
                # there ("New conversation started" over "Agent: default.").
                self.notify(
                    title="Cancelled",
                    body=("The turn was stopped, along with the background "
                          "agents it had started." if stopped
                          else "The turn was stopped."),
                    source="runtime", session_key=session_key, persist=False)
                return RuntimeResult(data={"cancelled": True,
                                           "subagents_stopped": stopped})
            if action_type in {"answer_approval", "cancel"} and session.cs.phase == PHASE_APPROVING_REQUEST:
                pass  # fall through and dispatch
            elif action_type == "send_text":
                # Queue mid-turn text instead of rejecting it. The running
                # ConversationLoop drains the queue at its next boundary; if
                # the turn ends first, handle_action starts a fresh turn with
                # the leftovers (see the re-drive loop below).
                text = _disp.text_of(payload)
                if not text:
                    return RuntimeResult(False, error={"code": "empty_input", "message": "No input."})
                with session.lock:
                    session.pending_user_messages.append(text)
                # A notification rather than a reply, because nothing was
                # answered: the message was accepted and nobody has read it
                # yet. Saying "Got it — I'll read that as soon as I finish
                # this step" put the agent's voice on a receipt the agent had
                # no part in, in the first person, in the middle of a turn the
                # user is watching that agent take.
                #
                # ``persist=False`` — a receipt is worth seeing for a moment
                # and worth nothing afterwards. What it acknowledges arrives
                # in the transcript under its own steam a few seconds later.
                self.notify(
                    title="Message queued",
                    body="It will be read when the current step finishes.",
                    source="runtime", session_key=session_key, persist=False)
                return RuntimeResult(data={"queued": True})
            else:
                return RuntimeResult(False, error={"code": "busy", "message": "Still working. Send /cancel to interrupt."})

        # No-conversation guard: chat actions need a conversation to write
        # into. Fresh installs (and sessions whose saved conversation was
        # deleted out from under them) land here with conversation_id=None
        # and would otherwise silently auto-create a "main" conversation —
        # the new model funnels conversation creation through
        # /conversations, so route the user there instead. Skipped when
        # there is no DB (unit tests without persistence rely on the
        # implicit auto-create path).
        if (user_driven and self.db is not None
                and action_type in {"send_text", "send_attachment"}
                and session.conversation_id is None):
            if not (self.config.get("llm_profiles") or {}):
                return RuntimeResult(False, error={
                    "code": "no_llm",
                    "message": "Welcome to Second Brain. Run /setup to configure an LLM and the Telegram frontend."})
            return RuntimeResult(False, error={
                "code": "no_conversation",
                "message": "No conversation loaded.\nTry /new."})

        with session.lock:
            _cfg.refresh_specs(self, session)
            try:
                out = self._dispatch(session, action_type, payload)
            finally:
                if action_type not in {"load_history", "new_conversation"}:
                    _persist.persist_marker(self, session)

        # The agent turn runs *outside* the session lock on purpose. A tool
        # inside the turn may call ``runtime.request_input(...)`` and block
        # synchronously waiting for the user — the user's answer arrives via
        # another ``handle_action`` call which needs to acquire the same
        # lock. Holding the lock through the whole turn would deadlock that
        # round-trip. Per-mutation atomicity is preserved by the dispatch
        # lock above and the lock acquired in ``inject_user_message`` and
        # in ``iterate_agent_turn`` after the handle_action returns.
        #
        # A *command* body is the same hazard and was not covered by this for
        # a long time: it runs inside ``_dispatch``, and therefore inside the
        # lock taken just above. ``/packages install`` asked for approval from
        # there and froze the process. It is handled one level down instead —
        # ``_CallableAction._run`` wraps only the plugin body in
        # ``cs.unlocked()`` (``RuntimeSession.unlocked``), so the dispatch
        # mutations around it keep this lock and the body does not.
        drives = 0
        restart_drive = False
        while out.data.pop("_drive_agent_turn", False) and drives < 5:
            drives += 1
            # Turn starters run once per logical turn: a restart re-drive is
            # the same turn (same user message, session state carried over),
            # so it skips them; the closing-race follow-up below is a fresh
            # turn and runs them again.
            if not restart_drive:
                hooks = getattr(self, "hooks", None)
                if hooks is not None:
                    hooks.start_turn(session, runtime=self)
            restart_drive = False
            self._drive_agent_turn(session, out, allow_restart=drives < 5)
            with session.lock:
                _persist.persist_marker(self, session)
            # A tool requested a turn restart (session.restart_turn): re-drive
            # immediately. build_loop re-resolves the LLM/registry/prompt, so
            # hooks can swap them for the re-driven turn. The drives bound
            # keeps a restart-happy plugin finite.
            if session.restart_turn:
                session.restart_turn = False
                restart_drive = True
                # A restart set after end_turn (e.g. a turn_finish barrier
                # holding for subagent reports) finds priority already handed
                # back to the user; the re-driven half needs agent priority or
                # the loop exits immediately without acting.
                if session.cs.turn_priority != "agent":
                    session.cs.set_priority("agent")
                out.data["_drive_agent_turn"] = True
                continue
            # Closing-race check: a message queued after the loop's final
            # drain (but before busy went False) would otherwise sit unread
            # until the next user input. Start a fresh turn with the first
            # leftover; the new turn's own drain absorbs any others. The
            # drives bound keeps a pathological ping-pong finite.
            with session.lock:
                if not session.pending_user_messages:
                    break
                if session.cs.phase != BASE_PHASE or session.cs.turn_priority != "user":
                    # Turn ended into a form/approval — a user send_text is
                    # not legal here. Leave the queue; the next agent turn's
                    # drain absorbs it.
                    break
                text = session.pending_user_messages.pop(0)
                _cfg.refresh_specs(self, session)
                follow = self._dispatch(session, "send_text", text)
                _persist.persist_marker(self, session)
            out.ok = out.ok and follow.ok
            out.messages.extend(follow.messages)
            out.callable_output.extend(follow.callable_output)
            out.attachments.extend(follow.attachments)
            out.events.extend(follow.events)
            if follow.error:
                out.error = follow.error
            out.data.update(follow.data)

        if user_driven:
            current_conv = self.active_conversation_id
            if current_conv != prior_conv:
                self._persist_active_conversation(current_conv)

        return out

    # ──────────────────────────────────────────────────────────────────
    # The single user-side dispatch. One labeled `cs.enact()` line, plus a
    # hand-off to ConversationLoop when the action transferred priority to
    # the agent.
    # ──────────────────────────────────────────────────────────────────

    def _dispatch(self, session: RuntimeSession, action_type: str, payload: dict | str | None) -> RuntimeResult:
        """Apply one user-side action and decide whether an agent turn should follow."""
        # Transport-level actions that never enter the state machine.
        if action_type == "load_history":
            return self.load_history(session.key, int((payload or {}).get("conversation_id")))
        if action_type == "new_conversation":
            return self.new_conversation(session.key)

        text = _disp.text_of(payload)
        inbound_attachments = _disp.attachments_of(payload)
        actor_id = _disp.actor_id_of(payload)

        # Callers that bypass SendAttachment (e.g. iterate_agent_turn) can
        # pass attachments straight on the payload — push them onto
        # cs.pending_attachments so the next agent turn picks them up. Their
        # records go on the message row like anybody else's: the row is what
        # says the message carried a file, and a background driver's message
        # is not less of a message.
        inbound_records = []
        if inbound_attachments:
            from attachments.attachment import Attachment
            for entry in inbound_attachments:
                if isinstance(entry, dict):
                    entry = Attachment.from_dict(entry)
                if isinstance(entry, Attachment):
                    session.cs.pending_attachments.append(entry)
                    inbound_records.append(entry.record())

        # Empty-input guard, matching v1 behavior. Skips state-machine entry
        # so we don't pollute history/events with a doomed action.
        if action_type == "send_text" and not text:
            return RuntimeResult(False, error={"code": "empty_input", "message": "No input."})

        # Predicate captured *before* enact, since the action itself may
        # transition phase/priority. This is what tells us whether an agent
        # reply turn should follow.
        expects_agent_reply = (
            action_type in {"send_text", "send_attachment"}
            and session.cs.phase == BASE_PHASE
            and session.cs.turn_priority == "user"
        )

        content = _disp.content_for_action(action_type, text, payload)

        out = RuntimeResult()
        old_phase = session.cs.phase
        old_priority = session.cs.turn_priority

        # ──────────────── THE enact() SITE (user-side) ────────────────
        request_id = _approvals.current_request_id(session, action_type)
        enact_started = time.perf_counter()
        result = session.cs.enact(action_type, content, actor_id)
        # ──────────────────────────────────────────────────────────────
        _ledger.record_enact(
            self.db, origin="user_enact", session_key=session.key,
            conversation_id=session.conversation_id,
            user_id=self.session_user_id(session.key), actor_id=actor_id,
            action_type=action_type, content=content, result=result,
            duration_ms=int((time.perf_counter() - enact_started) * 1000),
        )

        out.add_action_result(result)
        _approvals.resolve_answered_request(self, session.key, request_id, result)
        text = _disp.text_after_action(action_type, text, result)
        _disp.absorb_user_action(self, session, action_type, text, result,
                                 records=inbound_records)
        _disp.emit_state_change(session, old_phase, old_priority)
        _disp.decorate_form(session, out)
        _disp.echo_callable_result(action_type, result, out)

        if (expects_agent_reply
                and result.ok
                and session.cs.turn_priority == "agent"
                and session.cs.phase == BASE_PHASE):
            out.data["_drive_agent_turn"] = True

        return out

    # ──────────────────────────────────────────────────────────────────
    # Driving the agent's turn. ConversationLoop has its OWN labeled
    # cs.enact() site inside it; this method just sets up persistence and
    # surfaces the loop's outputs.
    #
    # Persistence ordering matters here: we set ``busy=True`` and snapshot
    # BEFORE calling drive(), so a crash mid-turn leaves a marker that
    # tells the next runtime "this session was mid-turn — recover."
    # ──────────────────────────────────────────────────────────────────

    def _drive_agent_turn(self, session: RuntimeSession, out: RuntimeResult, *, allow_restart: bool = True) -> RuntimeResult:
        """Run the agent loop for a session and surface its outputs back to the frontend.

        ``allow_restart=False`` voids any ``session.restart_turn`` the drive
        set (the caller's drive budget is exhausted), so the turn ends
        normally: priority returns to the user and SESSION_TURN_COMPLETED is
        emitted instead of being suppressed for a re-drive that never comes."""
        _persist.ensure_conversation(session=session, runtime=self, title_text=_disp.latest_user_text(session))
        session.busy = True
        session.cancel_event.clear()
        _persist.persist_marker(self, session)  # busy=True snapshot for crash recovery
        from events.event_channels import SESSION_TURN_STARTED
        old_phase, old_priority = session.cs.phase, session.cs.turn_priority
        (self.emit_event or bus.emit)(SESSION_TURN_STARTED, {
            "session_key": session.key,
            "conversation_id": session.conversation_id,
            "actor_id": "agent",
        })
        crash_error: str | None = None
        reply, new_messages, attachments = None, [], []
        try:
            loop = _cfg.build_loop(self, session.key)
            reply, new_messages, attachments = loop.drive(
                session.cs,
                "agent",
                session.history,
                self.db,
                session.conversation_id,
            )
            # The subagent backstop. The loop consults the barrier at its
            # end_turn doorways, which is the right place: it holds agent
            # priority for the whole wait and there is no window for a user
            # message between the halves of one logical turn. But a drive can
            # leave by other routes — a failed action, a priority handoff, an
            # iteration budget — and on those the children were abandoned with
            # their reports produced and never delivered, which is silent.
            # This is the one line every drive passes through, so the question
            # gets asked here too. ``barrier`` settles what it collects, so
            # the second ask on a normal turn finds nothing and costs nothing.
            if self.subagents.barrier(session):
                session.restart_turn = True
            if session.restart_turn and not allow_restart:
                logger.warning("Turn restart requested with the drive budget exhausted; ending the turn instead.")
                session.restart_turn = False
        except Exception as e:
            # A cancel is not a crash. Interrupting the turn *is* how it stops
            # — the model box is killed out from under the call, so the
            # exception says "box 'llm_0_0' died during '__chat__'", which is
            # the mechanism working exactly as asked. Reporting it would put
            # an ``Error:`` on screen immediately after ``Cancelled.``, which
            # is both alarming and the specific thing this change promised not
            # to do. Read here rather than in the finally below, because that
            # is where the flag gets cleared.
            interrupted = session.cancel_event.is_set()
            if not interrupted:
                err = ActionError("agent_failed", str(e))
                session.cs.last_error = err
                out.ok = False
                out.error = err.to_dict()
                # Deliberately *not* also appended to ``out.messages``: a
                # frontend renders both channels, so the same sentence arrived
                # twice — once bare and once as "Error: ...".
                # ``render_error`` is required of every frontend, so the error
                # channel alone is enough.
                crash_error = str(e)
            else:
                logger.info("agent turn for %s ended by cancellation: %s",
                            session.key, e)
            # A restart requested by a turn that then crashed is void — the
            # finally below must reclaim priority for the user as usual.
            session.restart_turn = False
        finally:
            # Read before the clear below — this is the only record of whether
            # the turn was actually interrupted (used for the no-reply label).
            was_cancelled = session.cancel_event.is_set()
            session.busy = False
            # Safety net: EndTurn should already have handed priority back, but
            # if drive() raised partway through, force the user back into
            # priority so the conversation can continue. A pending turn restart
            # keeps agent priority on purpose: the re-driven loop ends the turn.
            if session.cs.turn_priority != "user" and not session.restart_turn:
                session.cs.set_priority("user")
            session.cancel_event.clear()
            hooks = getattr(self, "hooks", None)
            if hooks is not None and not session.restart_turn:
                # turn_finish observers fire once per LOGICAL turn: a pending
                # re-drive means this drive was only the first half, so they
                # wait for the drive that actually ends the turn. (Crash and
                # exhausted-budget paths void restart_turn above, so those
                # turns always reach their observers.)
                from runtime.hooks import TurnOutcome
                hooks.finish_turn(session, TurnOutcome(
                    ok=crash_error is None,
                    cancelled=was_cancelled,
                    final_text=reply or "",
                ), runtime=self)

        from events.event_channels import SESSION_TURN_COMPLETED
        if crash_error is not None:
            # Crash path: broadcast the state change (the finally above just
            # reclaimed priority for the user) before completing the turn.
            _disp.emit_state_change(session, old_phase, old_priority)
            (self.emit_event or bus.emit)(SESSION_TURN_COMPLETED, {
                "session_key": session.key,
                "conversation_id": session.conversation_id,
                "user_id": self.session_user_id(session.key),
                "ok": False,
                "error": crash_error,
                "final_text": "",
                "new_messages": [],
                "attachments": [],
            })
            return out

        if was_cancelled:
            # Nothing. The ``/cancel`` action already answered "Cancelled." on
            # its own thread, and this result would render a second copy — or,
            # worse, fall through to the ``new_messages`` branch below and
            # surface the last assistant content, which is agent output
            # arriving after the person stopped the agent. Silence here is
            # what makes the cancel the last word.
            pass
        elif reply:
            # The reply's SESSION_MESSAGE was already emitted when the loop
            # recorded the send_text row (see ConversationLoop._record).
            out.messages.append(reply)
        elif session.restart_turn:
            # Half of a logical turn with a re-drive pending: the drive that
            # actually ends the turn supplies the reply (or its own fallback).
            pass
        elif new_messages:
            # Agent ended without final text but produced messages — surface
            # the last assistant content if any, otherwise say what happened.
            last_assistant = next((m for m in reversed(new_messages) if m.get("role") == "assistant"), None)
            fallback = "(The agent ended its turn without a reply.)"
            out.messages.append(last_assistant.get("content") if last_assistant and last_assistant.get("content") else fallback)
        else:
            out.messages.append("(The agent ended its turn without a reply.)")

        out.attachments.extend(attachments)
        out.data.setdefault("conversation_id", session.conversation_id)
        out.data.setdefault("new_messages", []).extend(new_messages)
        # Agent-side state changes (end_turn's priority hand-back, phase
        # reset) get the same broadcast that user actions get in _dispatch.
        _disp.emit_state_change(session, old_phase, old_priority)
        if not session.restart_turn:
            (self.emit_event or bus.emit)(SESSION_TURN_COMPLETED, {
                "session_key": session.key,
                "conversation_id": session.conversation_id,
                "user_id": self.session_user_id(session.key),
                "ok": True,
                "cancelled": was_cancelled,
                "final_text": reply or "",
                "new_messages": list(new_messages),
                "attachments": list(attachments),
            })
        return out

    # ──────────────────────────────────────────────────────────────────
    # Session lifecycle.
    # ──────────────────────────────────────────────────────────────────

    def get_session(self, key: str) -> RuntimeSession:
        """Return the live runtime session for a session key, creating it if needed."""
        return _persist.get_or_create_session(self, key)

    def open_session(
        self,
        session_key: str,
        *,
        conversation_id: int | None = None,
        kind: str = "user",
        category: str | None = None,
        title: str = "New Conversation",
        agent_profile: str | None = None,
        system_prompt_extras: dict[str, Any] | None = None,
        override: bool = False,
    ) -> RuntimeSession:
        """Create-or-load a session bound to a specific conversation.

        See :func:`state_machine.runtime_persistence.open_session`. When binding to
        an *existing* conversation, refuses (raises ``PermissionError``) if the
        session's effective user does not own it, unless ``override`` is set.
        """
        if conversation_id is not None and not self.assert_conversation_access(session_key, conversation_id, override=override):
            raise PermissionError(f"Conversation {conversation_id} is not accessible to this session.")
        return _persist.open_session(
            self, session_key,
            conversation_id=conversation_id, kind=kind, category=category,
            title=title, agent_profile=agent_profile,
            system_prompt_extras=system_prompt_extras,
        )

    def create_conversation(self, title: str = "New Conversation", *, kind: str = "user", category: str | None = None, user_id: int = DEFAULT_USER_ID) -> int | None:
        """Create a persisted conversation row (owned by ``user_id``) and return its ID."""
        cid = _persist.create_conversation(self, title, kind=kind, category=category, user_id=user_id)
        if cid is not None:
            _ledger.record_system(self.db, action_type="conversation_create", ok=True,
                                  conversation_id=cid, user_id=user_id,
                                  args={"title": title, "kind": kind, "category": category})
            bus.emit(CONVERSATION_CHANGED, {"action": "created", "conversation_id": cid, "user_id": user_id, "category": category})
        return cid

    def load_conversation(self, session_key: str, conversation_id: int, *, agent_profile: str | None = None, system_prompt_extras: dict[str, Any] | None = None, override: bool = False) -> RuntimeSession:
        """Load a persisted conversation into a runtime session.

        Refuses (raises ``PermissionError``) if the session's effective user does
        not own the conversation, unless ``override`` is set."""
        if not self.assert_conversation_access(session_key, conversation_id, override=override):
            raise PermissionError(f"Conversation {conversation_id} is not accessible to this session.")
        return _persist.load_conversation(self, session_key, conversation_id, agent_profile=agent_profile, system_prompt_extras=system_prompt_extras)

    def load_history(self, session_key: str, conversation_id: int, *, override: bool = False) -> RuntimeResult:
        """Load saved transcript history for one conversation.

        Refuses cross-user access with a non-leaking message (the conversation is
        reported as if it does not exist)."""
        if not self.assert_conversation_access(session_key, conversation_id, override=override):
            return RuntimeResult(False, error={
                "code": "not_found", "message": "No such conversation."})
        return _persist.load_history(self, session_key, conversation_id)

    def reset_conversation(self, session_key: str) -> RuntimeSession:
        """Drop the in-memory conversation state for one session."""
        return _persist.reset_conversation(self, session_key)

    def new_conversation(self, session_key: str) -> RuntimeResult:
        """Create and switch to a fresh user conversation for the session."""
        return _persist.new_conversation(self, session_key)

    def iterate_agent_turn(self, session_key: str, prompt: str, *, attachments=None, actor_id: str = "user") -> RuntimeResult:
        """Inject input and immediately drive the agent turn outside a frontend transport."""
        return _persist.iterate_agent_turn(self, session_key, prompt, attachments=attachments, actor_id=actor_id)

    def inject_user_message(self, session_key: str, text: str, *, conversation_id: int | None = None, actor_id: str = "user", override: bool = False) -> RuntimeResult:
        """Append a message directly to a session before handing control to the agent loop."""
        if conversation_id is not None and not self.assert_conversation_access(session_key, conversation_id, override=override):
            return RuntimeResult(False, error={
                "code": "not_found", "message": "No such conversation."})
        return _persist.inject_user_message(self, session_key, text, conversation_id=conversation_id, actor_id=actor_id)

    def close_session(self, session_key: str) -> bool:
        """Close one live session and persist its final marker state."""
        return _persist.close_session(self, session_key)

    def delete_conversation(self, session_key: str, conversation_id: int, *, override: bool = False) -> bool:
        """Delete a conversation the session's effective user owns. Returns False
        (refused) on a cross-user attempt; raw deletes go through ``db`` directly.

        The access guard is the authorization — once it passes we delete by id
        (the ``db`` ``user_id`` scope is a separate defence-in-depth path for
        callers that bypass this guard)."""
        allowed = self.assert_conversation_access(session_key, conversation_id, override=override)
        _ledger.record_system(self.db, action_type="conversation_delete", ok=allowed,
                              session_key=session_key, conversation_id=conversation_id,
                              user_id=self.session_user_id(session_key),
                              error_code=None if allowed else "access_denied")
        if not allowed:
            return False
        if self.db is not None:
            self.db.delete_conversation(conversation_id)
        self._detach_deleted_conversation(conversation_id)
        bus.emit(CONVERSATION_CHANGED, {"action": "deleted", "conversation_id": conversation_id, "user_id": self.session_user_id(session_key)})
        return True

    def _detach_deleted_conversation(self, conversation_id: int) -> None:
        """Unbind any live session still holding a now-deleted conversation.

        A conversation can be deleted from a *different* session than the one
        viewing it (another tab, another frontend, the agent itself, or simply
        ``/conversations`` deleting the conversation that is currently open).
        The holding session would otherwise keep ``conversation_id`` pointing at
        a row that no longer exists and crash on its next write with a FOREIGN
        KEY violation. Detaching to ``None`` is safe: the no-conversation guard
        in ``handle_action`` then routes the user through ``/conversations``
        (the new-conversation model), and any stale per-user last-active pointer
        is dropped so startup restore doesn't trip over it either.
        """
        with self._sessions_lock:
            holders = [s for s in self.sessions.values()
                       if getattr(s, "conversation_id", None) == conversation_id]
        for session in holders:
            session.conversation_id = None
            # The conversation has ended in the most final way available. A
            # consumer keyed on it (reflection, summarization) has to hear that
            # here or it waits forever for a switch that cannot come.
            _persist.announce_conversation_ended(
                self, session.key, conversation_id, "deleted")
        for user_id, conv in list(self._persisted_active_conv_by_user.items()):
            if conv == conversation_id:
                self._persisted_active_conv_by_user.pop(user_id, None)

    def _reconcile_session_binding(self, session) -> None:
        """Self-heal a session whose conversation binding has gone stale.

        Defence-in-depth backstop for the whole "a mutation skipped an invariant
        a guard relies on" class. Before *any* action writes against
        ``session.conversation_id``, verify that row still exists and is still
        owned by the session's user. If not — deleted out from under it, or an
        ownership desync that slipped past some future mutator — detach to
        ``None`` so the no-conversation guard / lazy-create take over, instead of
        a FOREIGN KEY crash or a cross-user write. This catches desyncs no
        individual mutator remembered to reconcile, including ones not yet
        written; point fixes on the mutation side remain (cheaper, earlier), but
        this is the structural net under them.
        """
        cid = getattr(session, "conversation_id", None)
        if cid is None or self.db is None:
            return
        row = self.db.get_conversation(cid)
        if row is None:
            logger.warning(f"Session {session.key!r} held deleted conversation {cid}; detaching.")
            session.conversation_id = None
            return
        owner = row["user_id"] if "user_id" in row.keys() else DEFAULT_USER_ID
        eff = self.session_user_id(session.key)
        if owner != eff:
            logger.warning(
                f"Session {session.key!r} (user {eff}) held conversation {cid} "
                f"owned by user {owner}; detaching."
            )
            session.conversation_id = None

    def set_conversation_category(self, session_key: str, conversation_id: int, category: str | None, *, override: bool = False) -> bool:
        """Re-category a conversation the session's effective user owns. Returns
        False (refused) on a cross-user attempt."""
        allowed = self.assert_conversation_access(session_key, conversation_id, override=override)
        _ledger.record_system(self.db, action_type="conversation_recategorize", ok=allowed,
                              session_key=session_key, conversation_id=conversation_id,
                              user_id=self.session_user_id(session_key),
                              args={"category": category},
                              error_code=None if allowed else "access_denied")
        if not allowed:
            return False
        if self.db is not None:
            self.db.set_conversation_category(conversation_id, category)
        bus.emit(CONVERSATION_CHANGED, {"action": "recategorized", "conversation_id": conversation_id, "user_id": self.session_user_id(session_key), "category": category})
        return True

    def set_conversation_notification_mode(self, session_key: str, conversation_id: int, mode: str, *, override: bool = False) -> str | None:
        """Update notification mode for a live or stored conversation. Returns the
        normalized mode, or None (refused) on a cross-user attempt."""
        allowed = self.assert_conversation_access(session_key, conversation_id, override=override)
        _ledger.record_system(self.db, action_type="conversation_notification_mode", ok=allowed,
                              session_key=session_key, conversation_id=conversation_id,
                              user_id=self.session_user_id(session_key),
                              args={"mode": mode},
                              error_code=None if allowed else "access_denied")
        if not allowed:
            return None
        from runtime.notifications import notification_mode as normalize
        from state_machine.serialization import latest_state, save_state_marker
        normalized = normalize(mode)
        for session in list(self.sessions.values()):
            if session.conversation_id == conversation_id:
                with session.lock:
                    session.notification_mode = normalized
                    _persist._sync_notification_mode(session)
                    _persist.persist_marker(self, session)
                return normalized
        if self.db:
            marker = (latest_state(self.db.get_conversation_messages(conversation_id)) or {}).copy()
            marker["notification_mode"] = normalized
            save_state_marker(self.db, conversation_id, marker)
        return normalized

    @property
    def active_conversation_id(self) -> int | None:
        """Conversation id bound to the most recent user-driven session.

        Returns ``None`` if no user session has been touched yet, or if the
        active session has no conversation row.
        """
        key = self.active_session_key
        if not key:
            return None
        session = self.sessions.get(key)
        return session.conversation_id if session else None

    # ──────────────────────────────────────────────────────────────────
    # Interruption — the other half of ``cancel_event``. The flag says a turn
    # is over; this ends the call that is holding it open. Two stoppers,
    # because the two things a turn blocks on are answered differently: the
    # model call is one named box the pool leased, and tool calls are whatever
    # ephemeral runs the sandbox is tracking for this session.
    # ──────────────────────────────────────────────────────────────────

    def _interrupt_work(self, session) -> int:
        """End whatever this session is currently blocked on. Never raises.

        One place, so a third kind of blocking call cannot reintroduce the
        freeze by forgetting to be stopped — the same argument
        ``sandbox.events.publish`` and ``handlers.kernel._drive`` make for
        theirs. Best-effort at every step: a stopper that fails must not leave
        the cancel half-applied, since a turn that is flagged cancelled and
        still running is worse than either end of that.
        """
        stopped = 0
        try:
            stopped += session.interrupt()
        except Exception:
            logger.exception("interrupting session %s failed", session.key)
        try:
            from sandbox.bridge import get_sandbox
            stopped += get_sandbox().interrupt_session(session.key)
        except Exception:
            logger.exception("interrupting sandbox runs for %s failed", session.key)
        return stopped

    # ──────────────────────────────────────────────────────────────────
    # Attendance — "is a human present at this session right now?" The
    # kernel only *reads* this (interactive-tool gating, notification
    # routing, the notify prompt block). A frontend *owns* the policy:
    # by default a session is unattended unless it is the global active
    # one, but a concurrent multi-user frontend can override per session
    # via ``set_session_attended`` (e.g. on socket connect/disconnect).
    # ──────────────────────────────────────────────────────────────────

    def is_attended(self, session_key: str) -> bool:
        """Whether a human is present at ``session_key`` to answer prompts /
        see output. The owning frontend's explicit opinion wins; otherwise
        fall back to the global single-active-session rule."""
        session = self.sessions.get(session_key)
        if session is not None and session.attended is not None:
            return session.attended
        return session_key == self.active_session_key

    def set_session_attended(self, session_key: str, attended: bool | None) -> None:
        """Frontend hook: declare whether a human is present at ``session_key``.
        Pass ``None`` to relinquish the override and defer to the global rule."""
        session = self.sessions.get(session_key)
        if session is not None:
            session.attended = attended

    # ──────────────────────────────────────────────────────────────────
    # Security mode — "how does this conversation answer approval dialogs?"
    # The kernel only *reads* it in two places: the sandbox approver
    # (``sandbox/approval.py``) and the state machine's command grant
    # (``state_machine/action.py``). The person *owns* the policy, through
    # ``/mode``; nothing else may loosen it without being asked (see
    # ``sandbox.policy.classify``). Same division of labour as attendance
    # above, one axis over: that one asks whether anybody is there, this one
    # asks what they already said.
    # ──────────────────────────────────────────────────────────────────

    def security_mode(self, session_key: str) -> str:
        """The mode in force for ``session_key``.

        Precedence: a turn-scoped override, then the conversation's own mode,
        then the default. A missing session answers the default, which is the
        only safe direction — an approver that cannot find the session must
        ask rather than assume.

        The conversation check is the whole of "per conversation": a mode set
        against one ``conversation_id`` simply does not apply to another, so
        loading a different conversation, starting a new one, or being a
        subagent with a conversation of its own all begin at ``ask`` with
        nothing having to remember to reset anything.

        **Setting a mode before a conversation exists binds it late.** A
        session exists from the moment a frontend has a session key, but its
        conversation is created by the first message — so ``/mode lockdown``
        typed at a fresh prompt stamps ``None``, and the plain equality check
        then dropped the mode the instant the user said anything. Silently,
        and in the permissive direction, which is the worst pairing available:
        you set lockdown, the conversation opened, and the next shell command
        raised a dialog as though you had never typed it.

        The rule that fixes it is the one the check was always trying to
        express. Not "the id must match" but "this must still be the same
        piece of work" — and the conversation a session opens *right after*
        the mode was set is that same piece of work, not another one. So an
        unbound stamp adopts the first conversation it sees and is an ordinary
        stamp from then on, which keeps the leak the check exists to prevent:
        switching away afterwards still drops it.
        """
        session = self.sessions.get(session_key)
        if session is None:
            return DEFAULT_SECURITY_MODE
        if session.turn_security_mode:
            return _normalize_mode(session.turn_security_mode)
        if session.security_mode is None:
            return DEFAULT_SECURITY_MODE
        if session.security_mode_conversation is None:
            # Bind late. A reader that writes is worth the smell here: the
            # alternative is stamping at every site that assigns a
            # conversation_id, which is the list this design exists to avoid
            # keeping in step. Idempotent, and two threads can only ever write
            # the same value.
            session.security_mode_conversation = session.conversation_id
        if session.security_mode_conversation != session.conversation_id:
            return DEFAULT_SECURITY_MODE
        return _normalize_mode(session.security_mode)

    def set_security_mode(self, session_key: str, mode: str, *,
                          scope: str = CONVERSATION_SCOPE) -> str | None:
        """Set the mode for a conversation or for the rest of the turn.

        Returns the normalized mode that is now in force, or ``None`` when
        there is no such session. Normalizing here rather than at the call
        sites means an unknown value degrades to ``ask`` in one place instead
        of reaching the approver as something it has no answer for.
        """
        session = self.sessions.get(session_key)
        if session is None:
            return None
        resolved = _normalize_mode(mode)
        if scope_name(scope) == TURN_SCOPE:
            session.turn_security_mode = resolved
        else:
            session.security_mode = resolved
            session.security_mode_conversation = session.conversation_id
        bus.emit(SESSION_SECURITY_MODE_CHANGED, {
            "session_key": session_key,
            "conversation_id": session.conversation_id,
            "mode": resolved,
            "scope": scope_name(scope),
        })
        return resolved

    def clear_turn_security_mode(self, session_key: str) -> None:
        """Drop a turn-scoped mode. Called once per logical turn at its end."""
        session = self.sessions.get(session_key)
        if session is not None:
            session.turn_security_mode = None

    def is_turn_in_flight(self, session_key: str) -> bool:
        """Whether an agent turn is running right now for ``session_key``.

        The question a *turn-scoped* grant has to be able to answer before it
        is offered, and it is deliberately narrower than "does this chain name
        a session". ``session.busy`` is set for exactly the span of
        ``_drive_agent_turn`` — the same span that ends at
        ``HookRegistry.finish_turn``, which is what drops
        ``turn_security_mode``. So this is not a proxy for the turn: it is the
        window in which the grant has an expiry at all.

        Outside it there is a session, and a person watching, and no turn —
        a frontend acting as one of its sessions
        (``sdk.frontend.act``), a command mid-run. A turn-scoped grant made
        there is cleared by the *next* turn's end, which may be minutes and
        several actions away and is not what the button says. A missing
        session answers False for the same reason the mode reader answers the
        default: an approver that cannot find the session must not invent a
        scope for it.
        """
        session = self.sessions.get(session_key)
        return bool(session is not None and session.busy)

    # ──────────────────────────────────────────────────────────────────
    # User identity — "whose data is this session acting on?" Ephemeral,
    # frontend-bound (like attendance). Orthogonal to authorization, which
    # lives in frontend_profile. Ownership of conversations is the source of
    # truth (the user_id column); the session binding only decides which
    # user new rows are stamped with and is checked by the access guard.
    # ──────────────────────────────────────────────────────────────────

    def set_session_user(self, session_key: str, user_id: int | None) -> None:
        """Frontend hook: bind the user behind ``session_key`` (None ⇒ base user).

        Creates the session if it doesn't exist yet, so a frontend can bind
        identity up-front — before any conversation is created — and the first
        conversation gets stamped with the right owner.

        **Identity switch on a live session behaves like logging into another
        account.** If the session already holds a conversation and the user
        actually changes, the departing user's conversation is remembered as
        *their* last-active, then detached — a session must never keep holding a
        conversation its new identity does not own (the ownership guard runs on
        load/mutate-by-id paths, not on identity reassignment, so without this a
        re-identified session could read/append to the previous user's thread).
        The new identity is then dropped into *their* last-active conversation
        (from ``user_config``); if they have none, the session is left unbound
        and the next turn lazily creates a fresh conversation for them.
        """
        session = self.sessions.get(session_key)
        prev_uid = session.user_id if session is not None else None
        prev_conv = session.conversation_id if session is not None else None
        identity_changed = session is not None and prev_uid != user_id

        if identity_changed and prev_conv is not None:
            self._remember_last_active(
                prev_uid if prev_uid is not None else DEFAULT_USER_ID, prev_conv
            )
            self.close_session(session_key)

        _persist.get_or_create_session(self, session_key).user_id = user_id

        # Restore on every identity *switch*, not only when the departing user
        # held a conversation — the new user's landing spot shouldn't depend on
        # what the previous user happened to have open. (_load_last_active
        # no-ops when the session is already bound.)
        if identity_changed:
            self._load_last_active(session_key)

    def session_user_id(self, session_key: str) -> int:
        """The session's *effective* user — its frontend-bound user, or the base
        user when none was bound."""
        session = self.sessions.get(session_key)
        if session is not None and session.user_id is not None:
            return session.user_id
        return DEFAULT_USER_ID

    def user_config(self, session_key: str) -> dict:
        """Current user's config blob for ``session_key``."""
        return self.db.get_user_config(self.session_user_id(session_key)) if self.db is not None else {}

    def user_setting(self, session_key: str, key: str, default=None):
        """Current user's setting value, falling back to legacy/global config."""
        cfg = self.user_config(session_key)
        return cfg[key] if key in cfg else (self.config or {}).get(key, default)

    def assert_conversation_access(self, session_key: str, conversation_id: int, *, override: bool = False) -> bool:
        """Whether ``session_key``'s effective user may load/mutate the
        conversation. ``override=True`` (system/background callers) skips the
        check. Returns ``False`` rather than raising so user-driven paths can
        degrade to a clean "no such conversation" without leaking existence."""
        if override or self.db is None:
            return True
        row = self.db.get_conversation(conversation_id)
        if row is None:
            return False
        # A NULL owner belongs to the base user, never to everyone.
        owner = row.get("user_id")
        if owner is None:
            owner = DEFAULT_USER_ID
        return owner == self.session_user_id(session_key)

    # ──────────────────────────────────────────────────────────────────
    # Reopen-where-you-left-off persistence.
    #
    # The first user-driven action after process restart picks up that user's
    # last active conversation from their user config. After that, every
    # conversation *change* writes the new id back to the same user config so
    # the next restart lands in the same place. Per-action persistence is
    # intentionally avoided — only the actual switch event hits disk.
    # ──────────────────────────────────────────────────────────────────

    def restore_last_active(self, session_key: str) -> None:
        """Eager restore entry point for frontends to call at startup,
        before the user's first action — so the "Loaded last
        conversation" notice arrives right after the frontend's
        ready/online banner instead of mid-command.

        Announces through :meth:`notify` rather than handing back a string. It
        used to return the notice for the caller to render into the chat, which
        put a startup banner in the transcript and — worse — meant the *other*
        caller of ``_load_last_active`` (an identity switch, from
        ``set_session_user``) discarded the string and announced nothing at
        all. A notification reaches both, and a frontend without a notification
        surface still sees it flattened into chat exactly as before.
        """
        if session_key in self._restore_consumed_keys:
            return
        self._restore_consumed_keys.add(session_key)
        if not self.user_setting(session_key, "startup_restore_conversation", True):
            return
        self._load_last_active(session_key)

    def _load_last_active(self, session_key: str) -> None:
        """Load the current user's last-active conversation into the session.

        Shared by startup restore and identity-switch (``set_session_user``).
        No-ops when there is nothing accessible to restore or the session is
        already bound to a conversation."""
        conv_id = self._last_active_conversation_id(session_key)
        try:
            conv_id = int(conv_id) if conv_id not in (None, "") else None
        except (TypeError, ValueError):
            conv_id = None
        if conv_id is None or self.db is None or not self.assert_conversation_access(session_key, conv_id):
            return
        existing = self.sessions.get(session_key)
        if existing is not None and existing.conversation_id is not None:
            # Frontend re-attached to a session that already has a
            # conversation (mid-process reload, hot reattach) — leave it
            # alone, no restore needed.
            return
        try:
            session = self.load_conversation(session_key, conv_id)
        except Exception:
            logger.exception(f"Failed to restore last active conversation {conv_id}")
            return
        self._persisted_active_conv_by_user[self.session_user_id(session_key)] = conv_id
        title = (self.db.get_conversation(conv_id) or {}).get("title") or ""
        profile = session.profile_override or session.active_agent_profile or "default"
        suffix = f": {title.strip()}" if title.strip() else ""
        # The permission mode, on the same footing as the agent profile, and
        # read *after* the load rather than assumed. Restoring a conversation
        # does not restore the mode it was in — the mode is ephemeral, so a
        # restart always lands on the default — and the person who set
        # lockdown before quitting has no reason to suspect otherwise. Saying
        # it here costs one line and is the only moment it can be said.
        # Recovery notices are deliberately absent: ``open_session`` raised
        # them as notifications on the way here. A crash report is not a
        # footnote to "here is where you left off".
        self.notify(
            title=f"Loaded last conversation{suffix}",
            body=(f"Agent: {profile}"
                  f"\nPermission mode: {self.security_mode(session_key)}"),
            source="runtime", session_key=session_key, persist=False)

    def _persist_active_conversation(self, conv_id: int | None) -> None:
        """Remember the active conversation ID for the active session's user."""
        session_key = self.active_session_key
        if not session_key or self.db is None:
            return
        self._remember_last_active(self.session_user_id(session_key), conv_id)

    def _remember_last_active(self, user_id: int | None, conv_id: int | None) -> None:
        """Persist ``conv_id`` as ``user_id``'s last-active conversation.

        Split out from :meth:`_persist_active_conversation` so identity changes
        (``set_session_user``) can stamp the *departing* user's last-active even
        though that user is no longer the active session's user."""
        if self.db is None or user_id is None:
            return
        if self._persisted_active_conv_by_user.get(user_id) == conv_id:
            return
        cfg = self.db.get_user_config(user_id)
        if cfg.get("last_active_conversation_id") == conv_id:
            self._persisted_active_conv_by_user[user_id] = conv_id
            return
        cfg["last_active_conversation_id"] = conv_id
        try:
            self.db.set_user_config(user_id, cfg)
            self._persisted_active_conv_by_user[user_id] = conv_id
            logger.info(f"Persisted last_active_conversation_id={conv_id} for user_id={user_id}")
        except Exception:
            logger.exception("Failed to persist last_active_conversation_id")

    def _last_active_conversation_id(self, session_key: str):
        """Return the current user's remembered conversation id."""
        if self.db is None:
            return None
        cfg = self.db.get_user_config(self.session_user_id(session_key))
        if cfg.get("last_active_conversation_id") not in (None, ""):
            return cfg.get("last_active_conversation_id")
        return None

    # ──────────────────────────────────────────────────────────────────
    # Approval / typed-input requests.
    # ──────────────────────────────────────────────────────────────────

    def request_approval(self, session_key: str, title: str, body: str, pending_action: dict[str, Any]) -> StateMachineApprovalRequest:
        """Suspend a session on a yes-or-no approval request."""
        return _approvals.request_approval(self, session_key, title, body, pending_action)

    def request_input(
        self,
        session_key: str,
        title: str,
        prompt: str,
        *,
        type: str = "boolean",
        enum: list | None = None,
        enum_labels: list | None = None,
        default: Any = None,
        required: bool = True,
        pending_action: dict[str, Any] | None = None,
    ) -> StateMachineApprovalRequest:
        """Suspend a session on a typed-input request."""
        return _approvals.request_input(
            self, session_key, title, prompt,
            type=type, enum=enum, enum_labels=enum_labels, default=default,
            required=required, pending_action=pending_action,
        )

    def answer_request(self, session_key: str, request_id: str, value) -> RuntimeResult:
        """Resume a pending approval or input request with a provided answer."""
        return _approvals.answer_request(self, session_key, request_id, value)

    # ──────────────────────────────────────────────────────────────────
    # Plugin-facing API.
    #
    # Tools, tasks, and services can reach the runtime via
    # ``context.runtime`` (built in ``runtime/context.py``). The methods
    # below are the supported surface for *interacting* with sessions —
    # producing messages, mutating the agent's profile or system prompt,
    # registering session-pinned tools, and inspecting state.
    #
    # Anything not listed in this section is internal and may change.
    # ──────────────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return lightweight debug metadata for every live runtime session."""
        out = []
        for key, s in list(self.sessions.items()):
            out.append({
                "key": key,
                "agent_profile": s.profile_override or s.active_agent_profile,
                "phase": s.cs.phase,
                "turn_priority": s.cs.turn_priority,
                "conversation_id": s.conversation_id,
                "busy": s.busy,
                "plugin_state": list((s.plugin_state or {}).keys()),
                "system_prompt_extras": list(s.system_prompt_extras.keys()),
                "session_tools": [t.name for t in s.extra_tool_instances],
            })
        return out

    def get_session_plugin_state(self, session_key: str, plugin: str, key: str | None = None, default=None):
        """Return one plugin's session state, or one key inside it."""
        session = self.sessions.get(session_key)
        if session is None:
            return default
        state = (session.plugin_state or {}).get(plugin) or {}
        return state.get(key, default) if key is not None else state

    def update_session_plugin_state(self, session_key: str, plugin: str, patch: dict[str, Any] | None = None, **values) -> bool:
        """Merge values into one plugin's session state."""
        session = self.sessions.get(session_key)
        if session is None:
            return False
        session.plugin_state.setdefault(plugin, {}).update({**(patch or {}), **values})
        _persist.persist_marker(self, session)
        return True

    def push_message(self, session_key: str, text: str, *, title: str | None = None,
                     source: str | None = None, source_id: str | None = None,
                     attachments: list[str] | None = None) -> None:
        """Surface a message in a session (typically the foreground one).

        ``attachments`` are local paths the frontend renders alongside the
        text. This is the only outbound file route for anything that is not a
        tool: a tool hands files back on its ``ToolResult``, but a task, a
        command and a service each return something with no room for them.
        """
        payload = {"message": text, "session_key": session_key}
        if title:
            payload["title"] = title
        if source:
            payload["source"] = source
        if source_id:
            payload["source_id"] = source_id
        if attachments:
            payload["attachments"] = [str(p) for p in attachments]
        bus.emit(CHAT_MESSAGE_PUSHED, payload)

    def notify(self, *, title: str = "", body: str = "", source: str = "system",
               source_id: str | None = None, level: str = "info",
               session_key: str | None = None,
               source_session_key: str | None = None,
               conversation_id: int | None = None,
               persist: bool = True) -> int | None:
        """Raise a notification, filling in what only the runtime knows.

        The counterpart to :meth:`push_message`, and the distinction between
        them is *who is speaking*. A push is the conversation — the model's
        narration, a tool's files — and belongs in the chat view of every
        frontend. A notification is the system telling the user something, and
        a frontend may put it wherever it likes.

        Everything this adds over ``notifications.notify`` is context the
        module-level function has no way to reach: the database to persist to,
        and whose notification it is. ``user_id`` is read from the originating
        session rather than accepted as an argument, for the same reason the
        ledger's ``identity_of`` reads it from the context — ownership is the
        kernel's answer, not the caller's.
        """
        origin = source_session_key or session_key
        session = self.sessions.get(origin) if origin else None
        return _notifications.notify(
            title=title, body=body, source=source, source_id=source_id,
            level=level, session_key=session_key,
            source_session_key=source_session_key,
            conversation_id=conversation_id,
            user_id=getattr(session, "user_id", None),
            persist=persist, db=self.db)

    def set_agent_profile(self, session_key: str, profile: str) -> bool:
        """Switch the active agent profile for one live session."""
        session = self.sessions.get(session_key)
        if session is None:
            return False
        old = session.profile_override or session.active_agent_profile
        session.profile_override = profile
        session.active_agent_profile = profile
        _cfg.refresh_specs(self, session)
        bus.emit(SESSION_AGENT_PROFILE_CHANGED, {
            "session_key": session_key, "old_profile": old, "new_profile": profile,
        })
        return True

    def add_system_prompt_extra(self, session_key: str, key: str, value: str | None) -> bool:
        """Attach or clear one named system-prompt overlay for a session."""
        session = self.sessions.get(session_key)
        if session is None:
            return False
        if value is None:
            session.system_prompt_extras.pop(key, None)
        else:
            session.system_prompt_extras[key] = value
        bus.emit(SYSTEM_PROMPT_EXTRA_CHANGED, {
            "session_key": session_key, "key": key, "value": value,
        })
        return True

    def remove_system_prompt_extra(self, session_key: str, key: str) -> bool:
        """Remove system prompt extra."""
        return self.add_system_prompt_extra(session_key, key, None)

    def add_turn_attachment(self, session_key: str, attachment) -> bool:
        """Attach media to the next model call in this live session.

        This existed once with no caller and was deleted for it. What it was
        missing was the ``session.add_attachment`` Request, which is what any
        plugin authored since the sandbox migration would have to reach it by.
        """
        session = self.sessions.get(session_key)
        if session is None:
            return False
        return self.hooks.stage_attachment(session, attachment)

    def add_session_tool(self, session_key: str, tool_instance) -> bool:
        """Expose an extra tool instance to one live session."""
        session = self.sessions.get(session_key)
        if session is None:
            return False
        session.extra_tool_instances = [
            t for t in session.extra_tool_instances if getattr(t, "name", None) != getattr(tool_instance, "name", None)
        ]
        session.extra_tool_instances.append(tool_instance)
        _cfg.refresh_specs(self, session)
        return True

    def remove_session_tool(self, session_key: str, tool_name: str) -> bool:
        """Remove session tool."""
        session = self.sessions.get(session_key)
        if session is None:
            return False
        before = len(session.extra_tool_instances)
        session.extra_tool_instances = [
            t for t in session.extra_tool_instances if getattr(t, "name", None) != tool_name
        ]
        if len(session.extra_tool_instances) != before:
            _cfg.refresh_specs(self, session)
            return True
        return False

    def cancel_session(self, session_key: str) -> RuntimeResult | None:
        """Cancel the current in-flight action for a session, if it exists."""
        if session_key not in self.sessions:
            return None
        return self.handle_action(session_key, "cancel")

    def refresh_session_specs(self) -> None:
        """Re-read the global commands/tools into every live session.

        Plugin reload paths call this so freshly-built tools are visible
        to running agents without needing /restart.
        """
        for session in list(self.sessions.values()):
            _cfg.refresh_specs(self, session)
