from __future__ import annotations

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

import logging
import threading
from typing import Any, Callable

from events.event_bus import bus
from events.event_channels import (
    CHAT_MESSAGE_PUSHED,
    SESSION_AGENT_PROFILE_CHANGED,
    SESSION_CLOSED,
    SYSTEM_PROMPT_EXTRA_CHANGED,
)

from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversation import CallableSpec
from state_machine.conversation_phases import BASE_PHASE, BUSY_PHASES, FORM_PHASES, PHASE_APPROVING_REQUEST
from state_machine.errors import ActionError
from runtime.session import RuntimeResult, RuntimeSession, SessionConflict

from runtime import runtime_approvals as _approvals
from runtime import runtime_config as _cfg
from runtime import dispatch as _disp
from runtime import persistence as _persist

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
        self._pending_restore_conv_id = self.config.get("last_active_conversation_id") if self.config else None
        self._restore_consumed_keys: set[str] = set()
        self._persisted_active_conv_id = self._pending_restore_conv_id
        # Compaction request registry: token -> {"event": Event, "summary": str|None, "error": str|None}.
        # ConversationLoop fills this with a fresh entry, emits COMPACT_CHAT,
        # and waits on the event. The compact_chat task posts the summary
        # back via _finish_compaction.
        self._compaction_pending: dict[str, dict] = {}
        self._compaction_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────
    # Public entrypoint — every action a frontend can take ends up here.
    # ──────────────────────────────────────────────────────────────────

    def handle_action(self, session_key: str, action_type: str, payload: dict | str | None = None, *, user_driven: bool = True) -> RuntimeResult:
        session = self.get_session(session_key)
        if user_driven:
            self.active_session_key = session_key
            prior_conv = getattr(self, "_persisted_active_conv_id", None)

        # Cron-handoff guard: a non-user-driven send_text must never be
        # interpreted as form input. If the user is mid-form, refuse the turn.
        if (not user_driven
                and action_type == "send_text"
                and session.cs.phase in FORM_PHASES):
            return RuntimeResult(False,
                                 messages=["Session is mid-form — handoff deferred."],
                                 error={"code": "busy", "message": "form in progress"})

        # Busy guard: if the session is mid-turn, only ``cancel`` and the
        # specific ``answer_approval`` for an active approval frame may
        # proceed. Everything else is told to wait or cancel first.
        if session.busy or session.cs.phase in BUSY_PHASES:
            if action_type == "cancel" and session.cs.phase == PHASE_APPROVING_REQUEST:
                pass
            elif action_type == "cancel":
                session.cancel_event.set()
                return RuntimeResult(messages=["Cancelled."])
            if action_type == "answer_approval" and session.cs.phase == PHASE_APPROVING_REQUEST:
                pass  # fall through and dispatch
            elif action_type == "send_text":
                return RuntimeResult(messages=["Not your turn - I'm still working. Send /cancel to interrupt."])
            elif action_type != "answer_approval" or session.cs.phase != PHASE_APPROVING_REQUEST:
                return RuntimeResult(False, messages=["Still working. Send /cancel to interrupt."], error={"code": "busy", "message": "Still working."})

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
            return RuntimeResult(False, messages=["No conversation loaded.\nTry /conversations to add a new one."])

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
        if out.data.pop("_drive_agent_turn", False):
            self._drive_agent_turn(session, out)
            with session.lock:
                _persist.persist_marker(self, session)

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
        # cs.pending_attachments so the next agent turn picks them up.
        if inbound_attachments:
            from attachments.attachment import Attachment
            for entry in inbound_attachments:
                if isinstance(entry, Attachment):
                    session.cs.pending_attachments.append(entry)
                elif isinstance(entry, dict):
                    session.cs.pending_attachments.append(Attachment.from_dict(entry))

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
        result = session.cs.enact(action_type, content, actor_id)
        # ──────────────────────────────────────────────────────────────

        out.add_action_result(result)
        _approvals.resolve_answered_request(self, request_id, result)
        text = _disp.text_after_action(action_type, text, result)
        _disp.absorb_user_action(self, session, action_type, text, result)
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

    def _drive_agent_turn(self, session: RuntimeSession, out: RuntimeResult) -> RuntimeResult:
        _persist.ensure_conversation(session=session, runtime=self, title_text=_disp.latest_user_text(session))
        session.busy = True
        session.cancel_event.clear()
        _persist.persist_marker(self, session)  # busy=True snapshot for crash recovery
        try:
            loop = _cfg.build_loop(self, session.key)
            reply, new_messages, attachments = loop.drive(
                session.cs,
                "agent",
                session.history,
                self.db,
                session.conversation_id,
            )
        except Exception as e:
            err = ActionError("agent_failed", str(e))
            session.cs.last_error = err
            out.ok = False
            out.error = err.to_dict()
            out.messages.append(str(e))
            return out
        finally:
            session.busy = False
            # Safety net: EndTurn should already have handed priority back, but
            # if drive() raised partway through, force the user back into
            # priority so the conversation can continue.
            if session.cs.turn_priority != "user":
                session.cs.set_priority("user")
            session.cancel_event.clear()

        if reply:
            out.messages.append(reply)
            from events.event_channels import SESSION_MESSAGE
            bus.emit(SESSION_MESSAGE, {
                "session_key": session.key,
                "role": "assistant",
                "content": reply,
                "actor_id": "agent",
            })
        elif new_messages:
            # Agent ended without final text but produced messages — surface
            # the last assistant content if any, otherwise flag as cancelled.
            last_assistant = next((m for m in reversed(new_messages) if m.get("role") == "assistant"), None)
            out.messages.append(last_assistant.get("content") if last_assistant and last_assistant.get("content") else "Cancelled.")
        else:
            out.messages.append("Cancelled.")

        out.attachments.extend(attachments)
        out.data.setdefault("conversation_id", session.conversation_id)
        out.data.setdefault("new_messages", []).extend(new_messages)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Session lifecycle.
    # ──────────────────────────────────────────────────────────────────

    def get_session(self, key: str) -> RuntimeSession:
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
    ) -> RuntimeSession:
        """Create-or-load a session bound to a specific conversation.

        See :func:`state_machine.runtime_persistence.open_session`.
        """
        return _persist.open_session(
            self, session_key,
            conversation_id=conversation_id, kind=kind, category=category,
            title=title, agent_profile=agent_profile,
            system_prompt_extras=system_prompt_extras,
        )

    def create_conversation(self, title: str = "New Conversation", *, kind: str = "user", category: str | None = None) -> int | None:
        return _persist.create_conversation(self, title, kind=kind, category=category)

    def load_conversation(self, session_key: str, conversation_id: int, *, agent_profile: str | None = None, system_prompt_extras: dict[str, Any] | None = None) -> RuntimeSession:
        return _persist.load_conversation(self, session_key, conversation_id, agent_profile=agent_profile, system_prompt_extras=system_prompt_extras)

    def load_history(self, session_key: str, conversation_id: int) -> RuntimeResult:
        return _persist.load_history(self, session_key, conversation_id)

    def reset_conversation(self, session_key: str) -> RuntimeSession:
        return _persist.reset_conversation(self, session_key)

    def new_conversation(self, session_key: str) -> RuntimeResult:
        return _persist.new_conversation(self, session_key)

    def iterate_agent_turn(self, session_key: str, prompt: str, *, attachments=None, actor_id: str = "user") -> RuntimeResult:
        return _persist.iterate_agent_turn(self, session_key, prompt, attachments=attachments, actor_id=actor_id)

    def inject_user_message(self, session_key: str, text: str, *, conversation_id: int | None = None, actor_id: str = "user") -> RuntimeResult:
        return _persist.inject_user_message(self, session_key, text, conversation_id=conversation_id, actor_id=actor_id)

    def close_session(self, session_key: str) -> bool:
        return _persist.close_session(self, session_key)

    def unload_conversation(self, session_key: str) -> bool:
        """Alias for plugin code that reads in conversation lifecycle terms."""
        return self.close_session(session_key)

    def set_conversation_notification_mode(self, conversation_id: int, mode: str) -> str:
        """Update notification mode for a live or stored conversation."""
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
    # Reopen-where-you-left-off persistence.
    #
    # The first user-driven action after process restart picks up the last
    # active conversation from config. After that, every conversation
    # *change* writes the new id back to config so the next restart lands
    # in the same place. Per-action persistence is intentionally avoided —
    # only the actual switch event hits disk.
    # ──────────────────────────────────────────────────────────────────

    def restore_last_active(self, session_key: str) -> str | None:
        """Eager restore entry point for frontends to call at startup,
        before the user's first action — so the "Loaded last
        conversation" notice arrives right after the frontend's
        ready/online banner instead of mid-command."""
        if session_key in self._restore_consumed_keys:
            return None
        self._restore_consumed_keys.add(session_key)
        if self.config and not self.config.get("startup_restore_conversation", True):
            return None
        conv_id = self._pending_restore_conv_id
        try:
            conv_id = int(conv_id) if conv_id not in (None, "") else None
        except (TypeError, ValueError):
            conv_id = None
        if conv_id is None or self.db is None or self.db.get_conversation(conv_id) is None:
            return None
        existing = self.sessions.get(session_key)
        if existing is not None and existing.conversation_id is not None:
            # Frontend re-attached to a session that already has a
            # conversation (mid-process reload, hot reattach) — leave it
            # alone, no restore needed.
            return None
        try:
            session = _persist.load_conversation(self, session_key, conv_id)
        except Exception:
            logger.exception(f"Failed to restore last active conversation {conv_id}")
            return None
        self._persisted_active_conv_id = conv_id
        title = (self.db.get_conversation(conv_id) or {}).get("title") or ""
        profile = session.profile_override or session.active_agent_profile or "default"
        suffix = f": {title.strip()}" if title.strip() else ""
        return f"Loaded last conversation{suffix}.\nAgent: {profile}"

    # ──────────────────────────────────────────────────────────────────
    # Compaction handoff. The actual summarization runs as an event-driven
    # task (``task_compact_chat``) so it has its own observable run record
    # and can be cancelled / inspected like any other task. The loop
    # blocks here waiting for the result.
    # ──────────────────────────────────────────────────────────────────

    def request_compaction(self, session_key: str | None, transcript: str, timeout: float = 120.0) -> str | None:
        import uuid
        from events.event_channels import COMPACT_CHAT
        token = f"compact:{uuid.uuid4().hex}"
        slot = {"event": threading.Event(), "summary": None, "error": None}
        with self._compaction_lock:
            self._compaction_pending[token] = slot
        try:
            bus.emit(COMPACT_CHAT, {
                "request_token": token,
                "session_key": session_key,
                "transcript": transcript,
            })
            if not slot["event"].wait(timeout=timeout):
                logger.warning(f"Compaction timed out after {timeout}s (token {token})")
                return None
            return slot["summary"]
        finally:
            with self._compaction_lock:
                self._compaction_pending.pop(token, None)

    def finish_compaction(self, token: str, summary: str | None, error: str | None = None) -> None:
        with self._compaction_lock:
            slot = self._compaction_pending.get(token)
        if slot is None:
            logger.warning(f"finish_compaction: no pending slot for token {token}")
            return
        slot["summary"] = summary
        slot["error"] = error
        slot["event"].set()

    def _persist_active_conversation(self, conv_id: int | None) -> None:
        self._persisted_active_conv_id = conv_id
        if self.config is None:
            return
        if self.config.get("last_active_conversation_id") == conv_id:
            return
        self.config["last_active_conversation_id"] = conv_id
        try:
            import config.config_manager as config_manager
            config_manager.save(self.config)
            logger.info(f"Persisted last_active_conversation_id={conv_id}")
        except Exception:
            logger.exception("Failed to persist last_active_conversation_id")

    # ──────────────────────────────────────────────────────────────────
    # Approval / typed-input requests.
    # ──────────────────────────────────────────────────────────────────

    def request_approval(self, session_key: str, title: str, body: str, pending_action: dict[str, Any]) -> StateMachineApprovalRequest:
        return _approvals.request_approval(self, session_key, title, body, pending_action)

    def request_input(
        self,
        session_key: str,
        title: str,
        prompt: str,
        *,
        type: str = "boolean",
        enum: list | None = None,
        default: Any = None,
        required: bool = True,
        pending_action: dict[str, Any] | None = None,
    ) -> StateMachineApprovalRequest:
        return _approvals.request_input(
            self, session_key, title, prompt,
            type=type, enum=enum, default=default,
            required=required, pending_action=pending_action,
        )

    def answer_request(self, session_key: str, request_id: str, value) -> RuntimeResult:
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
        out = []
        for key, s in list(self.sessions.items()):
            out.append({
                "key": key,
                "agent_profile": s.profile_override or s.active_agent_profile,
                "phase": s.cs.phase,
                "turn_priority": s.cs.turn_priority,
                "conversation_id": s.conversation_id,
                "busy": s.busy,
                "system_prompt_extras": list(s.system_prompt_extras.keys()),
                "session_tools": [t.name for t in s.extra_tool_instances],
            })
        return out

    def push_message(self, session_key: str, text: str, *, title: str | None = None,
                     source: str | None = None, source_id: str | None = None) -> None:
        """Surface a message in a session (typically the foreground one)."""
        payload = {"message": text, "session_key": session_key}
        if title:
            payload["title"] = title
        if source:
            payload["source"] = source
        if source_id:
            payload["source_id"] = source_id
        bus.emit(CHAT_MESSAGE_PUSHED, payload)

    def set_agent_profile(self, session_key: str, profile: str) -> bool:
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
        return self.add_system_prompt_extra(session_key, key, None)

    def add_session_tool(self, session_key: str, tool_instance) -> bool:
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
