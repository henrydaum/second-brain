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
        self._legacy_restore_conv_id = self.config.get("last_active_conversation_id") if self.config else None
        self._restore_consumed_keys: set[str] = set()
        self._persisted_active_conv_by_user: dict[int, int | None] = {}
    # ──────────────────────────────────────────────────────────────────
    # Public entrypoint — every action a frontend can take ends up here.
    # ──────────────────────────────────────────────────────────────────

    def handle_action(self, session_key: str, action_type: str, payload: dict | str | None = None, *, user_driven: bool = True) -> RuntimeResult:
        """Route one frontend action through guards, dispatch, and optional agent follow-up."""
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

        if action_type == "cancel" and not session.busy and session.cs.phase == BASE_PHASE:
            return RuntimeResult(messages=["Nothing to cancel."])

        # Busy guard: if the session is mid-turn, only ``cancel`` and the
        # specific ``answer_approval`` for an active approval frame may
        # proceed. Everything else is told to wait or cancel first.
        if session.busy or session.cs.phase in BUSY_PHASES:
            if action_type == "cancel" and session.cs.phase != PHASE_APPROVING_REQUEST:
                session.cancel_event.set()
                return RuntimeResult(messages=["Cancelled."])
            if action_type in {"answer_approval", "cancel"} and session.cs.phase == PHASE_APPROVING_REQUEST:
                pass  # fall through and dispatch
            elif action_type == "send_text":
                return RuntimeResult(messages=["Not your turn - I'm still working. Send /cancel to interrupt."])
            else:
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
            if not (self.config.get("llm_profiles") or {}):
                return RuntimeResult(False, messages=[
                    "Welcome to Second Brain. Run /setup to configure an LLM and the Telegram frontend."
                ])
            return RuntimeResult(False, messages=["No conversation loaded.\nTry /new."])

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
        """Run the agent loop for a session and surface its outputs back to the frontend."""
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
            hooks = getattr(self, "hooks", None)
            if hooks is not None:
                hooks.finish_turn(session)

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
        return _persist.create_conversation(self, title, kind=kind, category=category, user_id=user_id)

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
            return RuntimeResult(False, messages=["No such conversation."])
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
            return RuntimeResult(False, messages=["No such conversation."])
        return _persist.inject_user_message(self, session_key, text, conversation_id=conversation_id, actor_id=actor_id)

    def close_session(self, session_key: str) -> bool:
        """Close one live session and persist its final marker state."""
        return _persist.close_session(self, session_key)

    def unload_conversation(self, session_key: str) -> bool:
        """Alias for plugin code that reads in conversation lifecycle terms."""
        return self.close_session(session_key)

    def delete_conversation(self, session_key: str, conversation_id: int, *, override: bool = False) -> bool:
        """Delete a conversation the session's effective user owns. Returns False
        (refused) on a cross-user attempt; raw deletes go through ``db`` directly.

        The access guard is the authorization — once it passes we delete by id
        (the ``db`` ``user_id`` scope is a separate defence-in-depth path for
        callers that bypass this guard)."""
        if not self.assert_conversation_access(session_key, conversation_id, override=override):
            return False
        if self.db is not None:
            self.db.delete_conversation(conversation_id)
        return True

    def set_conversation_category(self, session_key: str, conversation_id: int, category: str | None, *, override: bool = False) -> bool:
        """Re-category a conversation the session's effective user owns. Returns
        False (refused) on a cross-user attempt."""
        if not self.assert_conversation_access(session_key, conversation_id, override=override):
            return False
        if self.db is not None:
            self.db.set_conversation_category(conversation_id, category)
        return True

    def set_conversation_notification_mode(self, session_key: str, conversation_id: int, mode: str, *, override: bool = False) -> str | None:
        """Update notification mode for a live or stored conversation. Returns the
        normalized mode, or None (refused) on a cross-user attempt."""
        if not self.assert_conversation_access(session_key, conversation_id, override=override):
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

        if identity_changed and prev_conv is not None:
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

    def set_user_setting(self, session_key: str, key: str, value) -> None:
        """Persist one setting in the current user's config blob."""
        if self.db is None:
            return
        user_id = self.session_user_id(session_key)
        cfg = self.db.get_user_config(user_id)
        cfg[key] = value
        self.db.set_user_config(user_id, cfg)

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
        owner = row.get("user_id")
        return owner is None or owner == self.session_user_id(session_key)

    # ──────────────────────────────────────────────────────────────────
    # Reopen-where-you-left-off persistence.
    #
    # The first user-driven action after process restart picks up that user's
    # last active conversation from their user config. After that, every
    # conversation *change* writes the new id back to the same user config so
    # the next restart lands in the same place. Per-action persistence is
    # intentionally avoided — only the actual switch event hits disk.
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
        return self._load_last_active(session_key)

    def _load_last_active(self, session_key: str) -> str | None:
        """Load the current user's last-active conversation into the session.

        Shared by startup restore and identity-switch (``set_session_user``).
        No-ops (returns ``None``) when there is nothing accessible to restore or
        the session is already bound to a conversation."""
        conv_id = self._last_active_conversation_id(session_key)
        try:
            conv_id = int(conv_id) if conv_id not in (None, "") else None
        except (TypeError, ValueError):
            conv_id = None
        if conv_id is None or self.db is None or not self.assert_conversation_access(session_key, conv_id):
            return None
        existing = self.sessions.get(session_key)
        if existing is not None and existing.conversation_id is not None:
            # Frontend re-attached to a session that already has a
            # conversation (mid-process reload, hot reattach) — leave it
            # alone, no restore needed.
            return None
        try:
            session = self.load_conversation(session_key, conv_id)
        except Exception:
            logger.exception(f"Failed to restore last active conversation {conv_id}")
            return None
        self._persisted_active_conv_by_user[self.session_user_id(session_key)] = conv_id
        title = (self.db.get_conversation(conv_id) or {}).get("title") or ""
        profile = session.profile_override or session.active_agent_profile or "default"
        suffix = f": {title.strip()}" if title.strip() else ""
        return f"Loaded last conversation{suffix}.\nAgent: {profile}"

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
        """Return the current user's remembered conversation id.

        One-time legacy fallback reads the old global config key for the base user
        so existing local installs restore once and then rewrite into user config.
        """
        if self.db is None:
            return None
        user_id = self.session_user_id(session_key)
        cfg = self.db.get_user_config(user_id)
        if cfg.get("last_active_conversation_id") not in (None, ""):
            return cfg.get("last_active_conversation_id")
        if user_id == DEFAULT_USER_ID and self._legacy_restore_conv_id not in (None, ""):
            cfg["last_active_conversation_id"] = self._legacy_restore_conv_id
            try:
                self.db.set_user_config(user_id, cfg)
            except Exception:
                logger.exception("Failed to migrate last_active_conversation_id to user config")
            return self._legacy_restore_conv_id
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
        default: Any = None,
        required: bool = True,
        pending_action: dict[str, Any] | None = None,
    ) -> StateMachineApprovalRequest:
        """Suspend a session on a typed-input request."""
        return _approvals.request_input(
            self, session_key, title, prompt,
            type=type, enum=enum, default=default,
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

    def clear_session_plugin_state(self, session_key: str, plugin: str, *keys: str) -> bool:
        """Clear one plugin state bag, or selected keys in it."""
        session = self.sessions.get(session_key)
        if session is None:
            return False
        if keys:
            for key in keys:
                (session.plugin_state.get(plugin) or {}).pop(key, None)
        else:
            session.plugin_state.pop(plugin, None)
        _persist.persist_marker(self, session)
        return True

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
