from __future__ import annotations

"""Adapter-facing conversation runtime.

This is the single dispatcher between a frontend transport (REPL, Telegram,
future event bus) and the state machine. The runtime owns sessions and
persistence, but every state-changing decision goes through one labeled
`cs.enact(...)` site (see `_dispatch`). When the user's action hands turn
priority to the agent, the runtime hands off to `ConversationLoop.drive()`,
which contains its own labeled `cs.enact(...)` site for the agent's moves.

That two-call-site shape mirrors PokerMonster's `run_game`: one obvious line
where everything flows through, easy to find, easy to read.
"""

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversationClass import CallableSpec, ConversationState, FormStep, Participant, PhaseFrame
from state_machine.conversation_loop import ConversationLoop
from state_machine.conversation_phases import BASE_PHASE, BUSY_PHASES, PHASE_APPROVING_REQUEST
from state_machine.errors import ActionError, ActionResult
from events.event_bus import bus
from events.event_channels import (
    CHAT_MESSAGE_PUSHED,
    SESSION_AGENT_PROFILE_CHANGED,
    SESSION_CLOSED,
    SESSION_CREATED,
    SESSION_MESSAGE,
    SESSION_PHASE_CHANGED,
    SESSION_TURN_COMPLETED,
    SESSION_TURN_CHANGED,
    SYSTEM_PROMPT_EXTRA_CHANGED,
    TOOL_CALL_FINISHED,
    TOOL_CALL_STARTED,
)
from state_machine.forms import schema_to_form_steps
from state_machine.persistence import latest_state, messages_to_history, save_history_message, save_state_marker


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
        self.ok = self.ok and result.ok
        self.events.extend(result.events)
        if result.message:
            self.messages.append(result.message)
        if result.error:
            self.error = result.error.to_dict()
            self.messages.append(result.error.message)
        return self


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
    # instances (e.g. NotifyTool) that are not part of the global
    # tool_registry. When None / empty, the session follows the runtime's
    # active profile and uses the global tool registry.
    profile_override: str | None = None
    extra_tool_instances: list = field(default_factory=list)
    is_subagent: bool = False
    subagent_meta: dict[str, Any] = field(default_factory=dict)
    system_prompt_extras: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_marker(self) -> dict[str, Any]:
        state = self.cs.to_dict()
        state.update({
            "conversation_id": self.conversation_id,
            "active_agent_profile": self.active_agent_profile,
            "profile_override": self.profile_override,
            "is_subagent": self.is_subagent,
            "subagent_meta": self.subagent_meta,
            "system_prompt_extras": self.system_prompt_extras,
            "busy": self.busy,
        })
        return state


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
        title_callback: Callable[[int], None] | None = None,
        on_tool_start=None,
        on_tool_result=None,
        on_notice=None,
    ):
        self.db = db
        self.services = services or {}
        self.config = config or {}
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.commands = {**(commands or {}), **self._command_specs(command_specs or {})}
        self.emit_event = emit_event
        self.title_callback = title_callback
        self.on_tool_start = on_tool_start
        self.on_tool_result = on_tool_result
        self.on_notice = on_notice
        self.sessions: dict[str, RuntimeSession] = {}
        self._sessions_lock = threading.RLock()

    # ──────────────────────────────────────────────────────────────────────
    # Public entrypoint — every action a frontend can take ends up here.
    # ──────────────────────────────────────────────────────────────────────

    def handle_action(self, session_key: str, action_type: str, payload: dict | str | None = None) -> RuntimeResult:
        session = self.get_session(session_key)
        normalized = "send_text" if action_type == "chat_message" else action_type
        if session.busy or session.cs.phase in BUSY_PHASES:
            if normalized == "cancel":
                session.cancel_event.set()
                return RuntimeResult(messages=["Cancelled."])
            if normalized == "send_text":
                return RuntimeResult(messages=["Not your turn - I'm still working. Send /cancel to interrupt."])
            return RuntimeResult(False, messages=["Still working. Send /cancel to interrupt."], error={"code": "busy", "message": "Still working."})
        with session.lock:
            try:
                self._refresh_session_specs(session)
                out = self._dispatch(session, action_type, payload)
            finally:
                if action_type not in {"load_history", "new_conversation"}:
                    self._persist_marker(session)
        drive_images = out.data.pop("_drive_agent_image_paths", None)
        if drive_images is not None:
            self._drive_agent_turn(session, out, drive_images)
            with session.lock:
                self._persist_marker(session)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # The single user-side dispatch. One labeled `cs.enact()` line, plus a
    # hand-off to ConversationLoop when the action transferred priority to
    # the agent.
    # ──────────────────────────────────────────────────────────────────────

    def _dispatch(self, session: RuntimeSession, action_type: str, payload: dict | str | None) -> RuntimeResult:
        # Transport-level actions that never enter the state machine.
        if action_type == "load_history":
            return self.load_history(session.key, int((payload or {}).get("conversation_id")))
        if action_type == "new_conversation":
            return self.new_conversation(session.key)

        # Normalize legacy aliases.
        if action_type == "chat_message":
            action_type = "send_text"

        text = self._text(payload)
        image_paths = self._image_paths(payload)
        actor_id = self._actor_id(payload)

        # Empty-input guard, matching v1 behavior. Skips state-machine entry
        # so we don't pollute history/events with a doomed action.
        if action_type in {"send_text", "send_attachment"} and not text and not image_paths and action_type != "send_attachment":
            return RuntimeResult(False, error={"code": "empty_input", "message": "No input."})

        # Predicate captured *before* enact, since the action itself may
        # transition phase/priority. This is what tells us whether an agent
        # reply turn should follow.
        expects_agent_reply = (
            action_type in {"send_text", "send_attachment"}
            and session.cs.phase == BASE_PHASE
            and session.cs.turn_priority == "user"
        )

        content = self._content_for_action(action_type, text, payload)

        out = RuntimeResult()

        old_phase = session.cs.phase
        old_priority = session.cs.turn_priority

        # ──────────────── THE enact() SITE (user-side) ────────────────
        result = session.cs.enact(action_type, content, actor_id)
        # ──────────────────────────────────────────────────────────────

        out.add_action_result(result)
        text = self._result_text(action_type, text, result)
        image_paths = self._result_image_paths(action_type, image_paths, result)
        self._absorb_user_action(session, action_type, text, result)
        self._emit_state_change(session, old_phase, old_priority)
        self._decorate_form(session, out)
        self._echo_callable_result(action_type, result, out)

        if (expects_agent_reply
                and result.ok
                and session.cs.turn_priority == "agent"
                and session.cs.phase == BASE_PHASE):
            out.data["_drive_agent_image_paths"] = image_paths

        return out

    # ──────────────────────────────────────────────────────────────────────
    # Driving the agent's turn. ConversationLoop has its OWN labeled
    # cs.enact() site inside it; this method just sets up persistence and
    # surface the loop's outputs.
    # ──────────────────────────────────────────────────────────────────────

    def _drive_agent_turn(self, session: RuntimeSession, out: RuntimeResult, image_paths: list[str] | None) -> RuntimeResult:
        self._ensure_conversation(session, self._latest_user_text(session))
        session.busy = True
        session.cancel_event.clear()
        try:
            loop = self._loop(session.key)
            reply, new_messages, attachments = loop.drive(
                session.cs,
                "agent",
                session.history,
                self.db,
                session.conversation_id,
                image_paths,
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

        if (self.title_callback and session.conversation_id
                and any(m.get("role") == "assistant" and not m.get("tool_calls") for m in new_messages)):
            self.title_callback(session.conversation_id)

        if reply:
            out.messages.append(reply)
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

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _emit_state_change(self, session: RuntimeSession, old_phase: str, old_priority: str) -> None:
        if session.cs.phase != old_phase:
            bus.emit(SESSION_PHASE_CHANGED, {
                "session_key": session.key,
                "old_phase": old_phase,
                "new_phase": session.cs.phase,
            })
        if session.cs.turn_priority != old_priority:
            bus.emit(SESSION_TURN_CHANGED, {
                "session_key": session.key,
                "from_actor": old_priority,
                "to_actor": session.cs.turn_priority,
            })

    def _absorb_user_action(self, session: RuntimeSession, action_type: str, text: str, result: ActionResult) -> None:
        """Translate user-side action outcomes into history rows + side effects.

        Mirrors `ConversationLoop._absorb` but for actions originating from
        the frontend. Only `send_text` / `send_attachment` (with text) add a
        chat-transcript row; commands/forms/approvals have no provider-history
        impact, only state-machine impact.
        """
        if not result.ok:
            return
        if action_type in {"send_text", "send_attachment"} and text:
            msg = {"role": "user", "content": text}
            session.history.append(msg)
            if self.db and session.conversation_id:
                save_history_message(self.db, session.conversation_id, msg)
            bus.emit(SESSION_MESSAGE, {
                "session_key": session.key,
                "role": "user",
                "content": text,
                "actor_id": "user",
            })

    def _echo_callable_result(self, action_type: str, result: ActionResult, out: RuntimeResult) -> None:
        """Surface command/tool return values to the frontend (v1 behavior)."""
        if action_type not in {"call_command", "call_tool"} and getattr(result, "action", None) not in {"call_command", "call_tool"}:
            return
        if not result.ok:
            return
        value = (result.data or {}).get("result")
        if value is not None:
            out.messages.append(str(value))

    def _content_for_action(self, action_type: str, text: str, payload: Any) -> Any:
        """Pick the right content shape per action type.

        SendText accepts a plain string. Form/approval/attachment actions
        accept the original payload so the action can read structured fields.
        """
        if action_type == "send_text":
            return text
        if action_type == "submit_form_text":
            return text
        return payload

    def _result_text(self, action_type: str, text: str, result: ActionResult) -> str:
        if action_type != "send_attachment" or not result.ok:
            return text
        parsed = (result.data or {}).get("parsed")
        return str((parsed or {}).get("text") or text) if isinstance(parsed, dict) else text

    def _result_image_paths(self, action_type: str, image_paths: list[str], result: ActionResult) -> list[str]:
        if action_type != "send_attachment" or not result.ok:
            return image_paths
        parsed = (result.data or {}).get("parsed")
        return list((parsed or {}).get("image_paths") or image_paths) if isinstance(parsed, dict) else image_paths

    # ──────────────────────────────────────────────────────────────────────
    # Session + persistence + setup
    # ──────────────────────────────────────────────────────────────────────

    def get_session(self, key: str) -> RuntimeSession:
        with self._sessions_lock:
            if key not in self.sessions:
                session = RuntimeSession(key, self._new_state())
                session.cs = self._new_state(session=session)
                self.sessions[key] = session
                bus.emit(SESSION_CREATED, {
                    "session_key": key,
                    "is_subagent": False,
                    "agent_profile": session.active_agent_profile,
                })
            return self.sessions[key]

    def create_conversation(self, title: str = "New conversation", *, kind: str = "user") -> int | None:
        return self.db.create_conversation(title, kind=kind) if self.db else None

    def load_conversation(
        self,
        session_key: str,
        conversation_id: int,
        *,
        agent_profile: str | None = None,
        system_prompt_extras: dict[str, Any] | None = None,
    ) -> RuntimeSession:
        rows = self.db.get_conversation_messages(conversation_id) if self.db else []
        marker = latest_state(rows) or {}
        conv = self.db.get_conversation(conversation_id) if self.db else {}
        is_subagent = (conv or {}).get("kind") == "subagent"
        profile = agent_profile or marker.get("profile_override") or marker.get("active_agent_profile") or self.config.get("active_agent_profile") or "default"
        session = RuntimeSession(
            session_key,
            self._new_state(marker),
            messages_to_history(rows),
            conversation_id,
            False,
            profile,
            profile_override=agent_profile or marker.get("profile_override"),
            is_subagent=is_subagent,
            subagent_meta=dict(marker.get("subagent_meta") or {}),
            system_prompt_extras={**dict(marker.get("system_prompt_extras") or {}), **dict(system_prompt_extras or {})},
        )
        # Re-seed cs with session-aware specs.
        session.cs = self._new_state(marker, session=session)
        with self._sessions_lock:
            self.sessions[session_key] = session
        bus.emit(SESSION_CREATED, {
            "session_key": session_key,
            "is_subagent": is_subagent,
            "agent_profile": profile,
        })
        self._restore_pending_requests(session)
        return session

    def load_history(self, session_key: str, conversation_id: int) -> RuntimeResult:
        session = self.load_conversation(session_key, conversation_id)
        return RuntimeResult(
            messages=[f"Loaded conversation: {self._conversation_title(conversation_id)}"],
            data={"conversation_id": conversation_id, "history": session.history},
        )

    def reset_conversation(self, session_key: str) -> RuntimeSession:
        with self._sessions_lock:
            existed = session_key in self.sessions
            session = RuntimeSession(session_key, self._new_state())
            session.cs = self._new_state(session=session)
            self.sessions[session_key] = session
        if existed:
            bus.emit(SESSION_CLOSED, {"session_key": session_key})
        bus.emit(SESSION_CREATED, {
            "session_key": session_key,
            "is_subagent": False,
            "agent_profile": session.active_agent_profile,
        })
        return session

    def iterate_agent_turn(
        self,
        session_key: str,
        prompt: str,
        *,
        image_paths: list[str] | None = None,
        actor_id: str = "user",
    ) -> RuntimeResult:
        payload = {"text": prompt, "actor_id": actor_id}
        if image_paths:
            payload["image_paths"] = image_paths
        out = self.handle_action(session_key, "send_text", payload)
        session = self.sessions.get(session_key)
        if out.ok and session and self.db and session.conversation_id:
            self.db.replace_conversation_messages(session.conversation_id, list(session.history))
            self._persist_marker(session)
        final_text = "\n".join(m for m in out.messages if m).strip()
        event = {
            "session_key": session_key,
            "conversation_id": session.conversation_id if session else None,
            "final_text": final_text,
            "new_messages": list(out.data.get("new_messages") or []),
            "attachments": list(out.attachments),
        }
        (self.emit_event or bus.emit)(SESSION_TURN_COMPLETED, event)
        out.data.update(event)
        return out

    def inject_user_message(
        self,
        session_key: str,
        text: str,
        *,
        conversation_id: int | None = None,
        actor_id: str = "user",
    ) -> RuntimeResult:
        """Append a user-authored message without driving the agent turn."""
        if conversation_id is not None:
            session = self.sessions.get(session_key)
            if session is None or session.conversation_id != conversation_id:
                session = self.load_conversation(session_key, conversation_id)
        else:
            session = self.get_session(session_key)
            self._ensure_conversation(session, text)
        msg = {"role": "user", "content": text}
        with session.lock:
            session.history.append(msg)
            if self.db and session.conversation_id:
                save_history_message(self.db, session.conversation_id, msg)
            bus.emit(SESSION_MESSAGE, {
                "session_key": session.key,
                "role": "user",
                "content": text,
                "actor_id": actor_id,
            })
            self._persist_marker(session)
        return RuntimeResult(data={"conversation_id": session.conversation_id})

    def _restore_pending_requests(self, session: RuntimeSession) -> None:
        """Re-emit `approval_requested` events for any phase frames that were
        mid-flight when the session was last persisted, so frontend adapters
        can re-register them in their pending-request tables and re-prompt
        the user.
        """
        if not self.emit_event:
            return
        frames = session.cs.cache.get("phases", []) if isinstance(session.cs.cache, dict) else []
        for frame in frames:
            if getattr(frame, "phase", None) != PHASE_APPROVING_REQUEST:
                continue
            data = getattr(frame, "data", {}) or {}
            req = StateMachineApprovalRequest(
                title=data.get("title") or frame.name or "Input required",
                body=data.get("prompt") or "",
                pending_action=data.get("pending"),
                id=data.get("request_id") or f"approve_{uuid.uuid4().hex}",
                type=data.get("type", "boolean"),
                enum=data.get("enum"),
                default=data.get("default"),
            )
            self.emit_event("approval_requested", req)

    def new_conversation(self, session_key: str) -> RuntimeResult:
        self.reset_conversation(session_key)
        return RuntimeResult(messages=[f"New conversation started. Agent: {self.config.get('active_agent_profile') or 'default'}."])

    @staticmethod
    def subagent_session_key(job_name: str) -> str:
        return f"subagent:{job_name}"

    # ──────────────────────────────────────────────────────────────────────
    # Plugin-facing API.
    #
    # Tools, tasks, and services can reach the runtime via
    # ``context.runtime`` (built in ``runtime/context.py``). The methods
    # below are the supported surface for *interacting* with sessions —
    # producing messages, mutating the agent's profile or system prompt,
    # registering session-pinned tools, and inspecting state.
    #
    # Anything not listed here is internal and may change without notice.
    # ──────────────────────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return a snapshot of every live session as a dict.

        Includes ``key``, ``is_subagent``, ``agent_profile``, ``phase``,
        ``turn_priority``, ``conversation_id``, ``busy``, and the keys of
        any system-prompt extras pinned to the session.
        """
        out = []
        for key, s in list(self.sessions.items()):
            out.append({
                "key": key,
                "is_subagent": s.is_subagent,
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
        """Surface a message in a session (typically the foreground one).

        Plugins should prefer this over emitting CHAT_MESSAGE_PUSHED directly
        — the runtime forwards through the same channel but stamps the
        session_key, which lets multi-session frontends route correctly.
        """
        from events.event_channels import CHAT_MESSAGE_PUSHED
        payload = {"message": text, "session_key": session_key}
        if title:
            payload["title"] = title
        if source:
            payload["source"] = source
        if source_id:
            payload["source_id"] = source_id
        bus.emit(CHAT_MESSAGE_PUSHED, payload)

    def set_agent_profile(self, session_key: str, profile: str) -> bool:
        """Pin a different agent profile to an existing session.

        Returns False if the session doesn't exist. The change is reflected
        on the next ``handle_action`` call (registry/scope/system-prompt all
        reread the override).
        """
        session = self.sessions.get(session_key)
        if session is None:
            return False
        old = session.profile_override or session.active_agent_profile
        session.profile_override = profile
        session.active_agent_profile = profile
        self._refresh_session_specs(session)
        bus.emit(SESSION_AGENT_PROFILE_CHANGED, {
            "session_key": session_key, "old_profile": old, "new_profile": profile,
        })
        return True

    def add_system_prompt_extra(self, session_key: str, key: str, value: str) -> bool:
        """Add or replace a named system-prompt addendum on a session.

        Whatever string the plugin stores under ``key`` is appended to the
        session's system prompt on every turn. Removal: pass ``None`` for
        ``value`` or call ``remove_system_prompt_extra``.

        Use cases: a service injecting current-state context (active doc, on-
        call status); a tool pinning a remember-this note; a task running on
        a schedule pinning its mode.
        """
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
        """Pin an extra tool instance to a session.

        The tool is layered on top of the session's scoped registry, so the
        agent on this session can call it but no other session sees it.
        Useful for ephemeral, session-specific tools (e.g. a NotifyTool
        bound to a single scheduled run).
        """
        session = self.sessions.get(session_key)
        if session is None:
            return False
        # Replace any existing tool with the same name.
        session.extra_tool_instances = [
            t for t in session.extra_tool_instances if getattr(t, "name", None) != getattr(tool_instance, "name", None)
        ]
        session.extra_tool_instances.append(tool_instance)
        self._refresh_session_specs(session)
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
            self._refresh_session_specs(session)
            return True
        return False

    def cancel_session(self, session_key: str) -> RuntimeResult | None:
        """Drive a ``cancel`` action on the session if one is live."""
        if session_key not in self.sessions:
            return None
        return self.handle_action(session_key, "cancel")

    def close_session(self, session_key: str) -> bool:
        """Discard a session entirely. Persistence is unaffected."""
        with self._sessions_lock:
            existed = self.sessions.pop(session_key, None) is not None
        if existed:
            bus.emit(SESSION_CLOSED, {"session_key": session_key})
        return existed

    def unload_conversation(self, session_key: str) -> bool:
        """Alias for plugin code that reads in conversation lifecycle terms."""
        return self.close_session(session_key)

    def request_approval(self, session_key: str, title: str, body: str, pending_action: dict[str, Any]) -> StateMachineApprovalRequest:
        """Boolean-approval gate. Thin wrapper around `request_input`."""
        return self.request_input(session_key, title, body, type="boolean", pending_action=pending_action)

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
        """Request typed user input (string, integer, number, boolean, array, object, enum).

        The phase frame stores everything needed to rebuild the in-memory
        `StateMachineApprovalRequest` after a restart (see `load_history`).
        """
        session = self.get_session(session_key)
        with session.lock:
            req = StateMachineApprovalRequest(
                title=title, body=prompt, pending_action=pending_action,
                type=type, enum=enum, default=default,
            )
            session.cs.push_phase(PhaseFrame(
                PHASE_APPROVING_REQUEST, "answer_approval", "user", title,
                {
                    "request_id": req.id,
                    "type": type,
                    "enum": enum,
                    "default": default,
                    "required": required,
                    "title": title,
                    "prompt": prompt,
                    "pending": pending_action,
                },
            ))
            session.cs.set_priority("user")
            if self.emit_event:
                self.emit_event("approval_requested", req)
            self._persist_marker(session)
            return req

    def _loop(self, session_key: str | None = None) -> ConversationLoop:
        session = self.sessions.get(session_key) if session_key else None
        llm = self._active_llm(session)
        if llm is None and hasattr(self, "llm"):
            llm = self.llm
        if llm is None or not getattr(llm, "loaded", True):
            raise RuntimeError("LLM service is not loaded.")
        def notice(text: str):
            if self.on_notice:
                self.on_notice(text)
            if session_key:
                bus.emit(CHAT_MESSAGE_PUSHED, {"session_key": session_key, "message": text, "source": "runtime", "kind": "alert"})

        return ConversationLoop(
            llm, self._active_tool_registry(session), self.config,
            self._session_system_prompt(session),
            *self._tool_callbacks(session_key), notice, session.cancel_event if session else None,
        )

    def _session_system_prompt(self, session: RuntimeSession | None):
        """Return a system_prompt callable bound to this session.

        For subagent sessions we route through ``build_system_prompt``
        directly so the scoped registry, profile name, and notification mode
        feed into the prompt. For user sessions we wrap the global callable
        and append the session's ``system_prompt_extras`` — letting any
        plugin pin contextual snippets to the prompt without touching the
        bootstrap closure.
        """
        if session is None:
            return self.system_prompt

        if session.is_subagent:
            from agent.system_prompt import build_system_prompt
            profile = session.profile_override or session.active_agent_profile or "default"
            scope = self._scope_for_profile(profile)
            registry = self._active_tool_registry(session)
            extras = session.system_prompt_extras or {}

            def _subagent_prompt():
                text = build_system_prompt(
                    self.db, getattr(self, "_orchestrator_ref", None) or self.services.get("orchestrator"),
                    registry, self.services,
                    scope=scope,
                    profile_name=profile,
                    subagent_mode=extras.get("subagent_mode"),
                    subagent_has_pending_messages=bool(extras.get("subagent_has_pending_messages")),
                )
                # Plugin-supplied addenda (anything that isn't a known
                # subagent-mode key) gets appended verbatim.
                for key, value in extras.items():
                    if key in {"subagent_mode", "subagent_has_pending_messages"}:
                        continue
                    if isinstance(value, str) and value:
                        text += "\n\n" + value
                return text
            return _subagent_prompt

        base = self.system_prompt
        extras = session.system_prompt_extras

        def _user_prompt():
            text = base() if callable(base) else (base or "")
            for value in (extras or {}).values():
                if isinstance(value, str) and value:
                    text += "\n\n" + value
            return text
        return _user_prompt

    def _tool_callbacks(self, session_key: str | None):
        def started(name, call_id="tc_unknown", args=None):
            if self.on_tool_start:
                self.on_tool_start(name)
            if self.emit_event:
                self.emit_event(TOOL_CALL_STARTED, {"session_key": session_key, "call_id": call_id, "tool_name": name, "args": args or {}})

        def finished(name, call_id="tc_unknown", result=None, error=None):
            tool_result = (getattr(result, "data", None) or {}).get("result") if result else None
            ok = bool(result and getattr(result, "ok", False) and getattr(tool_result, "success", True) and not error)
            err = error or getattr(getattr(result, "error", None), "message", None) or getattr(tool_result, "error", None)
            if self.on_tool_result:
                self.on_tool_result(name, tool_result)
            if self.emit_event:
                self.emit_event(TOOL_CALL_FINISHED, {"session_key": session_key, "call_id": call_id, "tool_name": name, "ok": ok, "error": err})

        return started, finished

    def _new_state(self, marker: dict[str, Any] | None = None, session: RuntimeSession | None = None) -> ConversationState:
        commands = dict(self.commands)
        tools = self._tool_specs(session)
        cache = dict((marker or {}).get("cache") or {})
        if session:
            cache["session_key"] = session.key
        phase = (marker or {}).get("phase", BASE_PHASE)
        return ConversationState(
            [Participant("user", "user", commands=commands), Participant("agent", "agent", tools=tools)],
            (marker or {}).get("turn_priority", "user"),
            phase,
            cache,
            attachment_parser=self._parse_attachment,
        )

    def _tool_specs(self, session: RuntimeSession | None = None) -> dict[str, CallableSpec]:
        # Expose direct tool calls as callable specs for `/call`-style flows.
        # ConversationLoop still uses the registry schemas directly when
        # marshalling the agent's tool calls.
        registry = self._active_tool_registry(session)
        if not registry:
            return {}
        specs = {}
        for schema in registry.get_all_schemas() or []:
            fn = schema.get("function", schema)
            name = fn.get("name")
            if name:
                specs[name] = CallableSpec(
                    name,
                    lambda _cs, _actor, args, n=name, reg=registry: reg.call(n, **args),
                    schema_to_form_steps(fn.get("parameters")),
                )
        return specs

    def refresh_session_specs(self) -> None:
        for session in list(self.sessions.values()):
            self._refresh_session_specs(session)

    def _refresh_session_specs(self, session: RuntimeSession) -> None:
        if not session.is_subagent:
            session.active_agent_profile = self.config.get("active_agent_profile") or "default"
        session.cs.participants["user"].commands = dict(self.commands)
        session.cs.participants["agent"].tools = self._tool_specs(session)

    def _profile_for_session(self, session: RuntimeSession | None) -> str:
        if session is not None and session.profile_override:
            return session.profile_override
        return self.config.get("active_agent_profile") or "default"

    def _scope_for_profile(self, profile: str):
        try:
            from runtime.agent_scope import load_scope
            scope = load_scope(profile, self.config)
        except ValueError:
            return None
        return scope if scope.has_tool_filter or scope.prompt_suffix else None

    def _active_scope(self, session: RuntimeSession | None = None):
        return self._scope_for_profile(self._profile_for_session(session))

    def _active_tool_registry(self, session: RuntimeSession | None = None):
        if not self.tool_registry:
            return None
        from runtime.agent_scope import scoped_registry
        scope = self._active_scope(session)
        registry = self.tool_registry
        if scope:
            registry = scoped_registry(self.tool_registry, scope, db=self.db)
        # Subagent sessions add their pinned tool instances (NotifyTool etc.)
        # on top of the scoped view so the agent in this session can call them.
        extras = list((session.extra_tool_instances if session else []) or [])
        if extras:
            from agent.tool_registry import ToolRegistry
            cloned = ToolRegistry(registry.db, registry.config, registry.services)
            cloned.orchestrator = getattr(registry, "orchestrator", None)
            cloned.is_subagent = bool(session and session.is_subagent) or getattr(registry, "is_subagent", False)
            cloned.runtime = getattr(registry, "runtime", None)
            cloned.tools.update(registry.tools)
            for tool in extras:
                cloned.tools[tool.name] = tool
            registry = cloned
        return registry

    def _active_llm(self, session: RuntimeSession | None = None):
        profile = self._profile_for_session(session)
        try:
            from runtime.agent_scope import resolve_agent_llm
            return resolve_agent_llm(profile, self.config, self.services)
        except Exception:
            return self.services.get("llm")

    def _parse_attachment(self, content: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(content.get("path") or ""))
        file_name = content.get("file_name") or path.name or "attachment"
        caption = str(content.get("caption") or content.get("text") or "").strip()
        suffix = path.suffix or ("." + str(content.get("extension") or "").lstrip(".") if content.get("extension") else "")
        try:
            from plugins.services.helpers.parser_registry import get_modality
            modality = get_modality(suffix) if suffix else "unknown"
        except Exception:
            modality = "unknown"
        text, image_paths = caption, list(content.get("image_paths") or [])
        if modality == "image" or content.get("is_photo"):
            has_vision = self.services.get("llm") and getattr(self.services["llm"], "vision", True) is not False
            image_paths = [str(path)] if has_vision and path else image_paths
            text = (text + f"\n\n[The user attached an image: {file_name} (cached at {path})]").strip()
        elif modality in {"text", "tabular", "unknown"} and self.services.get("parser"):
            try:
                parsed = self.services["parser"].parse(str(path), config={"max_chars": 4000}).output
                raw = parsed if isinstance(parsed, str) else str(parsed)
                text = (text + f"\n\n[The user attached a file: {file_name} (cached at {path})]\n{raw[:4000]}").strip()
            except Exception:
                text = (text + f"\n\n[The user attached a file: {file_name} (cached at {path}). Use read_file or search tools to access it.]").strip()
        else:
            text = (text + f"\n\n[The user attached a file: {file_name} (cached at {path}). Use read_file or search tools to access it.]").strip()
        return {**content, "text": text, "image_paths": image_paths}

    def _command_specs(self, specs: dict[str, dict]) -> dict[str, CallableSpec]:
        out = {}
        for name, spec in specs.items():
            out[name] = CallableSpec(
                name,
                spec.get("handler"),
                schema_to_form_steps(spec.get("parameters")),
                spec.get("require_approval", False),
                spec.get("approval_actor_id"),
                spec.get("validator"),
            )
        return out

    def _ensure_conversation(self, session: RuntimeSession, title_text: str = "") -> None:
        if session.conversation_id is None and self.db:
            session.conversation_id = self.db.create_conversation(
                (title_text or "New conversation").replace("\n", " ")[:80] or "New conversation",
                kind="subagent" if session.is_subagent else "user",
            )

    def _persist_marker(self, session: RuntimeSession) -> None:
        if self.db and session.conversation_id:
            save_state_marker(self.db, session.conversation_id, session.to_marker())

    def _conversation_title(self, conversation_id: int) -> str:
        row = self.db.get_conversation(conversation_id) if self.db else None
        return ((row or {}).get("title") or "").strip() or "New conversation"

    def _decorate_form(self, session: RuntimeSession, out: RuntimeResult) -> None:
        frame = session.cs.frame
        if frame and frame.step:
            out.form = {
                "name": frame.name,
                "field": frame.step.to_dict(),
                "collected": frame.data.get("args", {}),
            }

    def _latest_user_text(self, session: RuntimeSession) -> str:
        for msg in reversed(session.history):
            if msg.get("role") == "user":
                return msg.get("content") or ""
        return ""

    @staticmethod
    def _text(payload: dict | str | None) -> str:
        return payload if isinstance(payload, str) else str((payload or {}).get("text") or "")

    @staticmethod
    def _image_paths(payload: dict | str | None) -> list[str]:
        return list((payload or {}).get("image_paths") or []) if isinstance(payload, dict) else []

    @staticmethod
    def _actor_id(payload: dict | str | None) -> str | None:
        return (payload or {}).get("actor_id") if isinstance(payload, dict) else None
