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
from typing import Any, Callable

from state_machine.approval import StateMachineApprovalRequest
from state_machine.conversationClass import CallableSpec, ConversationState, FormStep, Participant, PhaseFrame
from state_machine.conversation_loop import ConversationLoop
from state_machine.conversation_phases import BASE_PHASE, PHASE_APPROVING_REQUEST
from state_machine.errors import ActionError, ActionResult
from events.event_channels import TOOL_CALL_FINISHED, TOOL_CALL_STARTED
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
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def to_marker(self) -> dict[str, Any]:
        state = self.cs.to_dict()
        state.update({
            "conversation_id": self.conversation_id,
            "active_agent_profile": self.active_agent_profile,
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
        with session.lock:
            try:
                self._refresh_session_specs(session)
                return self._dispatch(session, action_type, payload)
            finally:
                if action_type not in {"load_history", "new_conversation"}:
                    self._persist_marker(session)

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

        # ──────────────── THE enact() SITE (user-side) ────────────────
        result = session.cs.enact(action_type, content, actor_id)
        # ──────────────────────────────────────────────────────────────

        out.add_action_result(result)
        self._absorb_user_action(session, action_type, text, result)
        self._decorate_form(session, out)
        self._echo_callable_result(action_type, result, out)

        if (expects_agent_reply
                and result.ok
                and session.cs.turn_priority == "agent"
                and session.cs.phase == BASE_PHASE):
            self._drive_agent_turn(session, out, image_paths)

        return out

    # ──────────────────────────────────────────────────────────────────────
    # Driving the agent's turn. ConversationLoop has its OWN labeled
    # cs.enact() site inside it; this method just sets up persistence and
    # surface the loop's outputs.
    # ──────────────────────────────────────────────────────────────────────

    def _drive_agent_turn(self, session: RuntimeSession, out: RuntimeResult, image_paths: list[str] | None) -> RuntimeResult:
        self._ensure_conversation(session, self._latest_user_text(session))
        session.busy = True
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

        if (self.title_callback and session.conversation_id
                and any(m.get("role") == "assistant" and not m.get("tool_calls") for m in new_messages)):
            self.title_callback(session.conversation_id)

        if reply:
            out.messages.append(reply)
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

    # ──────────────────────────────────────────────────────────────────────
    # Session + persistence + setup
    # ──────────────────────────────────────────────────────────────────────

    def get_session(self, key: str) -> RuntimeSession:
        with self._sessions_lock:
            if key not in self.sessions:
                self.sessions[key] = RuntimeSession(key, self._new_state())
            return self.sessions[key]

    def load_history(self, session_key: str, conversation_id: int) -> RuntimeResult:
        rows = self.db.get_conversation_messages(conversation_id) if self.db else []
        marker = latest_state(rows) or {}
        session = RuntimeSession(
            session_key,
            self._new_state(marker),
            messages_to_history(rows),
            conversation_id,
            False,
            marker.get("active_agent_profile") or self.config.get("active_agent_profile") or "default",
        )
        with self._sessions_lock:
            self.sessions[session_key] = session
        self._restore_pending_requests(session)
        return RuntimeResult(
            messages=[f"Loaded conversation: {self._conversation_title(conversation_id)}"],
            data={"conversation_id": conversation_id, "history": session.history},
        )

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
        with self._sessions_lock:
            self.sessions[session_key] = RuntimeSession(session_key, self._new_state())
        return RuntimeResult(messages=[f"New conversation started. Agent: {self.config.get('active_agent_profile') or 'default'}."])

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
        llm = self._active_llm()
        if llm is None and hasattr(self, "llm"):
            llm = self.llm
        if llm is None or not getattr(llm, "loaded", True):
            raise RuntimeError("LLM service is not loaded.")
        return ConversationLoop(
            llm, self._active_tool_registry(), self.config, self.system_prompt,
            *self._tool_callbacks(session_key), self.on_notice,
        )

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

    def _new_state(self, marker: dict[str, Any] | None = None) -> ConversationState:
        commands = dict(self.commands)
        tools = self._tool_specs()
        cache = (marker or {}).get("cache")
        phase = (marker or {}).get("phase", BASE_PHASE)
        return ConversationState(
            [Participant("user", "user", commands=commands), Participant("agent", "agent", tools=tools)],
            (marker or {}).get("turn_priority", "user"),
            phase,
            cache,
        )

    def _tool_specs(self) -> dict[str, CallableSpec]:
        # Expose direct tool calls as callable specs for `/call`-style flows.
        # ConversationLoop still uses the registry schemas directly when
        # marshalling the agent's tool calls.
        registry = self._active_tool_registry()
        if not registry:
            return {}
        specs = {}
        for schema in registry.get_all_schemas() or []:
            fn = schema.get("function", schema)
            name = fn.get("name")
            if name:
                specs[name] = CallableSpec(
                    name,
                    lambda _cs, _actor, args, n=name: self._active_tool_registry().call(n, **args),
                    schema_to_form_steps(fn.get("parameters")),
                )
        return specs

    def refresh_session_specs(self) -> None:
        for session in list(self.sessions.values()):
            self._refresh_session_specs(session)

    def _refresh_session_specs(self, session: RuntimeSession) -> None:
        session.active_agent_profile = self.config.get("active_agent_profile") or "default"
        session.cs.participants["user"].commands = dict(self.commands)
        session.cs.participants["agent"].tools = self._tool_specs()

    def _active_scope(self):
        profile = self.config.get("active_agent_profile") or "default"
        try:
            from runtime.agent_scope import load_scope
            scope = load_scope(profile, self.config)
        except ValueError:
            return None
        return scope if scope.has_tool_filter or scope.prompt_suffix else None

    def _active_tool_registry(self):
        scope = self._active_scope()
        if not scope or not self.tool_registry:
            return self.tool_registry
        from runtime.agent_scope import scoped_registry
        return scoped_registry(self.tool_registry, scope, db=self.db)

    def _active_llm(self):
        profile = self.config.get("active_agent_profile") or "default"
        try:
            from runtime.agent_scope import resolve_agent_llm
            return resolve_agent_llm(profile, self.config, self.services)
        except Exception:
            return self.services.get("llm")

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
                (title_text or "New conversation").replace("\n", " ")[:80] or "New conversation"
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
