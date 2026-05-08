"""
Frontend interface.

Frontends are the user-facing transports of Second Brain (REPL, Telegram, HTTP,
future GUIs). They are first-class plugins, like tools/tasks/services: each
subclass declares its identity and capabilities, and implements two halves of
the contract:

    1. Turn user input into an Action and submit it to ConversationRuntime.
    2. Render the resulting RuntimeResult (and bus-borne events from other
       sessions) back to the user.

Everything else — slash-command parsing, form-step prompting, state-machine
request rendering, session bookkeeping — lives in the base.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from events.event_bus import bus
from events.event_channels import (
    APPROVAL_REQUESTED,
    CHAT_MESSAGE_PUSHED,
    COMMAND_CALL_FINISHED,
    COMMAND_CALL_PROGRESSED,
    COMMAND_CALL_STARTED,
    TASKS_CHANGED,
    TOOL_CALL_FINISHED,
    TOOL_CALL_STARTED,
    TOOLS_CHANGED,
)
from state_machine.action_map import (
    ACTION_ANSWER_APPROVAL,
    ACTION_CALL_COMMAND,
    ACTION_CANCEL,
    ACTION_SEND_ATTACHMENT,
    ACTION_SEND_TEXT,
    ACTION_SKIP_FORM,
    ACTION_SUBMIT_FORM_TEXT,
    legal_actions_in_phase,
)
from state_machine.conversation_phases import (
    FORM_PHASES,
    PHASE_APPROVING_REQUEST,
)
from state_machine.approval import StateMachineApprovalRequest

logger = logging.getLogger("Frontend")


def _form_step_accepts(step, text: str) -> bool:
    """Would ``text`` be a valid value for this form step?

    Used to decide whether typed input should fill the form or abort the
    form and become a chat message. Empty text is treated as "no" so the
    explicit /skip path stays in charge of optional fields.
    """
    if not text:
        return False
    try:
        ok, _ = step.validate(text)
    except Exception:
        return False
    return bool(ok)


@dataclass
class FrontendCapabilities:
    """What a frontend transport can do.

    Renderers may consult these to choose between rich and plaintext output
    (e.g. inline buttons vs. a numbered enum prompt). They are not enforced
    by the base — a subclass that lies here will just produce a worse UX.
    """

    supports_typing: bool = False
    supports_buttons: bool = False
    supports_message_edit: bool = False
    supports_attachments_in: bool = False
    supports_attachments_out: bool = False
    supports_inline_forms: bool = False
    supports_proactive_push: bool = False
    supports_rich_text: bool = False
    max_message_chars: int | None = None
    max_upload_size: int | None = None


class BaseFrontend:
    """
    The contract every frontend implements.

    Class attributes (override these):
        name:
            Stable identifier — "repl", "telegram", "http", ...
        description:
            Short operational description for /commands-style listings.
        capabilities:
            FrontendCapabilities describing the transport.
        config_settings:
            Same tuple format as SETTINGS_DATA. See plugins.BaseTool.

    Lifecycle (override):
        start()                     — begin the transport's main loop.
        stop()                      — shut down cleanly.
        session_key(ctx)            — derive a session key from a transport
                                       context (REPL: "default";
                                       Telegram: f"{user}:{chat}:{thread}").

    Rendering (override — all abstract):
        render_messages(session_key, messages)
        render_attachments(session_key, paths)
        render_form_field(session_key, form)
        render_approval_request(session_key, req)
        render_buttons(session_key, buttons)
        render_error(session_key, error)
        render_typing(session_key, on)            — default no-op.
        render_tool_status(session_key, payload)  — default no-op.

    Provided (do NOT override):
        bind(runtime, registry, config)
        submit(session_key, action_type, payload=None) -> RuntimeResult
        submit_text(session_key, text)
        submit_attachment(session_key, path)
        cancel(session_key)
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    capabilities: FrontendCapabilities = FrontendCapabilities()

    # --- Config settings this plugin needs ---
    # Each entry is a tuple:
    # (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if isinstance(cls.config_settings, list):
            cls.config_settings = list(cls.config_settings)

    def __init__(self):
        self.runtime = None
        self.commands = None     # CommandRegistry, set in bind()
        self.config: dict = {}
        self._unsubs: list = []
        self._bound = False
        self._approval_lock = threading.RLock()
        self._pending_approvals: dict[str, dict[str, object]] = {}
        self._pending_approval_order: dict[str, list[str]] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle — override these.
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        raise NotImplementedError(f"Frontend '{self.name}' must implement start()")

    def stop(self) -> None:
        raise NotImplementedError(f"Frontend '{self.name}' must implement stop()")

    def session_key(self, ctx) -> str:
        raise NotImplementedError(f"Frontend '{self.name}' must implement session_key()")

    # ──────────────────────────────────────────────────────────────────────
    # Rendering — override these. The base owns *when* to render; subclasses
    # own *how*.
    # ──────────────────────────────────────────────────────────────────────

    def render_messages(self, session_key: str, messages: list[str]) -> None:
        raise NotImplementedError

    def render_attachments(self, session_key: str, paths: list[str]) -> None:
        raise NotImplementedError

    def render_form_field(self, session_key: str, form: dict) -> None:
        """Render a form prompt.

        ``form`` shape (from state_machine/runtime.py:_decorate_form):
            {
                "name":      str,        # command/tool name
                "field":     dict,       # FormStep.to_dict() — name, prompt,
                                         # required, type, enum, default, ...
                "collected": dict,       # args gathered so far
                "display":   dict,       # frontend-neutral prompt, assist,
                                         # choices, skip/cancel affordances
            }
        """
        raise NotImplementedError

    def render_approval_request(self, session_key: str, req) -> None:
        """Render a typed-input/approval request.

        ``req`` is a StateMachineApprovalRequest with at least:
        ``id``, ``title``, ``body``, ``type``, ``enum``, ``default``.
        """
        raise NotImplementedError

    def render_buttons(self, session_key: str, buttons: list[dict]) -> None:
        raise NotImplementedError

    def render_error(self, session_key: str, error: dict) -> None:
        raise NotImplementedError

    def render_typing(self, session_key: str, on: bool) -> None:
        """Default no-op; rich frontends override to show a typing indicator."""
        return

    def render_tool_status(self, session_key: str, payload: dict) -> None:
        """Default no-op; frontends with status affordances override."""
        return

    # ──────────────────────────────────────────────────────────────────────
    # Wiring — provided by the base.
    # ──────────────────────────────────────────────────────────────────────

    def bind(self, runtime, commands, config: dict | None = None) -> None:
        """Attach to runtime + command registry and subscribe to bus channels.

        ``runtime``  — ConversationRuntime instance (the only state-machine
                       entry point a frontend uses).
        ``commands`` — CommandRegistry built from the project's command
                       plugins. Used for /-completions and to validate command
                       names before submitting actions.
        ``config``   — merged app config dict (read-only from a frontend's
                       perspective; mutate through a command or tool).
        """
        if self._bound:
            return
        self.runtime = runtime
        self.commands = commands
        self.config = config or {}
        self._unsubs = [
            bus.subscribe(APPROVAL_REQUESTED, self.on_bus_approval_requested),
            bus.subscribe(CHAT_MESSAGE_PUSHED, self.on_bus_message_pushed),
            bus.subscribe(COMMAND_CALL_STARTED, self.on_bus_command_call_started),
            bus.subscribe(COMMAND_CALL_PROGRESSED, self.on_bus_command_call_progressed),
            bus.subscribe(COMMAND_CALL_FINISHED, self.on_bus_command_call_finished),
            bus.subscribe(TOOL_CALL_STARTED, self.on_bus_tool_call_started),
            bus.subscribe(TOOL_CALL_FINISHED, self.on_bus_tool_call_finished),
            bus.subscribe(TOOLS_CHANGED, self.on_tools_changed),
            bus.subscribe(TASKS_CHANGED, self.on_tasks_changed),
        ]
        self._bound = True

    def unbind(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                logger.exception(f"Frontend '{self.name}' bus unsubscribe failed")
        self._unsubs.clear()
        self._bound = False

    # ──────────────────────────────────────────────────────────────────────
    # The single submission path. Every action a frontend performs ends up
    # calling submit(); submit() always renders the result.
    # ──────────────────────────────────────────────────────────────────────

    def submit(self, session_key: str, action_type: str, payload=None):
        if self.runtime is None:
            raise RuntimeError(
                f"Frontend '{self.name}' is not bound — call bind(runtime, ...) first."
            )
        result = self.runtime.handle_action(session_key, action_type, payload)
        self._render_result(session_key, result)
        return result

    def submit_text(self, session_key: str, text: str):
        """Coerce raw user text into the right action for the current phase.

        - In a form phase, ``/cancel`` becomes ``cancel``, blank text becomes
          ``skip_form`` (when the field is optional), anything else becomes
          ``submit_form_text``.
        - In an approval phase, text becomes ``answer_approval``.
        - Otherwise ``/foo args`` becomes ``call_command`` and plain text
          becomes ``send_text``.
        """
        phase = self._current_phase(session_key)
        legal = set(legal_actions_in_phase(phase))

        if phase in FORM_PHASES:
            stripped = (text or "").strip()
            if stripped == "/cancel" and ACTION_CANCEL in legal:
                return self.submit(session_key, ACTION_CANCEL)
            if stripped == "/skip" and ACTION_SKIP_FORM in legal:
                return self.submit(session_key, ACTION_SKIP_FORM)
            if stripped.startswith("/") and ACTION_CALL_COMMAND in legal:
                name, _, arg = stripped[1:].partition(" ")
                cmd = next((c for c in self.commands.all_commands() if c.name == name), None) if name and self.commands else None
                if cmd:
                    args = self.commands.parse_args(name, arg, session_key=session_key) if arg.strip() else {}
                    return self.submit(session_key, ACTION_CALL_COMMAND, {"name": name, "args": args})
            if not stripped and ACTION_SKIP_FORM in legal:
                return self.submit(session_key, ACTION_SKIP_FORM)
            # If the typed text doesn't fit the current form step (e.g. user
            # abandoned a half-finished command and started typing a chat
            # message), bail out of the form and dispatch as a regular
            # send_text. REPL-style form filling still works because valid
            # text falls through to ACTION_SUBMIT_FORM_TEXT below.
            step = self._current_form_step(session_key)
            if step is not None and not _form_step_accepts(step, stripped):
                self.cancel(session_key)
                return self.submit(session_key, ACTION_SEND_TEXT, text)
            return self.submit(session_key, ACTION_SUBMIT_FORM_TEXT, stripped)

        if phase == PHASE_APPROVING_REQUEST:
            result = self.submit(session_key, ACTION_ANSWER_APPROVAL, text)
            self._clear_pending_approval(session_key)
            return result

        stripped = (text or "").lstrip()
        if stripped == "/cancel":
            return self.cancel(session_key)
        if stripped.startswith("/"):
            name, _, arg = stripped[1:].partition(" ")
            cmd = next((c for c in self.commands.all_commands() if c.name == name), None) if name and self.commands else None
            if cmd:
                args = self.commands.parse_args(name, arg, session_key=session_key) if arg.strip() else {}
                return self.submit(
                    session_key,
                    ACTION_CALL_COMMAND,
                    {"name": name, "args": args},
                )
        return self.submit(session_key, ACTION_SEND_TEXT, text)

    def submit_attachment(self, session_key: str, path: str, extension: str | None = None):
        from pathlib import Path
        ext = extension or Path(path).suffix.lstrip(".")
        return self.submit(
            session_key,
            ACTION_SEND_ATTACHMENT,
            {"path": path, "extension": ext},
        )

    def cancel(self, session_key: str):
        return self.submit(session_key, ACTION_CANCEL)

    # ──────────────────────────────────────────────────────────────────────
    # Bus handlers. Subclasses can override for richer behavior, but the
    # defaults route everything through the abstract render_* methods.
    # ──────────────────────────────────────────────────────────────────────

    def on_bus_approval_requested(self, req) -> None:
        target = ((getattr(req, "metadata", None) or {}).get("session_key"))
        live = self._live_session_keys()
        keys = [target] if target in live else live
        for key in keys:
            try:
                with self._approval_lock:
                    self._pending_approvals.setdefault(key, {})[req.id] = req
                    self._pending_approval_order.setdefault(key, []).append(req.id)
                self.render_approval_request(key, req)
            except Exception:
                logger.exception(f"render_approval_request failed for '{self.name}'")

    def resolve_approval(self, session_key: str, request_id: str, value, resolved_by: str | None = None) -> bool:
        with self._approval_lock:
            req = self._pending_approvals.get(session_key, {}).get(request_id)
            if req is None:
                return False
            if resolved_by and hasattr(req, "metadata"):
                req.metadata["resolved_by"] = resolved_by
            target = (getattr(req, "metadata", {}) or {}).get("session_key") or session_key
        result = self.runtime.handle_action(target, ACTION_ANSWER_APPROVAL, {"value": value, "request_id": request_id})
        self._clear_pending_approval(session_key, request_id)
        return bool(result and result.ok)

    def resolve_next_approval(self, session_key: str, value, resolved_by: str | None = None) -> bool:
        with self._approval_lock:
            order = self._pending_approval_order.setdefault(session_key, [])
            pending = self._pending_approvals.setdefault(session_key, {})
            while order and (order[0] not in pending or getattr(pending[order[0]], "is_resolved", False)):
                order.pop(0)
            return bool(order) and self.resolve_approval(session_key, order.pop(0), value, resolved_by)

    def has_pending_approval(self, session_key: str) -> bool:
        with self._approval_lock:
            return any(not getattr(req, "is_resolved", False) for req in self._pending_approvals.get(session_key, {}).values())

    def _clear_pending_approval(self, session_key: str, request_id: str | None = None) -> None:
        with self._approval_lock:
            if request_id is None:
                self._pending_approvals.pop(session_key, None)
                self._pending_approval_order.pop(session_key, None)
                return
            self._pending_approvals.get(session_key, {}).pop(request_id, None)
            self._pending_approval_order[session_key] = [item for item in self._pending_approval_order.get(session_key, []) if item != request_id]

    def on_bus_message_pushed(self, payload: dict) -> None:
        message = (payload or {}).get("message")
        if not message:
            return
        title = (payload or {}).get("title")
        body = f"{title}\n\n{message}" if title else message
        target = (payload or {}).get("session_key")
        keys = [target] if target else self._live_session_keys()
        for key in keys:
            if key not in self._live_session_keys():
                continue
            try:
                self.render_messages(key, [body])
            except Exception:
                logger.exception(f"render_messages (push) failed for '{self.name}'")

    def on_bus_tool_call_started(self, payload: dict) -> None:
        self._render_tool_status_event({**(payload or {}), "status": "started"})

    def on_bus_tool_call_finished(self, payload: dict) -> None:
        self._render_tool_status_event({**(payload or {}), "status": "finished"})

    def on_bus_command_call_started(self, payload: dict) -> None:
        self._render_tool_status_event({**(payload or {}), "status": "started", "kind": "command"})

    def on_bus_command_call_progressed(self, payload: dict) -> None:
        self._render_tool_status_event({**(payload or {}), "status": "progressed", "kind": "command"})

    def on_bus_command_call_finished(self, payload: dict) -> None:
        self._render_tool_status_event({**(payload or {}), "status": "finished", "kind": "command"})

    def on_tools_changed(self, _payload) -> None:
        return

    def on_tasks_changed(self, _payload) -> None:
        return

    # ──────────────────────────────────────────────────────────────────────
    # Internals.
    # ──────────────────────────────────────────────────────────────────────

    def _render_result(self, session_key: str, result) -> None:
        if result is None:
            return
        if result.messages:
            self.render_messages(session_key, list(result.messages))
        if result.attachments:
            self.render_attachments(session_key, list(result.attachments))
        if result.form:
            self.render_form_field(session_key, dict(result.form))
        if result.buttons:
            self.render_buttons(session_key, list(result.buttons))
        if result.error:
            self.render_error(session_key, dict(result.error))
        req = self._current_approval_request(session_key)
        if req:
            self.render_approval_request(session_key, req)

    def _current_phase(self, session_key: str) -> str:
        session = self.runtime.get_session(session_key)
        return session.cs.phase

    def _current_form_step(self, session_key: str):
        frame = self.runtime.get_session(session_key).cs.frame
        return getattr(frame, "step", None) if frame else None

    def _current_approval_request(self, session_key: str):
        if self._current_phase(session_key) != PHASE_APPROVING_REQUEST:
            return None
        frame = self.runtime.get_session(session_key).cs.frame
        data = getattr(frame, "data", {}) or {}
        return StateMachineApprovalRequest(
            title=data.get("title") or frame.name or "Input required",
            body=data.get("prompt") or "",
            pending_action=data.get("pending"),
            id=data.get("request_id") or "pending",
            type=data.get("type", "boolean"),
            enum=data.get("enum"),
            default=data.get("default"),
            metadata={"session_key": session_key},
        )

    def _live_session_keys(self) -> list[str]:
        """Session keys this frontend currently has open.

        Default: every session the runtime knows about. Subclasses that
        multiplex multiple platforms behind one runtime should override to
        scope this to their own sessions.
        """
        if self.runtime is None:
            return []
        return list(self.runtime.sessions.keys())

    def _render_tool_status_event(self, payload: dict) -> None:
        key = (payload or {}).get("session_key")
        if not key or key not in self._live_session_keys():
            return
        try:
            self.render_tool_status(key, payload)
        except Exception:
            logger.exception(f"render_tool_status failed for '{self.name}'")
