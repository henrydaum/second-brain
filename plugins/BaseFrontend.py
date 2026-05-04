"""
Frontend interface.

Frontends are the user-facing transports of Second Brain (REPL, Telegram, HTTP,
future GUIs). They are first-class plugins, like tools/tasks/services: each
subclass declares its identity and capabilities, and implements two halves of
the contract:

    1. Turn user input into an Action and submit it to ConversationRuntime.
    2. Render the resulting RuntimeResult (and bus-borne events from other
       sessions) back to the user.

Everything else — slash-command parsing, form-step prompting, approval
bridging across the event bus, session bookkeeping — lives in the base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from events.event_bus import bus
from events.event_channels import (
    APPROVAL_REQUESTED,
    APPROVAL_RESOLVED,
    CHAT_MESSAGE_PUSHED,
    TASKS_CHANGED,
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

logger = logging.getLogger("Frontend")


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
            Short operational description for /help-style listings.
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

    # ──────────────────────────────────────────────────────────────────────
    # Wiring — provided by the base.
    # ──────────────────────────────────────────────────────────────────────

    def bind(self, runtime, commands, config: dict | None = None) -> None:
        """Attach to runtime + command registry and subscribe to bus channels.

        ``runtime``  — ConversationRuntime instance (the only state-machine
                       entry point a frontend uses).
        ``commands`` — CommandRegistry built from the project's command
                       plugins. Used for /-completions, /help rendering, and
                       to validate command names before submitting actions.
        ``config``   — merged app config dict (read-only from a frontend's
                       perspective; mutate via /configure).
        """
        if self._bound:
            return
        self.runtime = runtime
        self.commands = commands
        self.config = config or {}
        self._unsubs = [
            bus.subscribe(APPROVAL_REQUESTED, self.on_bus_approval_requested),
            bus.subscribe(APPROVAL_RESOLVED, self.on_bus_approval_resolved),
            bus.subscribe(CHAT_MESSAGE_PUSHED, self.on_bus_message_pushed),
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
            if not stripped and ACTION_SKIP_FORM in legal:
                return self.submit(session_key, ACTION_SKIP_FORM)
            return self.submit(session_key, ACTION_SUBMIT_FORM_TEXT, stripped)

        if phase == PHASE_APPROVING_REQUEST:
            return self.submit(session_key, ACTION_ANSWER_APPROVAL, text)

        stripped = (text or "").lstrip()
        if stripped.startswith("/"):
            name, _, arg = stripped[1:].partition(" ")
            if name and self.commands and name in {c.name for c in self.commands.all_commands()}:
                return self.submit(
                    session_key,
                    ACTION_CALL_COMMAND,
                    {"name": name, "args": {"arg": arg.strip()} if arg.strip() else {}},
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
        for key in self._live_session_keys():
            try:
                self.render_approval_request(key, req)
            except Exception:
                logger.exception(f"render_approval_request failed for '{self.name}'")

    def on_bus_approval_resolved(self, req) -> None:
        # Default: no-op. Frontends that draw lingering approval UI (Telegram
        # inline keyboards, etc.) should override to clear it.
        return

    def on_bus_message_pushed(self, payload: dict) -> None:
        message = (payload or {}).get("message")
        if not message:
            return
        title = (payload or {}).get("title")
        body = f"{title}\n\n{message}" if title else message
        for key in self._live_session_keys():
            try:
                self.render_messages(key, [body])
            except Exception:
                logger.exception(f"render_messages (push) failed for '{self.name}'")

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

    def _current_phase(self, session_key: str) -> str:
        session = self.runtime.get_session(session_key)
        return session.cs.phase

    def _live_session_keys(self) -> list[str]:
        """Session keys this frontend currently has open.

        Default: every session the runtime knows about. Subclasses that
        multiplex multiple platforms behind one runtime should override to
        scope this to their own sessions.
        """
        if self.runtime is None:
            return []
        return list(self.runtime.sessions.keys())
