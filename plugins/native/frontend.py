"""The native face of a frontend adapter — and the host-side routing itself.

Nothing subclasses this by hand. A frontend is sandboxed code, and
``sandbox.bridge`` builds a subclass of this class at load. Unlike the other
four bases this one is not a thin contract: it *is* the routing, and that is
deliberate. The base owns **when** — fourteen bus subscriptions funnelling
into nine ``render_*`` methods, and ``submit_*`` funnelling into
``runtime.handle_action`` — while the guest owns **how**. So the bridge
overrides the nine renderers with one ``render(kind, payload)`` box call and
inherits everything else.

Frontends are the user-facing transports of Second Brain (REPL, installed
Telegram, HTTP, future presentation layers). Each declares its identity and
capabilities, and implements two halves of the contract:

    1. Turn user input into an Action and submit it to ConversationRuntime.
    2. Render the resulting RuntimeResult (and bus-borne events from other
       sessions) back to the user.

Everything else — slash-command parsing, form-step prompting, state-machine
request rendering, session bookkeeping — lives here. See
``templates/frontend_template.py`` for what an author actually writes.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass

from events.event_bus import bus
from events.event_channels import (
    AGENT_TEXT_DELTA,
    APPROVAL_REQUESTED,
    APPROVAL_SETTLED,
    CHAT_MESSAGE_PUSHED,
    COMMAND_CALL_FINISHED,
    COMMAND_CALL_PROGRESSED,
    COMMAND_CALL_STARTED,
    CONVERSATION_CHANGED,
    FORM_REQUESTED,
    NOTIFICATION_PUSHED,
    SESSION_CONVERSATION_CHANGED,
    SESSION_TURN_CHANGED,
    TASKS_CHANGED,
    TOOL_CALL_FINISHED,
    TOOL_CALL_STARTED,
    TOOLS_CHANGED,
)
from state_machine.action_map import (
    ACTION_ANSWER_APPROVAL,
    ACTION_BACK_FORM,
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
from runtime.session import RuntimeResult
from pipeline.database import DEFAULT_USER_ID

logger = logging.getLogger("Frontend")

# User-binding styles — how a frontend maps its sessions to users (the "whose
# data" axis; orthogonal to authorization, which is owned by frontend_profile).
USER_BINDING_SINGLE = "single"      # every session acts as one user (default_user_id)
USER_BINDING_PER_USER = "per_user"  # each external identity gets its own user
_USER_BINDINGS = (USER_BINDING_SINGLE, USER_BINDING_PER_USER)


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


def _approval_body(text: str, limit: int = 900) -> str:
    """Trim an over-long approval body without losing its ending.

    Head-trimming cut the tail, and the tail is where the body puts what a
    person most needs: who asked, and why they are being asked. A
    scheduled-subagent dialog carries a prompt preview long enough to trigger
    this every time, so the line explaining the dialog was reliably replaced
    by a note saying something had been trimmed.

    The middle goes instead: the first paragraph is the act and the last two
    are the context, and what sits between them is argument detail — the part
    that gets long, and the part a summary survives.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    tail = "\n\n".join(text.split("\n\n")[-2:])[:limit // 3]
    head = text[:max(0, limit - len(tail) - 40)].rstrip()
    return f"{head}\n\n...trimmed...\n\n{tail}"


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
    # Frontend renders AGENT_TEXT_DELTA events incrementally (and implements
    # render_stream_delta). False = ignore the channel; the same text arrives
    # as whole messages exactly as before. Kept LAST so existing positional
    # FrontendCapabilities(...) constructions keep their meaning.
    supports_streaming: bool = False
    # Frontend draws notifications on a surface of its own (and implements
    # render_notification). False = the base formats each one into markdown and
    # sends it through render_messages, which is exactly what every frontend did
    # before notifications existed — so a transport whose only surface is the
    # chat needs no edits and loses nothing. Same arrangement as
    # supports_streaming above: declaring it opts into a structured channel,
    # declining it opts into the path that was always there.
    supports_notifications: bool = False
    # Frontend draws command/tool return values somewhere of its own (and
    # implements render_callable_output). False = the base sends them through
    # render_messages, which is what every frontend saw before the kind existed.
    # Same bargain as supports_notifications directly above.
    supports_callable_output: bool = False


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
            Same tuple format as SETTINGS_DATA. See plugins.native.tool.
        user_binding / default_user_id:
            How sessions map to users — "single" (default; all sessions act as
            default_user_id) or "per_user" (each identity its own user). See the
            attribute block below. This is the "whose data" axis; it does NOT set
            permissions — those come from frontend_profile.

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
        render_callable_output(session_key, messages)
                                                  — default no-op; needs
                                                    supports_callable_output.

    Provided (do NOT override):
        bind(runtime, registry, config)
        submit(session_key, action_type, payload=None) -> RuntimeResult
        submit_text(session_key, text)
        submit_attachment(session_key, path)
        cancel(session_key)
        bind_session(session_key, external_id=None) -> int   — apply user_binding
        identify(session_key, external_id, config=None) -> int — per_user upgrade
        mark_attended/mark_unattended(session_key)            — attendance
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    capabilities: FrontendCapabilities = FrontendCapabilities()

    # --- User binding (the "whose data" axis; authorization is frontend_profile) ---
    # How this frontend maps sessions to users. Declared, not guessed:
    #   "single"   — every session acts as ONE user, ``default_user_id``. REPL,
    #                Telegram, single-operator transports, kiosks, demos.
    #   "per_user" — each external identity gets its OWN user. Multi-user
    #                transports (a website). Unbound sessions act as
    #                ``default_user_id`` (point it at a guest user, NOT the base
    #                user, so anonymous traffic never lands on the operator);
    #                call ``bind_session(key, external_id)`` to upgrade a session
    #                to a real account on login.
    # The base auto-binds new sessions to ``default_user_id`` (only when unbound),
    # so a "single" frontend needs no per-session code at all.
    user_binding: str = USER_BINDING_SINGLE
    default_user_id: int = DEFAULT_USER_ID

    # --- Config settings this plugin needs ---
    # Each entry is a tuple:
    # (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []
    dependencies_files: list[str] = []
    dependencies_pip: list[str] = []

    # --- Agent system-prompt contribution ---
    # Guidance injected into the agent's system prompt when this frontend hosts the session.
    # Declare a plain string, or override with ``def agent_prompt(self, ctx)``
    # when the text depends on live state. ``ctx`` is a PromptContext,
    # carrying the session facts ``prompt_cues.SESSION_FACTS`` names —
    # session_key, conversation_id, user_id, profile_name, frontend_name,
    # security_mode — plus db/services/orchestrator/config/scope. The
    # collector accepts either shape.
    agent_prompt: str = ""

    # When a method-shaped contribution goes stale, and therefore which
    # block of the prompt it rides in. See ``prompt_cues.py`` for the
    # ladder; "" means the default rung.
    agent_prompt_refresh: str = ""

    def __init_subclass__(cls, **kwargs):
        """Internal helper to handle init subclass."""
        super().__init_subclass__(**kwargs)
        if isinstance(cls.config_settings, list):
            cls.config_settings = list(cls.config_settings)
        if isinstance(cls.dependencies_files, list):
            cls.dependencies_files = list(cls.dependencies_files)
        if isinstance(cls.dependencies_pip, list):
            cls.dependencies_pip = list(cls.dependencies_pip)
        if cls.user_binding not in _USER_BINDINGS:
            logger.warning(
                f"Frontend '{cls.name or cls.__name__}' declared invalid "
                f"user_binding={cls.user_binding!r}; falling back to "
                f"'{USER_BINDING_SINGLE}'. Valid: {_USER_BINDINGS}."
            )
            cls.user_binding = USER_BINDING_SINGLE

    def __init__(self):
        """Initialize the base frontend."""
        self.runtime = None
        self.commands = None     # CommandRegistry, set in bind()
        self.config: dict = {}
        self._unsubs: list = []
        self._bound = False
        self._approval_lock = threading.RLock()
        self._pending_approvals: dict[str, dict[str, object]] = {}
        self._pending_approval_order: dict[str, list[str]] = {}
        # Streaming bookkeeping: which stream this frontend is currently
        # rendering per session, and cleaned final texts already shown as
        # deltas (so the duplicate whole message can be skipped). Written on
        # the agent thread, consumed on the frontend thread — hence the lock.
        self._stream_lock = threading.Lock()
        self._active_stream_ids: dict[str, str] = {}
        self._streamed_finals: dict[str, list[str]] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle — override these.
    # ──────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start base frontend."""
        raise NotImplementedError(f"Frontend '{self.name}' must implement start()")

    def stop(self) -> None:
        """Stop base frontend."""
        raise NotImplementedError(f"Frontend '{self.name}' must implement stop()")

    def session_key(self, ctx) -> str:
        """Handle session key."""
        raise NotImplementedError(f"Frontend '{self.name}' must implement session_key()")

    # ──────────────────────────────────────────────────────────────────────
    # Rendering — override these. The base owns *when* to render; subclasses
    # own *how*.
    # ──────────────────────────────────────────────────────────────────────

    def render_messages(self, session_key: str, messages: list[str]) -> None:
        """Render messages: GitHub-flavored markdown, one string each.

        Markdown is deliberately the interchange format — it is also what the
        model emits, so a frontend needs exactly one rendering path. Tables
        must start their own block or GFM folds them into the paragraph above.
        """
        raise NotImplementedError

    def render_attachments(self, session_key: str, paths: list[str]) -> None:
        """Render attachments: filesystem paths on the host, not bytes.

        A transport that cannot reach the host's disk has to read them back
        (``fs.read_bytes``) before it can show anything.
        """
        raise NotImplementedError

    def render_form_field(self, session_key: str, form: dict) -> None:
        """Render a form prompt.

        ``form`` shape (from runtime/conversation_runtime.py form rendering):
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
        ``id``, ``title``, ``body``, ``type``, ``enum``, ``enum_labels``,
        ``default``.

        ``enum`` and ``enum_labels`` pair **by index**: answer with the value,
        show the label. A rich frontend renders one button per pair; a text one
        lists the labels. ``enum_labels`` may be ``None`` even when ``enum`` is
        not, in which case the values are the only text there is.
        """
        raise NotImplementedError

    def render_approval_settled(self, session_key: str, settled: dict) -> None:
        """A question stopped waiting. ``{request_id, reason}``.

        The counterpart to :meth:`render_approval_request`, and **the only way a
        surface that draws a dialog learns it may take it down.** Another
        frontend can answer the same question, and the approver denies by name
        after 300 seconds; neither of those is something this frontend did, and
        without this neither is something it could find out except by asking on
        a timer.

        ``reason`` is ``"answered"`` or ``"cancelled"``. It says how the question
        ended, not what the answer was — the answer went to whoever was blocked
        on it, and is deliberately not repeated to a bystander.

        Defaulted to nothing rather than ``NotImplementedError``, unlike its
        neighbours: a frontend written before this kind existed is correct as it
        stands, just chattier than it needs to be.
        """
        return

    def render_buttons(self, session_key: str, buttons: list[dict]) -> None:
        """Render quick replies, conventionally ``{value, label}`` each.

        The same pairing a form's ``display["choices"]`` uses, and answered the
        same way: submit the *value* as text. Nothing in the kernel currently
        fills ``RuntimeResult.buttons`` — this exists for store plugins, which
        is why the shape is a convention rather than a dataclass.
        """
        raise NotImplementedError

    def render_error(self, session_key: str, error: dict) -> None:
        """Render an error: ``ActionError.to_dict()``.

        ``{code, message, details, retry_phase}``. ``message`` is the part
        meant for a person; ``code`` is what a client branches on.
        """
        raise NotImplementedError

    def render_typing(self, session_key: str, on: bool) -> None:
        """Default no-op; rich frontends override to show a typing indicator."""
        return

    def render_tool_status(self, session_key: str, payload: dict) -> None:
        """Default no-op; frontends with status affordances override."""
        return

    def render_stream_delta(self, session_key: str, payload: dict) -> None:
        """Default no-op; frontends with ``supports_streaming`` override.

        Receives AGENT_TEXT_DELTA payloads for this frontend's sessions:
        fragments while the stream runs, then one ``done`` event (aborted
        streams have no ``final_text`` — discard the partial rendering and
        let the whole-message path deliver whatever follows)."""
        return

    def render_notification(self, session_key: str, payload: dict) -> None:
        """Default no-op; frontends with ``supports_notifications`` override.

        Receives NOTIFICATION_PUSHED payloads: ``title``, ``body``, ``source``,
        ``level`` (``info``/``success``/``warning``/``error``), and optionally
        ``source_id``, ``conversation_id``, ``load_hint``, ``notification_id``.

        A notification is the system telling the user something, as opposed to
        the conversation saying it — a plugin registering, a scheduled agent
        finishing, a setting changing. What to *do* with that is the whole
        reason the kind exists, so this method is deliberately empty: a
        transport with a notification area draws one, and a transport without
        never reaches here at all, because the base sends it down the message
        path instead (see :meth:`on_bus_notification_pushed`).

        ``load_hint`` is a pre-rendered slash command for reaching
        ``conversation_id``, there for surfaces that have no richer way to
        offer it. A client that can open a conversation itself should use the
        id and ignore the hint.
        """
        return

    def render_callable_output(self, session_key: str, messages: list[str]) -> None:
        """Default no-op; frontends with ``supports_callable_output`` override.

        What a slash command or a user-invoked tool *returned* — a `/config`
        listing, a `/conversations` table — as GitHub-flavored markdown, same
        wire convention as :meth:`render_messages`.

        Split out because it is an answer to something the person typed rather
        than something anybody said, and it was much the largest population
        making the message stream unreadable to a client: the agent's reply and
        a `/debug` dump arrived as the same kind of thing.

        Deliberately empty for the same reason ``render_notification`` is: a
        transport with somewhere to put it draws it there, and a transport
        without never reaches here, because ``_render_result`` sends it down
        the message path instead.
        """
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
            bus.subscribe(APPROVAL_SETTLED, self.on_bus_approval_settled),
            bus.subscribe(FORM_REQUESTED, self.on_bus_form_requested),
            bus.subscribe(CHAT_MESSAGE_PUSHED, self.on_bus_message_pushed),
            bus.subscribe(NOTIFICATION_PUSHED, self.on_bus_notification_pushed),
            bus.subscribe(AGENT_TEXT_DELTA, self.on_bus_agent_text_delta),
            bus.subscribe(COMMAND_CALL_STARTED, self.on_bus_command_call_started),
            bus.subscribe(COMMAND_CALL_PROGRESSED, self.on_bus_command_call_progressed),
            bus.subscribe(COMMAND_CALL_FINISHED, self.on_bus_command_call_finished),
            bus.subscribe(TOOL_CALL_STARTED, self.on_bus_tool_call_started),
            bus.subscribe(TOOL_CALL_FINISHED, self.on_bus_tool_call_finished),
            bus.subscribe(TOOLS_CHANGED, self.on_tools_changed),
            bus.subscribe(TASKS_CHANGED, self.on_tasks_changed),
            bus.subscribe(SESSION_CONVERSATION_CHANGED, self.on_bus_session_conversation_changed),
            bus.subscribe(CONVERSATION_CHANGED, self.on_bus_conversation_catalog_changed),
            bus.subscribe(SESSION_TURN_CHANGED, self.on_bus_session_turn_changed),
        ]
        self._bound = True

    def unbind(self) -> None:
        """Unbind base frontend."""
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
        """Submit base frontend."""
        if self.runtime is None:
            raise RuntimeError(
                f"Frontend '{self.name}' is not bound — call bind(runtime, ...) first."
            )
        self._tag_session(session_key)
        result = self.runtime.handle_action(session_key, action_type, payload)
        self._render_result(session_key, result)
        return result

    def submit_text(self, session_key: str, text: str):
        """Coerce raw user text into the right action for the current phase.

        - In a form phase, ``/cancel`` cancels, ``/back`` rewinds,
          blank text skips optional fields, and other text submits the field.
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
            if stripped == "/back" and ACTION_BACK_FORM in legal:
                return self.submit(session_key, ACTION_BACK_FORM)
            if stripped.startswith("/") and ACTION_CALL_COMMAND in legal:
                name, _, arg = stripped[1:].partition(" ")
                # Switching commands mid-form is a shortcut, and a step that
                # takes paths is where it stops being one: the reported value
                # "/Users/henry/My Drive/..." was split on its first space and
                # came back as the unknown command "Users/henry/My", so
                # ``ignored_folders`` could not be given a folder from a Mac.
                # ``FormStep.takes_literal_text`` says which steps those are —
                # the step knows, and the frontend cannot tell from the text.
                step = self._current_form_step(session_key)
                if step is not None and getattr(step, "takes_literal_text", False):
                    return self.submit(session_key, ACTION_SUBMIT_FORM_TEXT, stripped)
                cmd = next((c for c in self.commands.all_commands() if c.name == name), None) if name and self.commands else None
                if cmd and not self.command_allowed(name):
                    return self._command_not_allowed(session_key, name)
                if cmd:
                    args, handled = self._parse_command_args(session_key, name, arg)
                    if handled is not None:
                        return handled
                    return self.submit(session_key, ACTION_CALL_COMMAND, {"name": name, "args": args})
                return self._unknown_command(session_key, name)
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
            if (text or "").strip() == "/cancel" and ACTION_CANCEL in legal:
                result = self.submit(session_key, ACTION_CANCEL)
                if self._current_phase(session_key) != PHASE_APPROVING_REQUEST:
                    self._clear_pending_approval(session_key)
                return result
            if (text or "").lstrip().startswith("/"):
                return self._unknown_command(session_key, (text or "").lstrip()[1:].partition(" ")[0])
            result = self.submit(session_key, ACTION_ANSWER_APPROVAL, text)
            # Invalid input leaves the state machine on the approval frame.
            # Keep its rich prompt registered so a stray chat message cannot
            # make the permission dialog disappear.
            if self._current_phase(session_key) != PHASE_APPROVING_REQUEST:
                self._clear_pending_approval(session_key)
            return result

        stripped = (text or "").lstrip()
        if stripped == "/cancel":
            return self.cancel(session_key)
        if getattr(self.runtime.get_session(session_key), "busy", False):
            return self.submit(session_key, ACTION_SEND_TEXT, text)
        if stripped.startswith("/"):
            name, _, arg = stripped[1:].partition(" ")
            cmd = next((c for c in self.commands.all_commands() if c.name == name), None) if name and self.commands else None
            if cmd and not self.command_allowed(name):
                return self._command_not_allowed(session_key, name)
            if cmd:
                args, handled = self._parse_command_args(session_key, name, arg)
                if handled is not None:
                    return handled
                return self.submit(
                    session_key,
                    ACTION_CALL_COMMAND,
                    {"name": name, "args": args},
                )
            return self._unknown_command(session_key, name)
        return self.submit(session_key, ACTION_SEND_TEXT, text)

    def submit_attachment(self, session_key: str, path: str, extension: str | None = None):
        """Submit attachment."""
        from pathlib import Path
        ext = extension or Path(path).suffix.lstrip(".")
        return self.submit(
            session_key,
            ACTION_SEND_ATTACHMENT,
            {"path": path, "extension": ext},
        )

    def cancel(self, session_key: str):
        """Cancel base frontend."""
        return self.submit(session_key, ACTION_CANCEL)

    def _parse_command_args(self, session_key: str, name: str, arg: str):
        """Parse one-shot command args, rendering bad input instead of raising.

        ``FormStep.coerce`` raises ``ValueError`` on invalid values (wrong
        enum member, bad JSON, non-numeric numbers); an uncaught raise here
        dies inside the transport's handler thread and the user sees nothing.
        Returns ``(args, None)`` or ``(None, result)`` when already handled.
        """
        try:
            return (self.commands.parse_args(name, arg, session_key=session_key) if arg.strip() else {}), None
        except Exception as e:
            result = RuntimeResult(False, error={
                "code": "bad_command_args",
                "action": "call_command", "name": name,
                "message": f"Invalid arguments for `/{name}`: {e}\nType `/{name}` alone to fill them in step by step."})
            self._render_result(session_key, result)
            return None, result

    def _unknown_command(self, session_key: str, name: str):
        """Render an unknown slash-command error without waking the agent."""
        result = RuntimeResult(False, error={
            "code": "unknown_command",
            "action": "call_command", "name": name,
            "message": f"`/{name}` isn't a recognized slash command. Type `/commands` to see the full list of what's available."})
        self._render_result(session_key, result)
        return result

    def _command_not_allowed(self, session_key: str, name: str):
        """Render an error for a command blocked by this frontend's profile."""
        result = RuntimeResult(False, error={
            "code": "command_not_allowed",
            "action": "call_command", "name": name,
            "message": f"`/{name}` is not available on the '{self.name}' frontend. Type `/commands` to see what's available here."})
        self._render_result(session_key, result)
        return result

    def command_allowed(self, name: str) -> bool:
        """Whether ``name`` may run on this frontend under its profile."""
        from plugins.command_registry import command_allowed
        return command_allowed(self.config, self.name, name)

    def _tag_session(self, session_key: str) -> None:
        """Stamp the originating frontend onto a session so the runtime can
        apply this frontend's profile (agent scope + command access)."""
        try:
            session = self.runtime.get_session(session_key)
        except Exception:
            return
        if getattr(session, "frontend_name", None) != self.name:
            session.frontend_name = self.name
        # Apply this frontend's declared default binding the first time we see a
        # session — so a "single" frontend needs no per-session code, and a
        # "per_user" frontend's not-yet-identified sessions land on its guest
        # default (default_user_id) instead of the kernel base user. An explicit
        # bind_session()/identify() that already set a user is left untouched.
        if getattr(session, "user_id", None) is None:
            session.user_id = self.default_user_id

    def mark_attended(self, session_key: str) -> None:
        """Declare that a human is present at ``session_key`` (e.g. a websocket
        connected). Concurrent multi-user frontends call this so each user's
        session is treated as foreground independently; single-user frontends
        (REPL, Telegram) can ignore it and inherit the global active-session
        behavior unchanged."""
        if self.runtime is not None:
            self.runtime.set_session_attended(session_key, True)

    def mark_unattended(self, session_key: str) -> None:
        """Declare that no human is present at ``session_key`` (e.g. the socket
        closed). Interactive tools will be refused and replies delivered as
        notifications until ``mark_attended`` is called again."""
        if self.runtime is not None:
            self.runtime.set_session_attended(session_key, False)

    def identify(self, session_key: str, external_id, config: dict | None = None, user_type: str = "user") -> int | None:
        """Resolve (creating if needed) the user behind this session and bind it.

        This frontend's own ``name`` namespaces the identity, so ``external_id``
        only has to be unique within the frontend (a cookie, a chat id, an account
        username). Returns the user_id, or None when there's no database.

        Single-user frontends (REPL) never call this — their sessions stay on the
        base user. A public frontend binds a guest user for anonymous sessions and
        re-binds to a real account on login. Authorization is unaffected: it lives
        in the frontend_profile, not the user."""
        if self.runtime is None or getattr(self.runtime, "db", None) is None:
            return None
        if self.user_binding == USER_BINDING_SINGLE:
            logger.debug(
                f"Frontend '{self.name}' is user_binding='single' but called "
                f"identify(); the per-user binding is ignored unless you switch "
                f"to 'per_user'."
            )
        uid = self.runtime.db.upsert_user(self.name, str(external_id), config, user_type=user_type)
        self.runtime.set_session_user(session_key, uid)
        return uid

    def bind_session(self, session_key: str, external_id=None) -> int | None:
        """Bind a session to a user per this frontend's declared ``user_binding``.
        The one call a frontend needs — no guessing. Returns the bound user_id.

        - ``single``   → always ``default_user_id`` (``external_id`` ignored).
        - ``per_user`` → the user for ``external_id`` (created on first sight via
          ``identify``); with no ``external_id``, the anonymous ``default_user_id``
          (typically a guest user).

        Note the base already auto-binds new sessions to ``default_user_id``, so
        ``single`` frontends never need to call this. ``per_user`` frontends call
        it to *upgrade* a session to a real account once the user authenticates.
        """
        if self.runtime is None:
            return None
        if self.user_binding == USER_BINDING_PER_USER and external_id is not None:
            return self.identify(session_key, external_id)
        self.runtime.set_session_user(session_key, self.default_user_id)
        return self.default_user_id

    # ──────────────────────────────────────────────────────────────────────
    # Bus handlers. Subclasses can override for richer behavior, but the
    # defaults route everything through the abstract render_* methods.
    # ──────────────────────────────────────────────────────────────────────

    def on_bus_approval_requested(self, req) -> None:
        """Handle on bus approval requested."""
        target = ((getattr(req, "metadata", None) or {}).get("session_key"))
        for key in self._announce_to(target):
            try:
                self._register_pending_approval(key, req)
                self.render_approval_request(key, req)
            except Exception:
                logger.exception(f"render_approval_request failed for '{self.name}'")

    def on_bus_approval_settled(self, payload) -> None:
        """Forget a question that stopped waiting, and say so.

        The forgetting matters as much as the telling: ``frontend.pending`` and
        ``frontend.resolve`` both settle existence against this table, so a
        registration left behind reports an answered question as live and hands
        out an id that ``resolve`` then refuses.
        """
        payload = payload or {}
        request_id = payload.get("request_id")
        if not request_id:
            return
        for key in self._announce_to(payload.get("session_key")):
            try:
                self._clear_pending_approval(key, request_id)
                self.render_approval_settled(key, {
                    "request_id": request_id,
                    "reason": payload.get("reason") or "answered",
                })
            except Exception:
                logger.exception(f"render_approval_settled failed for '{self.name}'")

    def on_bus_form_requested(self, payload) -> None:
        """Re-prompt a form restored onto a session after a restart.

        Mirrors ``on_bus_approval_requested``: in normal flow the form rides
        back as ``RuntimeResult.form`` on a live submit(); this path only fires
        when restore re-emits one with no submit() in flight."""
        payload = payload or {}
        form = payload.get("form")
        if not form:
            return
        for key in self._announce_to(payload.get("session_key")):
            try:
                self.render_form_field(key, dict(form))
            except Exception:
                logger.exception(f"render_form_field failed for '{self.name}'")

    def _announce_to(self, target: str | None) -> list[str]:
        """Which of this frontend's sessions a bus event should reach.

        **A named target this frontend does not have reaches nothing.** Both
        callers carry one — ``request_input`` always stamps ``session_key`` into
        the request's metadata — so the old fallback to
        :meth:`_broadcast_session_keys` fired precisely when the question
        belonged to *somebody else's* session, and fanned it across every
        session of ours instead of dropping it. A question asked at the REPL
        raised a dialog in the browser, in a conversation it had nothing to do
        with, and answering it there drove the REPL's session.

        The generosity that remains is deliberate and lives one level down:
        :meth:`_live_session_keys` keeps *untagged* sessions, because a session
        nobody has claimed may be one this frontend is about to receive. A
        question aimed at one of those still reaches here. What no longer
        happens is a question aimed at a session another frontend owns.

        Broadcasting is left for the genuinely untargeted case, which is what
        it was written for.
        """
        if not target:
            return self._broadcast_session_keys()
        return [target] if target in self._live_session_keys() else []

    def resolve_approval(self, session_key: str, request_id: str, value, resolved_by: str | None = None) -> bool:
        """Resolve approval."""
        with self._approval_lock:
            req = self._pending_approvals.get(session_key, {}).get(request_id)
            if req is None:
                return False
            if resolved_by and hasattr(req, "metadata"):
                req.metadata["resolved_by"] = resolved_by
            target = (getattr(req, "metadata", {}) or {}).get("session_key") or session_key
        payload = {"value": value, "request_id": request_id}
        if (getattr(req, "metadata", {}) or {}).get("render_result_on_resolve"):
            # Callable approvals already returned "Approval required" to the
            # transport, so nothing else will render their resumed result.
            result = self.submit(target, ACTION_ANSWER_APPROVAL, payload)
        else:
            # Tool/policy approvals have an original sandbox call blocked on
            # the answer. It still owns the guest renderer lock; attempting to
            # render here would wait on that call while that call waits on us.
            result = self.runtime.handle_action(target, ACTION_ANSWER_APPROVAL, payload)
        if self._current_phase(target) != PHASE_APPROVING_REQUEST:
            self._clear_pending_approval(session_key, request_id)
        return bool(result and result.ok)

    def resolve_next_approval(self, session_key: str, value, resolved_by: str | None = None) -> bool:
        """Resolve next approval."""
        with self._approval_lock:
            order = self._pending_approval_order.setdefault(session_key, [])
            pending = self._pending_approvals.setdefault(session_key, {})
            while order and (order[0] not in pending or getattr(pending[order[0]], "is_resolved", False)):
                order.pop(0)
            return bool(order) and self.resolve_approval(session_key, order.pop(0), value, resolved_by)

    def has_pending_approval(self, session_key: str) -> bool:
        """Return whether pending approval."""
        with self._approval_lock:
            return any(not getattr(req, "is_resolved", False) for req in self._pending_approvals.get(session_key, {}).values())

    def is_approval_pending(self, session_key: str, request_id: str | None = None) -> bool:
        """Whether a specific approval is still waiting to be answered.

        A dictionary lookup, deliberately: it answers "does this still exist?"
        without driving the state machine, which is what lets a *detached*
        resolve report honestly instead of optimistically. Answering an
        approval has to run off the caller's thread — see ``_drive`` in the
        sandbox handlers — and a caller that only learns "accepted" cannot
        tell an already-answered request from a live one.
        """
        if request_id is None:
            return self.has_pending_approval(session_key)
        with self._approval_lock:
            req = self._pending_approvals.get(session_key, {}).get(request_id)
            return req is not None and not getattr(req, "is_resolved", False)

    def _clear_pending_approval(self, session_key: str, request_id: str | None = None) -> None:
        """Internal helper to clear pending approval."""
        with self._approval_lock:
            if request_id is None:
                self._pending_approvals.pop(session_key, None)
                self._pending_approval_order.pop(session_key, None)
                return
            self._pending_approvals.get(session_key, {}).pop(request_id, None)
            self._pending_approval_order[session_key] = [item for item in self._pending_approval_order.get(session_key, []) if item != request_id]

    def _register_pending_approval(self, session_key: str, req) -> None:
        """Register an approval once, preserving its display order."""
        with self._approval_lock:
            pending = self._pending_approvals.setdefault(session_key, {})
            pending[req.id] = req
            order = self._pending_approval_order.setdefault(session_key, [])
            if req.id not in order:
                order.append(req.id)

    def on_bus_message_pushed(self, payload: dict) -> None:
        """Handle on bus message pushed."""
        payload = payload or {}
        message = payload.get("message")
        # A push may be files with no words — ``sdk.ui.render`` with no caption
        # sends exactly that, so an early return on a falsy message would drop
        # the whole point of the call.
        paths = [str(p) for p in (payload.get("attachments") or [])]
        if not message and not paths:
            return
        title = payload.get("title")
        body = f"{title}\n\n{message}" if title and message else (message or title or "")
        target = payload.get("session_key")
        keys = [target] if target else self._broadcast_session_keys()
        live = self._live_session_keys()
        for key in keys:
            if key not in live:
                continue
            # A body already streamed in incrementally is suppressed; the files
            # never were, so they still go out.
            if body and not self._consume_streamed(key, body):
                try:
                    self.render_messages(key, [body])
                except Exception:
                    logger.exception(f"render_messages (push) failed for '{self.name}'")
            if paths:
                try:
                    self.render_attachments(key, list(paths))
                except Exception:
                    logger.exception(f"render_attachments (push) failed for '{self.name}'")

    def on_bus_notification_pushed(self, payload: dict) -> None:
        """Route a notification to whichever surface this frontend has for one.

        Targeting is the same as a push: a named ``session_key`` goes there,
        an unnamed one broadcasts to this frontend's own sessions. That
        distinction matters more here than it does for chat, because the
        notifications worth having are mostly raised *by* sessions nobody is
        watching — a scheduled agent's conversation has no frontend attached,
        so delivering to its origin would deliver to nowhere.

        A frontend that declared ``supports_notifications`` gets the payload
        whole. One that did not gets it flattened into markdown through
        ``render_messages``, which is byte-for-byte the path these
        announcements took before the kind existed.
        """
        payload = payload or {}
        if not payload.get("title") and not payload.get("body"):
            return
        target = payload.get("session_key")
        keys = [target] if target else self._broadcast_session_keys()
        live = self._live_session_keys()
        rich = getattr(self.capabilities, "supports_notifications", False)
        text = None if rich else self._notification_markdown(payload)
        for key in keys:
            if key not in live:
                continue
            try:
                if rich:
                    self.render_notification(key, dict(payload))
                elif text:
                    self.render_messages(key, [text])
            except Exception:
                logger.exception(f"render notification failed for '{self.name}'")

    @staticmethod
    def _notification_markdown(payload: dict) -> str:
        """Flatten a notification into the one thing every frontend can show.

        Deliberately plain. The markdown-on-the-wire convention means each
        frontend renders by policy, so anything cleverer here — a table, a
        detail card — would be a formatting decision made on behalf of
        transports that have their own opinion about it.
        """
        title = (payload.get("title") or "").strip()
        body = (payload.get("body") or "").strip()
        parts = [f"{title}\n\n{body}" if title and body else (title or body)]
        if hint := (payload.get("load_hint") or "").strip():
            parts.append(f"Load this conversation: `{hint}`")
        return "\n\n".join(p for p in parts if p)

    def on_bus_agent_text_delta(self, payload: dict) -> None:
        """Route streamed text deltas to ``render_stream_delta`` with dedup
        bookkeeping. Ignored entirely unless this frontend supports streaming
        and owns the session."""
        if not getattr(self.capabilities, "supports_streaming", False):
            return
        payload = payload or {}
        key = payload.get("session_key")
        stream_id = payload.get("stream_id")
        if not key or not stream_id or key not in self._live_session_keys():
            return
        if payload.get("done"):
            with self._stream_lock:
                rendered_here = self._active_stream_ids.pop(key, None) == stream_id
                if not rendered_here:
                    return  # never saw this stream's deltas — nothing to close
                if not payload.get("aborted") and payload.get("final_text"):
                    finals = self._streamed_finals.setdefault(key, [])
                    finals.append(payload["final_text"])
                    del finals[:-8]  # cap: stale entries expire instead of leaking
        else:
            with self._stream_lock:
                self._active_stream_ids[key] = stream_id
        try:
            self.render_stream_delta(key, dict(payload))
        except Exception:
            logger.exception(f"render_stream_delta failed for '{self.name}'")

    def _consume_streamed(self, session_key: str, message: str) -> bool:
        """True (and forget the entry) when ``message`` was already rendered
        as a completed stream for this session — the whole-message duplicate
        should be skipped."""
        with self._stream_lock:
            finals = self._streamed_finals.get(session_key)
            if finals and message in finals:
                finals.remove(message)
                return True
        return False

    def on_bus_tool_call_started(self, payload: dict) -> None:
        """Handle on bus tool call started."""
        self._render_tool_status_event({**(payload or {}), "status": "started"})

    def on_bus_tool_call_finished(self, payload: dict) -> None:
        """Handle on bus tool call finished."""
        self._render_tool_status_event({**(payload or {}), "status": "finished"})

    def on_bus_command_call_started(self, payload: dict) -> None:
        """Handle on bus command call started."""
        self._render_tool_status_event({**(payload or {}), "status": "started", "kind": "command"})

    def on_bus_command_call_progressed(self, payload: dict) -> None:
        """Handle on bus command call progressed."""
        self._render_tool_status_event({**(payload or {}), "status": "progressed", "kind": "command"})

    def on_bus_command_call_finished(self, payload: dict) -> None:
        """Handle on bus command call finished."""
        self._render_tool_status_event({**(payload or {}), "status": "finished", "kind": "command"})

    def on_bus_session_turn_changed(self, payload: dict) -> None:
        """Turn priority moved — track the typing indicator to whose turn it is.

        Priority (not per-drive lifecycle) is the right axis: it only moves on
        a real handoff. A barrier-held turn (e.g. spawn_agent wait=false waiting
        on subagents) or an escalation re-drive keeps priority with the agent
        across its interim drives, so no event fires and typing stays on until
        the logical turn truly hands back to the user. Crashes force priority
        back to the user, so that path clears typing too."""
        payload = payload or {}
        to_actor = payload.get("to_actor")
        if to_actor == "agent":
            self._route_typing(payload.get("session_key"), True)
        elif to_actor == "user":
            self._route_typing(payload.get("session_key"), False)

    def _route_typing(self, session_key: str | None, on: bool) -> None:
        """Route a turn-lifecycle typing change to ``render_typing``."""
        if not getattr(self.capabilities, "supports_typing", False):
            return
        if not session_key or session_key not in self._live_session_keys():
            return
        try:
            self.render_typing(session_key, on)
        except Exception:
            logger.exception(f"render_typing failed for '{self.name}'")

    def on_bus_session_conversation_changed(self, payload: dict) -> None:
        """Route a session's conversation switch/retitle to the banner hook."""
        payload = payload or {}
        key = payload.get("session_key")
        if not key or key not in self._live_session_keys():
            return
        try:
            self.render_conversation_banner(key, dict(payload))
        except Exception:
            logger.exception(f"render_conversation_banner failed for '{self.name}'")

    def on_bus_conversation_catalog_changed(self, payload: dict) -> None:
        """Refresh banners when the conversation a live session shows is retitled."""
        payload = payload or {}
        if payload.get("action") != "retitled":
            return
        cid = payload.get("conversation_id")
        if cid is None or self.runtime is None:
            return
        for key in self._live_session_keys():
            session = self.runtime.sessions.get(key)
            if session is None or session.conversation_id != cid:
                continue
            row = self.runtime.db.get_conversation(cid) if self.runtime.db else None
            title = ((row or {}).get("title") or "").strip() or "New Conversation"
            self.on_bus_session_conversation_changed(
                {"session_key": key, "conversation_id": cid, "title": title})

    def render_conversation_banner(self, session_key: str, info: dict) -> None:
        """Default no-op; frontends with a persistent surface (pinned message,
        window title) override to mirror the session's conversation title."""
        return

    def on_tools_changed(self, _payload) -> None:
        """Handle on tools changed."""
        return

    def on_tasks_changed(self, _payload) -> None:
        """Handle on tasks changed."""
        return

    # ──────────────────────────────────────────────────────────────────────
    # Internals.
    # ──────────────────────────────────────────────────────────────────────

    def _render_result(self, session_key: str, result) -> None:
        """Internal helper to render result."""
        if result is None:
            return
        if result.messages:
            messages = [m for m in result.messages if not self._consume_streamed(session_key, m)]
            if messages:
                self.render_messages(session_key, messages)
        if result.callable_output:
            # The fallback lives here rather than in a default
            # ``render_callable_output`` body, because ``residency`` replaces
            # every ``render_*`` wholesale with the box forwarder for sandboxed
            # frontends — which is all of them — so a default would never run.
            # Same reason the notification fallback lives on its bus handler.
            if self.capabilities.supports_callable_output:
                self.render_callable_output(session_key, list(result.callable_output))
            else:
                self.render_messages(session_key, list(result.callable_output))
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
            # Callable approvals arrive on RuntimeResult rather than the
            # APPROVAL_REQUESTED bus used by tool-originated requests. They
            # still need to be registered or frontend.pending/resolve cannot
            # see the very request we just rendered.
            self._register_pending_approval(session_key, req)
            self.render_approval_request(session_key, req)

    def _current_phase(self, session_key: str) -> str:
        """Return current phase."""
        session = self.runtime.get_session(session_key)
        return session.cs.phase

    def _current_form_step(self, session_key: str):
        """Return current form step."""
        frame = self.runtime.get_session(session_key).cs.frame
        return getattr(frame, "step", None) if frame else None

    def _current_approval_request(self, session_key: str):
        """Return current approval request."""
        if self._current_phase(session_key) != PHASE_APPROVING_REQUEST:
            return None
        frame = self.runtime.get_session(session_key).cs.frame
        data = getattr(frame, "data", {}) or {}
        if not data.get("request_id"):
            data["request_id"] = f"approve_{uuid.uuid4().hex}"
        return StateMachineApprovalRequest(
            title=data.get("title") or frame.name or "Input required",
            body=_approval_body(data.get("prompt") or ""),
            pending_action=data.get("pending"),
            id=data["request_id"],
            type=data.get("type", "boolean"),
            enum=data.get("enum"),
            enum_labels=data.get("enum_labels"),
            default=data.get("default"),
            metadata={
                "session_key": session_key,
                "render_result_on_resolve": True,
                **({"detail": data["detail"]} if data.get("detail") else {}),
            },
        )

    def _live_session_keys(self) -> list[str]:
        """Session keys this frontend currently has open.

        Every session the runtime knows about, *except* the ones no frontend
        owns. A background driver — a subagent above all — drives a session
        that belongs to nobody, and it renders exactly like any other turn:
        tool statuses, streamed text, typing. Without this filter every
        frontend rendered every one of them, so a subagent spawned from
        Telegram typed its tool calls into the REPL in front of whoever
        happened to be sitting there.

        The rule is ownership, not the subagent prefix specifically: a session
        driven by nothing a person is looking at has no frontend to render to.
        Subclasses that multiplex several platforms behind one runtime should
        still override this to scope it to their own sessions.
        """
        if self.runtime is None:
            return []
        return [key for key, session in list(self.runtime.sessions.items())
                if not self._is_background_session(key, session)]

    def _broadcast_session_keys(self) -> list[str]:
        """Where an *untargeted* announcement goes — this frontend's own sessions.

        Distinct from :meth:`_live_session_keys`, and the distinction is the
        whole point. A *targeted* render names a session, and the live set is
        deliberately generous about untagged ones: a session nobody has claimed
        may be one this frontend is about to receive, and dropping it would lose
        the first message of a conversation.

        A broadcast has no session to name. Fanning it across the live set
        therefore renders it once *per session*, and an untagged session is in
        every frontend's live set — so with the REPL and Telegram both running,
        one "Registered plugin: x" printed twice in the terminal: once for the
        REPL's own session, once for Telegram's not-yet-tagged one. The
        transport is one surface; the announcement belongs on it once.

        Falling back to the live set when this frontend owns nothing keeps a
        fresh install (no session has submitted yet, so nothing is tagged) from
        silently swallowing every announcement.
        """
        live = self._live_session_keys()
        if self.runtime is None or not live:
            return live
        sessions = getattr(self.runtime, "sessions", None) or {}
        owned = [key for key in live
                 if getattr(sessions.get(key), "frontend_name", None) == self.name]
        return owned or live

    @staticmethod
    def _is_background_session(key: str, session) -> bool:
        """Whether this session is driven by code rather than by a person.

        Asked of the runtime's own marker for it — a spawned agent's session
        key — rather than of anything the session claims about itself.
        """
        from runtime.subagents import is_subagent_session
        return is_subagent_session(key)

    def _render_tool_status_event(self, payload: dict) -> None:
        """Internal helper to render tool status event."""
        key = (payload or {}).get("session_key")
        if not key or key not in self._live_session_keys():
            return
        try:
            self.render_tool_status(key, payload)
        except Exception:
            logger.exception(f"render_tool_status failed for '{self.name}'")
