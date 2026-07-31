"""
Runtime context passed into plugins.

The context packages together the database handle, config, shared
services, and a few runtime helpers so plugins do not need to know how
the surrounding application is wired.

Parsing is deliberately *not* here and is not a service: it is kernel
routing plus importable parser functions. Use ``parsing.get_modality`` to
ask what a file is and ``parsing.parse`` to read one as text, or
``parsing.parser_for`` to pull a parser into your own box when you need a
heavier modality whose result cannot travel.
"""

from dataclasses import dataclass, field
from typing import Any

from config.config_manager import DEFAULTS, USER_CONFIG_KEYS
from pipeline.database import DEFAULT_USER_ID


@dataclass
class SecondBrainContext:
    """
    The runtime context every task and tool receives.

    db:
        Database instance for reads and writes.
    config:
        Global settings dict.
    services:
        Mapping of service name to service instance. Parsing is not among
        them — see the module docstring.
    call_tool:
        Helper for tool-to-tool composition. Only populated for tools.
        Example:
        context.call_tool("hybrid_search", query="revenue") -> ToolResult
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    services: dict = field(default_factory=dict)
    call_tool: Any = None        # callable(name, **kwargs) -> ToolResult (tools only)
    request_user_input: Any = None # callable(...)->StateMachineApprovalRequest (tools only)
    tool_registry: Any = None    # ToolRegistry instance (tools only)
    orchestrator: Any = None     # Orchestrator instance (tools only)
    runtime: Any = None          # ConversationRuntime — present for tasks that
                                 # need to drive a state-machine session.
    root_dir: Any = None         # Project root for repo/plugin operations.
    command_registry: Any = None # Slash-command registry for command plugins.
    app_control: Any = None      # object with stop()/restart() -> str; only the
                                 # composition root has these, and /quit and
                                 # /restart reach them via the app.stop Request.
    session_key: str | None = None # Frontend conversation/session key, when available.
    user_id: int = DEFAULT_USER_ID # Effective user this call acts for (the base user when no frontend bound one).
    current_user: Any = None     # callable() -> user row dict (config parsed) or None.
    user_config: dict = field(default_factory=dict)
    user_initiated: bool = False # Explicit user command, not an autonomous agent call.
    approved_by_state_machine: bool = False
    current_tool_name: str | None = None


def build_context(db, config: dict, services: dict, call_tool=None,
                   tool_registry=None, orchestrator=None,
                   runtime=None, app_control=None,
                   root_dir=None, command_registry=None,
                   session_key: str | None = None,
                   user_initiated: bool = False,
                   current_tool_name: str | None = None) -> SecondBrainContext:
    """
    Build a fully wired runtime context.

    Usage:
        # In orchestrator (tasks — no call_tool):
        context = build_context(self.db, self.config, self.services)

        # In tool registry (tools — with call_tool):
        context = build_context(self.db, self.config, self.services, call_tool=self.call)
    """
    def call_tool_with_session(name, **kwargs):
        """Call tool with session."""
        if session_key and "_session_key" not in kwargs:
            kwargs["_session_key"] = session_key
        return call_tool(name, **kwargs)

    # Resolve the effective user from the live session (frontend-bound, ephemeral).
    # Falls back to the base user when nothing was bound. This is the "whose data"
    # axis — orthogonal to authorization, which lives in frontend_profile.
    user_id = DEFAULT_USER_ID
    if runtime is not None and session_key:
        _s = getattr(runtime, "sessions", {}).get(session_key)
        if _s is not None and getattr(_s, "user_id", None) is not None:
            user_id = _s.user_id
    user_cfg = runtime.user_config(session_key) if runtime is not None and session_key and hasattr(runtime, "user_config") else (db.get_user_config(user_id) if db is not None else {})
    effective_config = dict(config or {})
    for key in USER_CONFIG_KEYS:
        if key in DEFAULTS:
            effective_config[key] = user_cfg.get(key, (config or {}).get(key, DEFAULTS.get(key)))
        elif key in user_cfg:
            effective_config[key] = user_cfg[key]
    current_user = (lambda: db.get_user(user_id)) if db is not None else None

    request_user_input = None
    if runtime is not None and session_key:
        def request_user_input(title: str, prompt: str, **kwargs):
            """Handle request user input."""
            return runtime.request_input(session_key, title, prompt, **kwargs)

    ctx = SecondBrainContext(
        db=db,
        config=effective_config,
        services=services,
        call_tool=call_tool_with_session if call_tool is not None else None,
        request_user_input=request_user_input,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        runtime=runtime,
        app_control=app_control,
        root_dir=root_dir,
        command_registry=command_registry,
        session_key=session_key,
        user_id=user_id,
        current_user=current_user,
        user_config=user_cfg,
        user_initiated=user_initiated,
        current_tool_name=current_tool_name,
    )
    return ctx


# ──────────────────────────────────────────────────────────────────────
# The kernel's own context, for code that belongs to no session.
# ──────────────────────────────────────────────────────────────────────
#
# Handlers on the host side of the sandbox answer Requests *from* a context.
# An ephemeral run is handed one per call and a frontend when its box opens,
# but a resident service has neither: it is loaded at boot, it acts on its own
# initiative, and there is no session to build one from. Without this it
# answered from nothing — ``sdk.config.read`` returned None for every key and
# ``sdk.config.write`` failed outright.
#
# It is a mutable holder rather than a value because boot order is fixed:
# services load at main.pyw step 3, long before the orchestrator (step 4), the
# tool registry (5b) or the runtime (bootstrap). The same problem the command
# registry solves with a ``ref`` closure, solved the same way and in the place
# that already owns context construction.

_KERNEL_PARTS: dict = {}


def set_kernel_parts(**parts) -> None:
    """Record whatever the composition root has built so far.

    ``None`` values are ignored, so a later call only ever adds: passing
    ``runtime=None`` before there is a runtime cannot erase a real one.
    """
    _KERNEL_PARTS.update({k: v for k, v in parts.items() if v is not None})


def kernel_config() -> dict:
    """The live config dict the composition root built, or ``{}``.

    A reader for host-side code that needs a setting but has no context and no
    session — the sandbox's policy function, which is handed a Request and a
    chain and nothing else. It returns the *same* dict the kernel holds rather
    than a reload, so a ``/config`` write is visible on the next read.
    """
    return _KERNEL_PARTS.get("config") or {}


def kernel_runtime():
    """The live :class:`ConversationRuntime`, or ``None``.

    The counterpart to :func:`kernel_config`, and for the same caller: policy
    code that is handed a Request and a chain and nothing else, but has to ask
    the runtime a question only it can answer — whether anybody is sitting at
    the session a chain names. ``None`` before the composition root has built
    one, which every reader must treat as "nobody is there".
    """
    return _KERNEL_PARTS.get("runtime")


_POSITIONAL = ("db", "config", "services")


def kernel_context(session_key: str | None = None) -> SecondBrainContext:
    """A context for work that belongs to the kernel rather than a session.

    Builds from whatever has been wired so far rather than requiring the full
    set, since a service loading at step 3 must not fail because the tool
    registry does not exist until 5b.
    """
    return build_context(
        _KERNEL_PARTS.get("db"),
        _KERNEL_PARTS.get("config") or {},
        _KERNEL_PARTS.get("services") or {},
        session_key=session_key,
        **{k: v for k, v in _KERNEL_PARTS.items() if k not in _POSITIONAL},
    )
