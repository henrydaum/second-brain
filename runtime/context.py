"""
Runtime context passed into plugins.

The context packages together the database handle, config, shared
services, and a few runtime helpers so plugins do not need to know how
the surrounding application is wired. Parsing is reached uniformly via
``context.services.get("parser").parse(path, modality)`` — no special
shortcut on the context.
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
        Mapping of service name to service instance. Includes the
        "parser" service for file parsing.
    call_tool:
        Helper for tool-to-tool composition. Only populated for tools.
        Example:
        context.call_tool("hybrid_search", query="revenue") -> ToolResult
    approve_command:
        Helper for user approval on sensitive actions. Tools only. None means
        no state-machine session is available, so tools should treat that as
        deny.
    """
    db: Any = None
    config: dict = field(default_factory=dict)
    services: dict = field(default_factory=dict)
    call_tool: Any = None        # callable(name, **kwargs) -> ToolResult (tools only)
    approve_command: Any = None  # callable(command, justification) -> bool (tools only)
    request_user_input: Any = None # callable(...)->StateMachineApprovalRequest (tools only)
    tool_registry: Any = None    # ToolRegistry instance (tools only)
    orchestrator: Any = None     # Orchestrator instance (tools only)
    runtime: Any = None          # ConversationRuntime — present for tasks that
                                 # need to drive a state-machine session.
    root_dir: Any = None         # Project root for repo/plugin operations.
    command_registry: Any = None # Slash-command registry for command plugins.
    session_key: str | None = None # Frontend conversation/session key, when available.
    user_id: int = DEFAULT_USER_ID # Effective user this call acts for (the base user when no frontend bound one).
    current_user: Any = None     # callable() -> user row dict (config parsed) or None.
    user_config: dict = field(default_factory=dict)
    user_initiated: bool = False # Explicit user command, not an autonomous agent call.
    current_tool_name: str | None = None
    approval_denial_reason: str = ""


def build_context(db, config: dict, services: dict, call_tool=None,
                   tool_registry=None, orchestrator=None,
                   runtime=None,
                   root_dir=None, command_registry=None,
                   session_key: str | None = None,
                   user_initiated: bool = False,
                   current_tool_name: str | None = None) -> SecondBrainContext:
    """
    Build a fully wired runtime context.

    approve_command is backed by the conversation state machine when a tool
    call belongs to a live session. The pending request is persisted with the
    conversation marker and resolved through ``answer_approval``.

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

    approve_command = None
    request_user_input = None
    ctx = None
    if runtime is not None and session_key:
        def request_user_input(title: str, prompt: str, **kwargs):
            """Handle request user input."""
            return runtime.request_input(session_key, title, prompt, **kwargs)

        def approve_command(command: str, justification: str) -> bool:
            """Approve command."""
            if ctx is not None:
                ctx.approval_denial_reason = ""
            session = getattr(runtime, "sessions", {}).get(session_key)
            # Consult opt-in permission gates registered by policy plugins
            # before the kernel's own logic. A gate may force allow/deny; None
            # from every gate means "no opinion — fall through".
            hooks = getattr(runtime, "hooks", None)
            if hooks is not None:
                verdict = hooks.vet_permission(session, current_tool_name, command)
                if verdict is not None:
                    if not verdict.allow and ctx is not None:
                        ctx.approval_denial_reason = verdict.reason
                    return verdict.allow
            if current_tool_name and current_tool_name in (effective_config.get("skip_permissions") or []):
                return True
            req = runtime.request_input(
                session_key,
                "Agent requests approval",
                f"{command}\n\n{justification}".strip(),
                type="boolean",
            )
            if not req.wait(timeout=300.0):
                req.metadata["timed_out"] = True
                runtime.answer_request(session_key, req.id, False)
                return False
            if req.metadata.get("cancelled"):
                return False
            return req.approved

    ctx = SecondBrainContext(
        db=db,
        config=effective_config,
        services=services,
        call_tool=call_tool_with_session if call_tool is not None else None,
        approve_command=approve_command,
        request_user_input=request_user_input,
        tool_registry=tool_registry,
        orchestrator=orchestrator,
        runtime=runtime,
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
