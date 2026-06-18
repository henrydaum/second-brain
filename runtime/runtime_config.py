"""Per-session configuration: profile, scope, registry, system prompt, loop.

The runtime owns one global agent profile + tool registry, but each session
can override the profile and pin extra tools. The
helpers in this module compute the *effective* configuration for a given
session — the LLM to use, the tool registry the agent sees, the system
prompt that gets sent on every turn — and build the :class:`ConversationLoop`
that drives the agent's turn.

These functions are thin: they read from ``runtime`` and ``session`` and
return derived values. They never mutate persistence; they don't touch
session locks. Keep it that way.
"""

from __future__ import annotations


import logging
from pathlib import Path
from typing import Any, Callable

from state_machine.conversation import CallableSpec, ConversationState, Participant
from runtime.conversation_loop import ConversationLoop
from state_machine.conversation_phases import BASE_PHASE
from state_machine.forms import schema_to_form_steps
from runtime.session import RuntimeSession
from events.event_bus import bus
from events.event_channels import (
    CHAT_MESSAGE_PUSHED,
    TOOL_CALL_FINISHED,
    TOOL_CALL_STARTED,
)

logger = logging.getLogger("Runtime.config")


# ──────────────────────────────────────────────────────────────────────
# Profile / scope / registry / LLM resolution
# ──────────────────────────────────────────────────────────────────────

def profile_for(runtime, session: RuntimeSession | None) -> str:
    """Return the effective agent profile for a runtime session.

    Precedence: an explicit per-session override (``/agent switch``) wins, then
    the originating frontend's configured profile, then the global active
    profile. The frontend profile is a baseline, so a frontend can pin a
    restricted agent while ``/agent switch`` (when permitted) still overrides.
    """
    if session is not None and session.profile_override:
        return session.profile_override
    if session is not None and session.frontend_name:
        pinned = _frontend_agent_profile(runtime.config, session.frontend_name)
        if pinned:
            return pinned
    if session is not None and hasattr(runtime, "user_setting"):
        return runtime.user_setting(session.key, "active_agent_profile", "default") or "default"
    return runtime.config.get("active_agent_profile") or "default"


def _frontend_agent_profile(config: dict, frontend_name: str) -> str | None:
    """Return the agent profile a frontend pins, if any and it exists."""
    fp = (config.get("frontend_profiles") or {}).get(frontend_name) or {}
    name = fp.get("agent_profile")
    if not name or name == "default":
        return None
    if name in (config.get("agent_profiles") or {}):
        return name
    return None


def scope_for_profile(runtime, profile: str):
    """Load the configured tool/prompt scope for one agent profile."""
    try:
        from runtime.agent_scope import load_scope
        scope = load_scope(profile, runtime.config)
    except ValueError:
        return None
    return scope if scope.has_tool_filter or scope.prompt_suffix else None


def active_scope(runtime, session: RuntimeSession | None = None):
    """Return the effective scope after session overrides are applied."""
    return scope_for_profile(runtime, profile_for(runtime, session))


def active_tool_registry(runtime, session: RuntimeSession | None = None):
    """The tool registry as the agent in this session sees it.

    Layered: global registry → optional profile-scoped view → optional
    session-pinned tools. Returns the deepest layer applicable.
    """
    if not runtime.tool_registry:
        return None
    from runtime.agent_scope import scoped_registry
    scope = active_scope(runtime, session)
    registry = runtime.tool_registry
    if scope:
        registry = scoped_registry(runtime.tool_registry, scope, db=runtime.db)
    extras = list((session.extra_tool_instances if session else []) or [])
    # Cloning needs the real ToolRegistry shape (db/config/services). When
    # the runtime is wired with a stub registry (tests), extras can't be
    # plumbed through anyway — fall back to the base registry.
    if extras and hasattr(registry, "db") and hasattr(registry, "config") and hasattr(registry, "services"):
        from agent.tool_registry import ToolRegistry
        cloned = ToolRegistry(registry.db, registry.config, registry.services)
        cloned.orchestrator = getattr(registry, "orchestrator", None)
        cloned.runtime = getattr(registry, "runtime", None)
        cloned.tools.update(registry.tools)
        if getattr(registry, "visible_tool_names", None) is not None:
            cloned.visible_tool_names = set(registry.visible_tool_names)
        for tool in extras:
            cloned.tools[tool.name] = tool
            if cloned.visible_tool_names is not None:
                cloned.visible_tool_names.add(tool.name)
        registry = cloned
    # Opt-in scope shapers can add/hide tools per session. No-op when no shaper is registered.
    hooks = getattr(runtime, "hooks", None)
    if hooks is not None and session is not None:
        registry = hooks.shape_scope(session, registry)
    return registry


def active_llm(runtime, session: RuntimeSession | None = None):
    """Return the LLM service instance that should drive this session."""
    profile = profile_for(runtime, session)
    try:
        from runtime.agent_scope import resolve_agent_llm
        return resolve_agent_llm(profile, runtime.config, runtime.services)
    except Exception:
        return runtime.services.get("llm")


# ──────────────────────────────────────────────────────────────────────
# State-machine construction
# ──────────────────────────────────────────────────────────────────────

def new_state(
    runtime,
    marker: dict[str, Any] | None = None,
    session: RuntimeSession | None = None,
) -> ConversationState:
    """Build a fresh ConversationState from persisted markers and runtime wiring."""
    commands = dict(runtime.commands)
    tools = tool_specs_for(runtime, session)
    cache = dict((marker or {}).get("cache") or {})
    if session:
        cache["session_key"] = session.key
    cache["agent_scoped_tool_names"] = scoped_tool_names(runtime, session, tools)
    phase = (marker or {}).get("phase", BASE_PHASE)
    cs = ConversationState(
        [Participant("user", "user", commands=commands), Participant("agent", "agent", tools=tools)],
        (marker or {}).get("turn_priority", "user"),
        phase,
        cache,
        attachment_parser=lambda content: parse_attachment(runtime, content),
        attachment_lifecycle=runtime.config.get("attachment_lifecycle", "per_turn"),
    )
    # Restore persisted attachments (only present when lifecycle == "persistent"
    # and the marker was saved mid-conversation).
    from attachments.attachment import Attachment
    cs.pending_attachments = [
        Attachment.from_dict(a) if isinstance(a, dict) else a
        for a in (marker or {}).get("pending_attachments") or []
    ]
    return cs


def tool_specs_for(runtime, session: RuntimeSession | None = None) -> dict[str, CallableSpec]:
    """Expose direct tool calls as callable specs for ``/call``-style flows.

    ConversationLoop still uses the registry schemas directly when
    marshalling the agent's tool calls.
    """
    registry = active_tool_registry(runtime, session)
    if not registry:
        return {}
    specs = {}
    for schema in registry.get_all_schemas() or []:
        fn = schema.get("function", schema)
        name = fn.get("name")
        if name:
            specs[name] = CallableSpec(
                name,
                lambda cs, _actor, args, n=name, reg=registry: reg.call(n, _session_key=(cs.cache or {}).get("session_key"), **args),
                schema_to_form_steps(fn.get("parameters")),
            )
    return specs


def refresh_specs(runtime, session: RuntimeSession) -> None:
    """Re-bind the session's command/tool specs to the runtime's current
    registries. Called when the active profile or registries change.

    Also normalizes per-session notification mode.
    """
    if not session.profile_override:
        session.active_agent_profile = profile_for(runtime, session)
    from runtime.persistence import _sync_notification_mode
    _sync_notification_mode(session)
    session.cs.participants["user"].commands = dict(runtime.commands)
    tools = tool_specs_for(runtime, session)
    session.cs.participants["agent"].tools = tools
    session.cs.cache["agent_scoped_tool_names"] = scoped_tool_names(runtime, session, tools)


def scoped_tool_names(runtime, session: RuntimeSession | None, visible: dict[str, CallableSpec]) -> list[str]:
    """Return hidden-but-callable tool names that remain in the current scoped registry."""
    registry = active_tool_registry(runtime, session)
    if not registry or getattr(registry, "visible_tool_names", None) is None:
        return []
    return sorted(set(getattr(registry, "tools", {})) - set(visible))


# ──────────────────────────────────────────────────────────────────────
# System prompt construction
# ──────────────────────────────────────────────────────────────────────

def session_system_prompt(runtime, session: RuntimeSession | None):
    """Return a system_prompt callable bound to this session.

    The main bootstrap prompt can return sectioned system messages. Session
    metadata and plugin overlays are appended to the dynamic section so the
    static prefix remains cacheable.
    """
    if session is None:
        return runtime.system_prompt

    from runtime.notifications import notify_block

    def _notify_suffix() -> str:
        # Only meaningful when the session is not the user's currently
        # active conversation — otherwise notify is redundant with the
        # agent's regular output. Evaluated lazily inside the prompt
        # closure so it reflects the active session at turn time.
        """Internal helper to append notification guidance for background conversations."""
        if runtime.is_attended(session.key):
            return ""
        return notify_block(session.notification_mode)

    def _account_suffix() -> str:
        """Tell the agent which account it is assisting — but only when the
        session's user is a real account (has a username). Anonymous / base /
        guest sessions add nothing, so the line never becomes noise on
        single-operator frontends. Lazy + in the dynamic section so it never
        touches the cacheable static prefix."""
        if runtime.db is None:
            return ""
        user = runtime.db.get_user(runtime.session_user_id(session.key))
        username = (user or {}).get("username")
        return f'You are assisting the user "{username}".' if username else ""

    def _conversation_meta() -> dict[str, Any] | None:
        """Return current conversation metadata for the dynamic prompt."""
        return runtime.db.get_conversation(session.conversation_id) if runtime.db and session.conversation_id else None

    def _append_dynamic(prompt, *parts: str):
        """Append session-only text to the context-update section when present."""
        from agent.system_prompt import SYSTEM_CONTEXT_MARKER

        extra = "\n\n".join(p for p in parts if p)
        if not extra:
            return prompt
        if isinstance(prompt, list):
            out = [dict(m) for m in prompt]
            target = next(
                (m for m in reversed(out)
                 if (m.get("content") or "").lstrip().startswith(SYSTEM_CONTEXT_MARKER)),
                None,
            )
            if target is None:
                target = {"role": "user", "content": SYSTEM_CONTEXT_MARKER}
                out.append(target)
            target["content"] = (target.get("content") or "").rstrip() + "\n\n" + extra
            return out
        return (prompt or "") + "\n\n" + extra

    # A live session always builds a frontend- and profile-aware prompt: the
    # effective profile/scope shape the tool view, and the session's frontend
    # contributes its own guidance + command-policy filter. The frontend-agnostic
    # base prompt (runtime.system_prompt) is only the no-session fallback above.
    from agent.system_prompt import build_prompt_sections
    profile = profile_for(runtime, session)
    scope = scope_for_profile(runtime, profile)

    def _session_prompt():
        """Internal helper to handle session prompt."""
        frontend, command_filter = _session_frontend_filter(runtime, session)
        sections = build_prompt_sections(
            runtime.db,
            getattr(runtime, "_orchestrator_ref", None) or runtime.services.get("orchestrator"),
            active_tool_registry(runtime, session), runtime.services,
            scope=scope,
            profile_name=profile,
            commands=getattr(runtime, "command_registry", None) or runtime.commands,
            config=runtime.config,
            conversation_metadata=_conversation_meta(),
            prompt_extras=dict(session.system_prompt_extras or {}),
            notification_suffix=_notify_suffix(),
            frontend_name=session.frontend_name,
            frontend=frontend,
            command_filter=command_filter,
        )
        return _append_dynamic(sections, _account_suffix())
    return _session_prompt


def _session_frontend_filter(runtime, session):
    """Resolve the active frontend instance and its command-policy predicate.

    The frontend instance contributes its own ``agent_prompt``; the predicate
    filters the command catalog/statements to what the frontend's profile allows.
    """
    name = getattr(session, "frontend_name", None)
    manager = getattr(runtime, "frontend_manager", None)
    frontend = (getattr(manager, "adapters", {}) or {}).get(name) if (name and manager is not None) else None
    from plugins.frontends.helpers.command_registry import frontend_command_filter
    return frontend, frontend_command_filter(runtime.config, name)


# ──────────────────────────────────────────────────────────────────────
# Loop construction
# ──────────────────────────────────────────────────────────────────────

def build_loop(runtime, session_key: str | None = None) -> ConversationLoop:
    """Build loop."""
    session = runtime.sessions.get(session_key) if session_key else None
    llm = active_llm(runtime, session)
    if llm is None and hasattr(runtime, "llm"):
        llm = runtime.llm
    if llm is not None and not getattr(llm, "loaded", True) and hasattr(llm, "load"):
        try:
            llm.load()
        except Exception:
            logger.exception("Failed to load active LLM")
    if llm is not None and not getattr(llm, "loaded", True):
        router = runtime.services.get("llm")
        if router is not llm and getattr(router, "loaded", False):
            llm = router
    if llm is None or not getattr(llm, "loaded", True):
        raise RuntimeError(
            "No LLM is configured or loaded. Run /setup to configure one "
            "(or /llm to add a profile), then try again."
        )

    def notice(text: str):
        """Handle notice."""
        if runtime.on_notice:
            runtime.on_notice(text)
        if session_key:
            bus.emit(CHAT_MESSAGE_PUSHED, {
                "session_key": session_key, "message": text,
                "source": "runtime", "kind": "alert",
            })

    started, finished = tool_callbacks(runtime, session_key)
    return ConversationLoop(
        llm,
        active_tool_registry(runtime, session),
        runtime.config,
        session_system_prompt(runtime, session),
        started, finished, notice,
        session.cancel_event if session else None,
        runtime=runtime,
        session_key=session_key,
    )


def tool_callbacks(runtime, session_key: str | None):
    """Handle tool callbacks."""
    def started(name, call_id="tc_unknown", args=None):
        """Handle started."""
        if runtime.on_tool_start:
            runtime.on_tool_start(name)
        if runtime.emit_event:
            runtime.emit_event(TOOL_CALL_STARTED, {
                "session_key": session_key, "call_id": call_id,
                "tool_name": name, "args": args or {},
            })

    def finished(name, call_id="tc_unknown", result=None, error=None):
        """Handle finished."""
        tool_result = (getattr(result, "data", None) or {}).get("result") if result else None
        ok = bool(result and getattr(result, "ok", False) and getattr(tool_result, "success", True) and not error)
        err = error or getattr(getattr(result, "error", None), "message", None) or getattr(tool_result, "error", None)
        if runtime.on_tool_result:
            runtime.on_tool_result(name, tool_result)
        if runtime.emit_event:
            runtime.emit_event(TOOL_CALL_FINISHED, {
                "session_key": session_key, "call_id": call_id,
                "tool_name": name, "ok": ok, "error": err,
            })

    return started, finished


# ──────────────────────────────────────────────────────────────────────
# Misc setup helpers
# ──────────────────────────────────────────────────────────────────────

def command_specs_from_dicts(specs: dict[str, dict]) -> dict[str, CallableSpec]:
    """Handle command specs from dicts."""
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


def parse_attachment(runtime, content: dict[str, Any]) -> dict[str, Any]:
    """Build an Attachment from a SendAttachment payload using the
    runtime's services, then return a dict carrying both the dataclass
    and the user-facing text the dispatch layer should record in
    history. The Attachment itself is what flows to the LLM."""
    from attachments import parse_attachment as build_attachment

    path = Path(str(content.get("path") or ""))
    file_name = content.get("file_name") or path.name or "attachment"
    caption = str(content.get("caption") or content.get("text") or "").strip()

    attachment = build_attachment(
        str(path),
        file_name=file_name,
        services=runtime.services,
        config={"max_chars": 4000},
    )

    # History row text: caption plus a short pointer line so future
    # replays of the conversation still know a file existed. The full
    # parsed-text blurb is added to the prompt only when we hit the
    # LLM (see AttachmentBundle.split_for_llm in the LLM service layer).
    pointer = f"[Attached {attachment.modality} file: {file_name} (cached at {path})]"
    text = f"{caption}\n\n{pointer}".strip() if caption else pointer

    return {**content, "text": text, "attachment": attachment}
