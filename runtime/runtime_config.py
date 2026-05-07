from __future__ import annotations

"""Per-session configuration: profile, scope, registry, system prompt, loop.

The runtime owns one global agent profile + tool registry, but each session
can override the profile and pin extra tools (subagent runs do this). The
helpers in this module compute the *effective* configuration for a given
session — the LLM to use, the tool registry the agent sees, the system
prompt that gets sent on every turn — and build the :class:`ConversationLoop`
that drives the agent's turn.

These functions are thin: they read from ``runtime`` and ``session`` and
return derived values. They never mutate persistence; they don't touch
session locks. Keep it that way.
"""

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
    if session is not None and session.profile_override:
        return session.profile_override
    return runtime.config.get("active_agent_profile") or "default"


def scope_for_profile(runtime, profile: str):
    try:
        from runtime.agent_scope import load_scope
        scope = load_scope(profile, runtime.config)
    except ValueError:
        return None
    return scope if scope.has_tool_filter or scope.prompt_suffix else None


def active_scope(runtime, session: RuntimeSession | None = None):
    return scope_for_profile(runtime, profile_for(runtime, session))


def active_tool_registry(runtime, session: RuntimeSession | None = None):
    """The tool registry as the agent in this session sees it.

    Layered: global registry → optional profile-scoped view → optional
    session-pinned tools (subagent NotifyTool, etc.). Returns the
    deepest layer applicable.
    """
    if not runtime.tool_registry:
        return None
    from runtime.agent_scope import scoped_registry
    scope = active_scope(runtime, session)
    registry = runtime.tool_registry
    if scope:
        registry = scoped_registry(runtime.tool_registry, scope, db=runtime.db)
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


def active_llm(runtime, session: RuntimeSession | None = None):
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
    commands = dict(runtime.commands)
    tools = tool_specs_for(runtime, session)
    cache = dict((marker or {}).get("cache") or {})
    if session:
        cache["session_key"] = session.key
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
    registries. Called when the active profile or registries change."""
    if not session.is_subagent and not session.profile_override:
        session.active_agent_profile = runtime.config.get("active_agent_profile") or "default"
    session.cs.participants["user"].commands = dict(runtime.commands)
    session.cs.participants["agent"].tools = tool_specs_for(runtime, session)


# ──────────────────────────────────────────────────────────────────────
# System prompt construction
# ──────────────────────────────────────────────────────────────────────

def session_system_prompt(runtime, session: RuntimeSession | None):
    """Return a system_prompt callable bound to this session.

    Subagent / profile-overridden sessions go through ``build_system_prompt``
    directly so the scoped registry, profile name, and notification mode
    feed into the prompt. Plain user sessions reuse the runtime's default
    system_prompt and append the session's ``system_prompt_extras`` —
    letting any plugin pin contextual snippets to the prompt without
    touching the bootstrap closure.
    """
    if session is None:
        return runtime.system_prompt

    if session.is_subagent or session.profile_override:
        from agent.system_prompt import build_system_prompt
        profile = session.profile_override or session.active_agent_profile or "default"
        scope = scope_for_profile(runtime, profile)
        registry = active_tool_registry(runtime, session)
        extras = session.system_prompt_extras or {}

        def _session_prompt():
            text = build_system_prompt(
                runtime.db,
                getattr(runtime, "_orchestrator_ref", None) or runtime.services.get("orchestrator"),
                registry, runtime.services,
                scope=scope,
                profile_name=profile,
                subagent_mode=extras.get("subagent_mode") if session.is_subagent else None,
                subagent_has_pending_messages=bool(extras.get("subagent_has_pending_messages")) if session.is_subagent else False,
            )
            for key, value in extras.items():
                if key in {"subagent_mode", "subagent_has_pending_messages"}:
                    continue
                if isinstance(value, str) and value:
                    text += "\n\n" + value
            return text
        return _session_prompt

    base = runtime.system_prompt
    extras = session.system_prompt_extras

    def _user_prompt():
        text = base() if callable(base) else (base or "")
        for value in (extras or {}).values():
            if isinstance(value, str) and value:
                text += "\n\n" + value
        return text
    return _user_prompt


# ──────────────────────────────────────────────────────────────────────
# Loop construction
# ──────────────────────────────────────────────────────────────────────

def build_loop(runtime, session_key: str | None = None) -> ConversationLoop:
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
        raise RuntimeError("LLM service is not loaded.")

    def notice(text: str):
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
    def started(name, call_id="tc_unknown", args=None):
        if runtime.on_tool_start:
            runtime.on_tool_start(name)
        if runtime.emit_event:
            runtime.emit_event(TOOL_CALL_STARTED, {
                "session_key": session_key, "call_id": call_id,
                "tool_name": name, "args": args or {},
            })

    def finished(name, call_id="tc_unknown", result=None, error=None):
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
    # LLM (see AttachmentBundle.for_llm in the LLM service layer).
    pointer = f"[Attached {attachment.modality} file: {file_name} (cached at {path})]"
    text = f"{caption}\n\n{pointer}".strip() if caption else pointer

    return {**content, "text": text, "attachment": attachment}
