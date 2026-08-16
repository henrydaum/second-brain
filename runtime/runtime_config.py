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
from runtime.conversation_loop import ConversationLoop, tool_summary
# ``runtime.hooks`` imports nothing from ``runtime``, so naming the moment
# rather than repeating the string costs no import-order risk.
from runtime.hooks import SHAPE_SCOPE
from state_machine.conversation_phases import BASE_PHASE
from state_machine.forms import schema_to_form_steps
from runtime.security_modes import YOLO, prompt_note
from runtime.session import RuntimeSession
from events.event_bus import bus
from events.event_channels import (
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
    if extras:
        cloned = _detached(registry)
        # Cloning needs the real ToolRegistry shape (db/config/services). When
        # the runtime is wired with a stub registry (tests), extras can't be
        # plumbed through anyway — fall back to the base registry.
        if cloned is not None:
            for tool in extras:
                cloned.tools[tool.name] = tool
                if cloned.visible_tool_names is not None:
                    cloned.visible_tool_names.add(tool.name)
            registry = cloned
    # Opt-in scope shapers can hide tools per session. No-op when no shaper is
    # registered — which is every install today, so the detach below is free.
    hooks = getattr(runtime, "hooks", None)
    if hooks is not None and session is not None and hooks.has(SHAPE_SCOPE):
        # Never hand a shaper the global registry. ``narrow_scope`` writes
        # ``visible_tool_names`` *in place*, and the layers above are all
        # conditional: with no profile scope and no pinned extras — the
        # ordinary case — ``registry`` is still ``runtime.tool_registry``
        # itself, the one object every session reads.
        #
        # That made a per-session decision permanent and process-wide, and it
        # ratcheted: ``narrow_scope`` intersects with the previous value, so a
        # shaper that legitimately varies its answer (the hook template's own
        # example keys on ``ctx.attended``, and one turn consults this doorway
        # 3 + one-per-model-call times) could narrow but never widen back. The
        # visible set walked toward empty and stayed there until restart.
        #
        # Detaching also restores the intersect to meaning what it was written
        # to mean — *this fold, between two shapers in one consultation* —
        # because each consultation now starts from the profile's own
        # visibility rather than from whatever the last one left behind.
        if registry is runtime.tool_registry:
            registry = _detached(registry) or registry
        registry = hooks.shape_scope(session, registry, runtime=runtime)
    return registry


def _detached(registry):
    """A copy of ``registry`` that can be narrowed without touching the original.

    Returns ``None`` when ``registry`` is not a real :class:`ToolRegistry` —
    the runtime is wired with a stub in several tests, and there is nothing to
    clone from one. Callers fall back to the registry they already had.

    Shallow on purpose: ``tools`` is copied so an entry can be added, and
    ``visible_tool_names`` is copied so it can be narrowed, but the tool
    *instances* are shared. A tool is stateless with respect to which session
    can see it.
    """
    if not all(hasattr(registry, attr) for attr in ("db", "config", "services")):
        return None
    from agent.tool_registry import ToolRegistry

    try:
        cloned = ToolRegistry(registry.db, registry.config, registry.services)
    except Exception:                                   # noqa: BLE001
        # A stub that carries the three attributes but is not constructible.
        # Answering None keeps the caller on the path it had before.
        logger.exception("could not detach the tool registry")
        return None
    cloned.orchestrator = getattr(registry, "orchestrator", None)
    cloned.runtime = getattr(registry, "runtime", None)
    cloned.tools.update(registry.tools)
    if getattr(registry, "visible_tool_names", None) is not None:
        cloned.visible_tool_names = set(registry.visible_tool_names)
    return cloned


def active_llm(runtime, session: RuntimeSession | None = None):
    """Return the LLM service instance that should drive this session.

    This is drive-time profile resolution only — the default brain. Plugins
    that want a different brain stand at the ``llm_call`` doorway and
    rewrite ``request.llm`` per call (see runtime/hooks.py), so their choice
    is invisible here and to any non-drive caller (e.g. /debug).
    """
    profile = profile_for(runtime, session)
    try:
        from runtime.agent_scope import resolve_agent_llm
        resolved = resolve_agent_llm(profile, runtime.config, runtime.services)
    except Exception:
        resolved = None
    if resolved is not None:
        return resolved
    # The registry knows nothing about a model somebody injected directly as
    # the ``llm`` service — a caller wiring its own, a test's fake. Those are
    # still legitimate brains, but nothing adapts them any more: whatever is
    # put here has to speak ``chat`` like a real one.
    return (runtime.services or {}).get("llm")


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
        attachment_lifecycle="persistent" if runtime.config.get("keep_attachments_available_across_turns") else "per_turn",
    )
    # Restore persisted attachments (only present when lifecycle == "persistent"
    # and the marker was saved mid-conversation).
    if session is not None:
        # Lets a command or tool body run outside the session lock the
        # dispatcher holds. Bound here because this is the one place a state
        # machine is built *with* a session behind it; one built without keeps
        # the no-op default and behaves as before.
        cs.unlocked = session.unlocked
        # And whether the person has pre-answered the approval a gated command
        # would raise. A closure over the live session rather than a value, so
        # a mode set mid-conversation takes effect on the next command instead
        # of at the next time something happens to rebuild the state machine.
        cs.auto_approve = lambda: runtime.security_mode(session.key) == YOLO
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
        bound = tool_spec(registry, schema)
        if bound is not None:
            specs[bound[0]] = bound[1]
    return specs


def tool_spec(registry, schema) -> tuple[str, CallableSpec] | None:
    """Bind one registry schema as a callable spec, or None if it has no name.

    Split out of ``tool_specs_for`` so a doorman's ``RequireTool`` can bind the
    single tool it demands. The participant's specs are built once per dispatch
    from the *visible* registry, and the forced call happens inside that
    dispatch — so a tool that became callable mid-turn has nowhere to be bound
    without this. See ``ConversationLoop._grant_required_tool``.
    """
    fn = schema.get("function", schema)
    name = fn.get("name")
    if not name:
        return None
    return name, CallableSpec(
        name,
        lambda cs, _actor, args, n=name, reg=registry: reg.call(n, _session_key=(cs.cache or {}).get("session_key"), **args),
        schema_to_form_steps(fn.get("parameters")),
    )


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

    def _mode_suffix() -> str:
        """Tell the agent how this conversation answers approval dialogs.

        Empty in ``ask``, which is the default and needs no saying. Lazy and
        in the dynamic section for the same reason as the two above — but here
        it is load-bearing rather than merely tidy: the mode changes *within* a
        conversation, so text baked into the cacheable prefix would go stale the
        moment somebody typed ``/mode``. A plugin's live ``agent_prompt`` now
        lands in this same section and refreshes on the same cadence, but the
        mode still does not belong to one: it is kernel state, and a safety
        surface that stops working when a package is uninstalled is worse than
        none.

        Worth telling it at all because a refusal it cannot explain is a
        refusal it retries. An agent that knows it is in lockdown reports the
        wall; one that does not looks for a way around it.
        """
        return prompt_note(runtime.security_mode(session.key))

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
            active_llm=active_llm(runtime, session),
        )
        return _append_dynamic(sections, _account_suffix(), _mode_suffix())
    return _session_prompt


def _session_frontend_filter(runtime, session):
    """Resolve the active frontend instance and its command-policy predicate.

    The frontend instance contributes its own ``agent_prompt``; the predicate
    filters the command catalog/statements to what the frontend's profile allows.
    """
    name = getattr(session, "frontend_name", None)
    manager = getattr(runtime, "frontend_manager", None)
    frontend = (getattr(manager, "adapters", {}) or {}).get(name) if (name and manager is not None) else None
    from plugins.command_registry import frontend_command_filter
    return frontend, frontend_command_filter(runtime.config, name)


def _frontend_streams(runtime, session) -> bool:
    """Whether the session's frontend can render a reply as it arrives.

    Asks the surface, because the surface is what knows. This was the global
    ``stream_responses`` setting, which put the question to the user — but a
    terminal that rewrites its last line can stream and a webhook that gets one
    POST per message cannot, and configuring changes neither. Worse, one global
    answer covered every frontend at once.

    ``FrontendCapabilities.supports_streaming`` is the declaration, and it was
    already there: ``_on_agent_text_delta`` has always gated *rendering* on it,
    so a frontend that could not stream was being sent deltas it discarded.
    Now the same flag gates emission, and the deltas are simply not produced.
    The LLM backend has the matching say via its own ``supports_streaming``;
    streaming needs both, and falls back to whole messages otherwise.

    Unresolvable frontends stream: a background driver or a test double has no
    surface to ask, and that is the behaviour every such caller had before.
    """
    if session is None:
        return True
    frontend, _filter = _session_frontend_filter(runtime, session)
    if frontend is None:
        return True
    return bool(getattr(frontend.capabilities, "supports_streaming", False))


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
        from llm import default_brain
        fallback = (default_brain(getattr(runtime, "config", None) or {})
                    or (runtime.services or {}).get("llm"))
        if fallback is not None and fallback is not llm and getattr(fallback, "loaded", False):
            llm = fallback
    if llm is None or not getattr(llm, "loaded", True):
        raise RuntimeError(
            "No LLM is configured or loaded. Run /setup to configure one "
            "(or /llm to add a profile), then try again."
        )

    def notice(text: str):
        """Tell the user what the loop is doing to their conversation.

        Compaction and overflow recovery: the turn is still running, but the
        history behind it just changed under the user's feet, which is worth
        interrupting for and is not something the agent said.

        ``persist=False`` — this is progress, not a record. "Compacting
        conversation…" is worth seeing while it happens and worth nothing an
        hour later, and a notification panel that fills up with them is one
        nobody reads.
        """
        if runtime.on_notice:
            runtime.on_notice(text)
        if not session_key:
            return
        # Guarded because this is called from inside compaction, mid-turn. The
        # module-level ``notify`` swallows its own failures, but reaching it
        # through the runtime does not — a background driver standing in for a
        # ConversationRuntime need not have the method, and losing the turn
        # because we could not narrate it is the wrong failure of the two.
        raise_notification = getattr(runtime, "notify", None)
        if raise_notification is None:
            return
        try:
            raise_notification(title="Conversation", body=text,
                               source="runtime", level="info",
                               session_key=session_key, persist=False)
        except Exception:
            logger.exception("could not raise a compaction notice (ignored)")

    on_delta = None
    if session_key and _frontend_streams(runtime, session):
        from events.event_channels import AGENT_TEXT_DELTA

        def on_delta(payload: dict):
            """Fan streamed text deltas out to frontends over the bus."""
            bus.emit(AGENT_TEXT_DELTA, {"session_key": session_key, **payload})

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
        on_delta=on_delta,
    )


def tool_blurb(raw) -> str:
    """The declared ``narration``, normalized once for every frontend.

    Collapsing whitespace and capping the length is *policy*, not styling, so
    it belongs here rather than in each renderer: a model is perfectly capable
    of writing a paragraph into a status line, and a frontend that forgot to
    cap it would wrap somebody's chat. Styling stays with the frontend, which
    is why this answers with a bare string and no markup.
    """
    text = " ".join(str(raw).split()) if raw else ""
    return f"{text[:77]}..." if len(text) > 80 else text


def tool_outcome(tool_result) -> str:
    """The finished event's ``summary``, or "" when there is nothing to say.

    Wraps ``conversation_loop.tool_summary`` in the two answers a *frontend*
    needs and the transcript does not. A tool with nothing to report gets an
    empty string rather than ``"null"``, since the alternative is a renderer
    drawing a heading over the word null. And a payload that will not
    serialize is dropped rather than raised: the transcript has to tell the
    model its result was lost, but here the tool genuinely succeeded, and
    failing the whole status event over an unprintable blob would take the
    ✓ down with it.
    """
    if tool_result is None:
        return ""
    # Neither half filled in. Checked on the fields rather than on the result,
    # because ``json.dumps(None)`` is the string "null" — which the transcript
    # keeps sending (an empty tool row is invalid to some providers) and a
    # person must never be shown.
    if (not getattr(tool_result, "llm_summary", "")
            and getattr(tool_result, "data", None) is None):
        return ""
    try:
        return tool_summary(tool_result, ConversationLoop.MAX_TOOL_RESULT_CHARS)
    except (TypeError, ValueError):
        logger.debug("tool result could not be summarized for frontends",
                     exc_info=True)
        return ""


def tool_callbacks(runtime, session_key: str | None):
    """Handle tool callbacks."""
    def started(name, call_id="tc_unknown", args=None):
        """Handle started."""
        if runtime.on_tool_start:
            runtime.on_tool_start(name)
        if runtime.emit_event:
            # ``narration`` is lifted out of ``args`` to the top level so both
            # events name it in the same place. It stays in ``args`` too, since
            # that is the model's verbatim call — but a frontend reading it
            # from there on one event and the top level on the other is a
            # difference nobody remembers, and getting it wrong shows up as a
            # blurb that renders until the tool returns and then disappears.
            runtime.emit_event(TOOL_CALL_STARTED, {
                "session_key": session_key, "call_id": call_id,
                "tool_name": name, "args": args or {},
                "narration": tool_blurb((args or {}).get("narration")),
            })

    def finished(name, call_id="tc_unknown", result=None, error=None, narration=None):
        """Handle finished."""
        tool_result = (getattr(result, "data", None) or {}).get("result") if result else None
        ok = bool(result and getattr(result, "ok", False) and getattr(tool_result, "success", True) and not error)
        err = error or getattr(getattr(result, "error", None), "message", None) or getattr(tool_result, "error", None)
        if runtime.on_tool_result:
            runtime.on_tool_result(name, tool_result)
        if runtime.emit_event:
            # ``narration`` is repeated here rather than left to the started
            # event: a frontend that overwrites its status line in place has
            # nothing left of the started payload by the time this lands.
            runtime.emit_event(TOOL_CALL_FINISHED, {
                "session_key": session_key, "call_id": call_id,
                "tool_name": name, "ok": ok, "error": err,
                "narration": tool_blurb(narration),
                "summary": tool_outcome(tool_result) if ok else "",
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
    runtime's services, then return a dict carrying the dataclass, the
    durable record of the file, and the user-facing text the dispatch
    layer should record in history. The Attachment itself is what flows
    to the LLM."""
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

    # The history row's *text* is what the person typed, and nothing else. The
    # pointer line that used to be welded on here — "[Attached image file: …]"
    # — is a rendering of the record for a model, and it is rendered at call
    # time from ``record`` instead. Welding it in put a machine-readable fact
    # into prose: a client could only get it back by parsing a sentence, and
    # a person who typed those same characters was indistinguishable from a
    # file. The full parsed-text blurb was never here either; it is added to
    # the prompt when we hit the LLM (AttachmentBundle.split_for_llm).
    return {**content, "text": caption, "attachment": attachment,
            "record": attachment.record()}
