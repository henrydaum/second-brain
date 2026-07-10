"""Escalation extension: a cheap-model cascade.

Any session whose effective LLM is *not* the configured strong model gets an
``escalate`` tool. Calling it ends the current drive, and the runtime
immediately re-drives the turn on the strong model (the kernel's
``session.restart_turn`` primitive + the ``llm_selector`` hook). The weak
model's partial turn — including the escalate call and its reason — stays in
history, so the strong model sees what was attempted. Escalation lasts one
turn: the next user message resolves the session's normal model again.

Only the strong model's profile name is configured; every other model is
implicitly "weak". The strong model itself never sees the tool, so it can
never escalate.
"""

from __future__ import annotations

dependencies_files = []
dependencies_pip = []

import logging

from plugins.BaseService import BaseService, EXTENSION
from plugins.BaseTool import BaseTool, ToolResult

logger = logging.getLogger("Escalate")

PLUGIN = "escalate"


def state(session) -> dict:
    """Return the escalate state bag for a session."""
    bag = getattr(session, "plugin_state", None)
    return bag.setdefault(PLUGIN, {}) if bag is not None else {}


def escalation_pending(session) -> bool:
    """Whether this session's next turn should run on the strong model."""
    return bool(state(session).get("pending"))


class EscalateTool(BaseTool):
    """Hand the current turn to the configured strong model."""

    name = "escalate"
    description = (
        "Hand this turn to a stronger model. Use when the request is beyond "
        "your capability: complex reasoning or math, high-stakes or subtle "
        "writing, intricate multi-step tool work, or anything you have "
        "already attempted and gotten wrong. The stronger model immediately "
        "retakes this turn with the full conversation, including your "
        "partial work. Prefer escalating over guessing — a wasted "
        "escalation is cheap, a wrong answer is not. Do not call it for "
        "requests you can handle."
    )
    parameters = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "One line on why this needs the stronger model.",
            },
        },
        "required": [],
    }
    requires_services = []
    max_calls = 1
    background_safe = True

    def __init__(self, service: "EscalateService"):
        """Initialize with the owning service (for config access)."""
        self._service = service

    def run(self, context, **kwargs) -> ToolResult:
        """Flag the session for escalation and request a turn restart."""
        runtime = getattr(context, "runtime", None)
        session = getattr(runtime, "sessions", {}).get(getattr(context, "session_key", None)) if runtime else None
        if runtime is None or session is None:
            return ToolResult.failed("Escalation unavailable: no live session.")
        strong = self._service.strong_name()
        if not strong or runtime.services.get(strong) is None:
            return ToolResult.failed(
                "Escalation unavailable: no strong model is configured "
                "(escalate_strong_llm). Answer as best you can."
            )
        reason = (kwargs.get("reason") or "").strip()
        runtime.update_session_plugin_state(session.key, PLUGIN, {"pending": True})
        session.restart_turn = True
        logger.info(f"Session {session.key!r} escalating to {strong!r}: {reason or '(no reason)'}")
        summary = f"Escalating this turn to {strong}."
        if reason:
            summary += f" Reason: {reason}"
        return ToolResult(data={"strong": strong, "reason": reason}, llm_summary=summary)


class EscalateService(BaseService):
    """Registers the escalate scope shaper, LLM selector, and turn finalizer."""

    model_name = "Escalate"
    shared = True
    lifecycle = EXTENSION

    config_settings = [
        ("Strong LLM Profile", "escalate_strong_llm",
         "llm_profiles key of the model escalated turns run on. Every other "
         "model gets the escalate tool; leave empty to disable escalation.",
         "", {"type": "text"}),
    ]

    def __init__(self, config=None):
        """Initialize the escalate service."""
        super().__init__()
        self.config = config if config is not None else {}
        self.runtime = None
        self._registered = False
        # Session keys whose pending escalation was actually served (the
        # selector answered during a busy drive). The finalizer only clears
        # ``pending`` for served sessions, so the flag survives the truncated
        # weak drive and dies after the strong one. In-memory on purpose.
        self._served: set[str] = set()

    # --- lifecycle / hook registration (plan-mode pattern) ---

    def bind_runtime(self, *, runtime=None, **_):
        """Receive runtime binding and register hooks if already loaded."""
        self.runtime = runtime
        if self.loaded:
            self._register()

    def _load(self) -> bool:
        """Load the extension and register hooks when runtime is available."""
        self.loaded = True
        self._register()
        return True

    def unload(self):
        """Remove hooks."""
        self._unregister()
        self.loaded = False

    def _register(self):
        hooks = getattr(self.runtime, "hooks", None) if self.runtime else None
        if hooks is None or self._registered:
            return
        hooks.add_scope_shaper(self._scope_shaper)
        hooks.add_llm_selector(self._llm_selector)
        hooks.add_turn_finalizer(self._turn_finalizer)
        self._registered = True

    def _unregister(self):
        hooks = getattr(self.runtime, "hooks", None) if self.runtime else None
        if hooks is not None:
            hooks.remove(self._scope_shaper)
            hooks.remove(self._llm_selector)
            hooks.remove(self._turn_finalizer)
        self._registered = False
        self._served.clear()

    # --- config ---

    def strong_name(self) -> str:
        """The configured strong model's profile/service name ('' = disabled)."""
        return (self.config.get("escalate_strong_llm") or "").strip()

    # --- hooks ---

    def _scope_shaper(self, session, registry):
        """Offer the escalate tool to every session not already on the strong model."""
        strong = self.strong_name()
        runtime = self.runtime
        if not strong or runtime is None or runtime.services.get(strong) is None:
            return registry
        if self._effective_llm_is_strong(session, strong):
            return registry
        from runtime.agent_scope import registry_with_tools
        return registry_with_tools(registry, [EscalateTool(self)])

    def _llm_selector(self, session, runtime):
        """Answer the strong model while this session's escalation is pending."""
        strong = self.strong_name()
        if not strong or not escalation_pending(session):
            return None
        # Only count the escalation as served when a real drive asks (busy is
        # set for the whole of _drive_agent_turn); stray active_llm callers
        # (e.g. /debug) must not burn the pending flag.
        if getattr(session, "busy", False):
            self._served.add(getattr(session, "key", None))
        return strong

    def _turn_finalizer(self, session):
        """Clear a pending escalation once its strong turn has run."""
        key = getattr(session, "key", None)
        if key in self._served:
            self._served.discard(key)
            if self.runtime is not None:
                self.runtime.update_session_plugin_state(key, PLUGIN, {"pending": False})
            else:
                state(session)["pending"] = False

    # --- helpers ---

    def _effective_llm_is_strong(self, session, strong: str) -> bool:
        """Whether profile resolution already lands this session on the strong model."""
        runtime = self.runtime
        try:
            from runtime.runtime_config import active_llm
            effective = active_llm(runtime, session)
        except Exception:
            return False
        target = runtime.services.get(strong)
        if effective is None or target is None:
            return False
        # The default-LLM router is a proxy; compare against what it resolves to.
        if effective is runtime.services.get("llm"):
            effective = getattr(effective, "active", None) or effective
        return effective is target

    def debug_flags(self, session) -> list[str]:
        """Human-readable status flags for debug surfaces."""
        return ["escalation pending"] if escalation_pending(session) else []


def build_services(config) -> dict:
    """Build the escalate service."""
    return {"escalate": EscalateService(config)}
