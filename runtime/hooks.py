"""Per-session extension hooks — the opt-in substrate for on-demand plugins.

The kernel makes several decisions per session that a plugin might want to bend:
whether to allow a sensitive tool call (permission), which tools the agent can
see (scope), and which files should ride along with the next model call.

A plugin registers from its service ``_load()`` via ``runtime.hooks.add_*``.
A plugin that never touches ``runtime.hooks`` behaves exactly as before — the
registry is empty and the kernel falls through to its own defaults. Nothing is
added to the BaseTool / BaseCommand / BaseService contract.

(System-prompt extras need no hook here: ``session.system_prompt_extras``
already exists and is already appended on every turn — a plugin just writes
into that dict.)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("Hooks")


class PermissionVerdict:
    """A gate's answer to "may this command run?".

    ``allow`` is the decision; ``reason`` is the model-facing explanation shown
    when a call is denied. A gate that has no opinion returns ``None`` instead
    of a verdict, letting the next gate (or the kernel default) decide.
    """

    __slots__ = ("allow", "reason")

    def __init__(self, allow: bool, reason: str = ""):
        """Initialize the verdict."""
        self.allow = bool(allow)
        self.reason = reason


# A gate inspects the session and the pending command; it returns a verdict to
# decide, or None to abstain.
PermissionGate = Callable[[Any, Optional[str], str], Optional[PermissionVerdict]]
# A shaper receives the session and the registry the agent would otherwise see,
# and returns a (possibly replaced) registry.
ScopeShaper = Callable[[Any, Any], Any]
# A finalizer runs after an agent turn ends.
TurnFinalizer = Callable[[Any], None]
# A selector names the LLM service that should drive a session's next turn,
# or returns None to abstain (kernel profile resolution then applies).
LLMSelector = Callable[[Any, Any], Optional[str]]


class HookRegistry:
    """The whole on-demand extension surface, hung off the runtime as ``runtime.hooks``."""

    def __init__(self):
        """Initialize an empty registry."""
        self._permission_gates: list[PermissionGate] = []
        self._scope_shapers: list[ScopeShaper] = []
        self._turn_finalizers: list[TurnFinalizer] = []
        self._llm_selectors: list[LLMSelector] = []
        self._turn_attachments: dict[str, list[Any]] = {}

    # --- registration (called by plugins at load) ---

    def add_permission_gate(self, gate: PermissionGate) -> None:
        """Register a gate consulted before the kernel's own permission logic."""
        self._permission_gates.append(gate)

    def add_scope_shaper(self, shaper: ScopeShaper) -> None:
        """Register a shaper that can add/hide tools for a session's registry."""
        self._scope_shapers.append(shaper)

    def add_turn_finalizer(self, finalizer: TurnFinalizer) -> None:
        """Register a callback run after each agent turn.

        Contract wrinkle: finalizers run after every *drive*, and a turn
        restart (``session.restart_turn``) splits one logical turn into two
        drives — so a finalizer can fire mid-logical-turn, before the
        re-driven half runs. A finalizer that clears per-turn state must
        tolerate that (e.g. the store Escalate service only clears its flag
        once the re-driven half has been served). If a second plugin ever
        trips on this, teach the kernel to skip finalizers on restarting
        drives instead of adding plugin-side bookkeeping.
        """
        self._turn_finalizers.append(finalizer)

    def add_llm_selector(self, selector: LLMSelector) -> None:
        """Register a selector that can override which LLM drives a turn."""
        self._llm_selectors.append(selector)

    def stage_attachment(self, session, attachment: Any) -> bool:
        """Queue one attachment for the next LLM call in this session."""
        key = getattr(session, "key", None)
        if not key:
            return False
        self._turn_attachments.setdefault(key, []).append(attachment)
        return True

    def remove(self, fn: Callable) -> None:
        """Drop a previously registered gate or shaper (for plugin unload)."""
        for bucket in (self._permission_gates, self._scope_shapers, self._turn_finalizers, self._llm_selectors):
            try:
                bucket.remove(fn)
            except ValueError:
                pass

    # --- consultation (called by the kernel at its decision points) ---

    def vet_permission(self, session, tool_name: str | None, command: str) -> PermissionVerdict | None:
        """Return the first decisive verdict, or None if every gate abstains."""
        for gate in self._permission_gates:
            try:
                verdict = gate(session, tool_name, command)
            except Exception:
                logger.exception("Permission gate raised; treating as abstain")
                continue
            if verdict is not None:
                return verdict
        return None

    def select_llm(self, session, runtime) -> str | None:
        """Return the first selector's non-None LLM service name, or None."""
        for selector in self._llm_selectors:
            try:
                name = selector(session, runtime)
            except Exception:
                logger.exception("LLM selector raised; treating as abstain")
                continue
            if name:
                return name
        return None

    def shape_scope(self, session, registry):
        """Fold every shaper over the registry, in registration order."""
        for shaper in self._scope_shapers:
            try:
                registry = shaper(session, registry)
            except Exception:
                logger.exception("Scope shaper raised; leaving registry unchanged")
        return registry

    def finish_turn(self, session) -> None:
        """Run registered turn finalizers."""
        for finalizer in self._turn_finalizers:
            try:
                finalizer(session)
            except Exception:
                logger.exception("Turn finalizer raised; continuing")
        self._turn_attachments.pop(getattr(session, "key", None), None)

    def drain_attachments(self, session) -> list[Any]:
        """Return and clear attachments staged for the next model call."""
        return self._turn_attachments.pop(getattr(session, "key", None), [])
