"""Per-session extension hooks — the opt-in substrate for on-demand plugins.

The kernel makes three decisions per session that a plugin might want to bend:
whether to allow a sensitive tool call (permission), which tools the agent can
see (scope), and what extra text rides in the system prompt. The first two are
hardcoded `if` branches in core today; this registry turns them into passive
hook points so a plugin can opt in without core knowing the plugin exists.

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


class HookRegistry:
    """The whole on-demand extension surface, hung off the runtime as ``runtime.hooks``."""

    def __init__(self):
        """Initialize an empty registry."""
        self._permission_gates: list[PermissionGate] = []
        self._scope_shapers: list[ScopeShaper] = []
        self._turn_finalizers: list[TurnFinalizer] = []

    # --- registration (called by plugins at load) ---

    def add_permission_gate(self, gate: PermissionGate) -> None:
        """Register a gate consulted before the kernel's own permission logic."""
        self._permission_gates.append(gate)

    def add_scope_shaper(self, shaper: ScopeShaper) -> None:
        """Register a shaper that can add/hide tools for a session's registry."""
        self._scope_shapers.append(shaper)

    def add_turn_finalizer(self, finalizer: TurnFinalizer) -> None:
        """Register a callback run after each agent turn."""
        self._turn_finalizers.append(finalizer)

    def remove(self, fn: Callable) -> None:
        """Drop a previously registered gate or shaper (for plugin unload)."""
        for bucket in (self._permission_gates, self._scope_shapers, self._turn_finalizers):
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
