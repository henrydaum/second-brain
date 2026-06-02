"""Plan-mode policy, expressed through the generic hook registry.

Step 1 of the plan-mode-as-plugin migration: the permission gate and the
tool-scope shaper that used to be hardcoded `if session.plan_mode` branches in
``context.py`` / ``runtime_config.py`` now live here and are *registered* on
``runtime.hooks``. Behavior is identical — this just proves the hook surface is
enough to express plan mode.

Step 2 will move this file verbatim into a ``plan/`` plugin bundle (registering
from ``service._load()`` instead of from bootstrap) and switch the session
fields it reads to ``session.plugin_state["plan"]``. Nothing else changes.
"""

from __future__ import annotations

from runtime.context import PLAN_MODE_PERMISSION_DENIED
from runtime.hooks import PermissionVerdict


def plan_permission_gate(session, tool_name, command):
    """Auto-reject sensitive calls while drafting a plan; honor a one-turn override."""
    if getattr(session, "plan_mode", False):
        return PermissionVerdict(False, PLAN_MODE_PERMISSION_DENIED)
    if getattr(session, "full_permissions_this_turn", False):
        return PermissionVerdict(True)
    return None


def plan_scope_shaper(session, registry):
    """Inject ``propose_plan`` into the agent's registry while plan mode is on."""
    if not getattr(session, "plan_mode", False):
        return registry
    # Cloning needs the real ToolRegistry shape (db/config/services). When the
    # runtime is wired with a stub registry (tests), leave it untouched.
    if not (hasattr(registry, "db") and hasattr(registry, "config") and hasattr(registry, "services")):
        return registry
    from agent.tool_registry import ToolRegistry
    from plugins.tools.tool_propose_plan import ProposePlan
    cloned = ToolRegistry(registry.db, registry.config, registry.services)
    cloned.orchestrator = getattr(registry, "orchestrator", None)
    cloned.runtime = getattr(registry, "runtime", None)
    cloned.tools.update(registry.tools)
    if getattr(registry, "visible_tool_names", None) is not None:
        cloned.visible_tool_names = set(registry.visible_tool_names)
        cloned.visible_tool_names.add("propose_plan")
    cloned.tools["propose_plan"] = ProposePlan()
    return cloned


def register_plan_mode_hooks(runtime) -> None:
    """Wire plan mode's gate + shaper onto the runtime's hook registry."""
    hooks = getattr(runtime, "hooks", None)
    if hooks is None:
        return
    hooks.add_permission_gate(plan_permission_gate)
    hooks.add_scope_shaper(plan_scope_shaper)
