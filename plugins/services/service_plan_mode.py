"""Plan mode runtime extension.

This service is intentionally thin: Plan mode is mostly helper functions, but
services are the plugin family that receive runtime binding and can register
hooks.
"""

from __future__ import annotations

from events.event_channels import CHAT_MESSAGE_PUSHED
from events.event_bus import bus
from plugins.BaseService import BaseService
from runtime.hooks import PermissionVerdict

PLUGIN = "plan_mode"
DENIED = "Permission dialogs automatically rejected in plan mode."
PROMPT = (
    "## Plan mode\n"
    "Plan mode is active. Permission dialogs are automatically rejected in plan mode. "
    "Inspect, read, search, and ask the user questions as needed, but do not try to modify state. "
    "When ready, call propose_plan with a concise plan for the user to approve. "
    "The user can approve normally, approve and auto-approve permission dialogs for this turn, or deny the plan."
)


def state(session) -> dict:
    """Return the Plan-mode state bag for a session."""
    bag = getattr(session, "plugin_state", None)
    return bag.setdefault(PLUGIN, {}) if bag is not None else {}


def enabled(session) -> bool:
    """Whether Plan mode is active for a session."""
    return bool(state(session).get("enabled"))


def full_permissions_this_turn(session) -> bool:
    """Whether the approved plan granted temporary permission bypass."""
    return bool(getattr(session, "busy", False) and state(session).get("full_permissions_this_turn"))


def plan_permission_gate(session, _tool_name, _command):
    """Reject permission dialogs in Plan mode; honor one active-turn override."""
    if session is None:
        return None
    if enabled(session):
        return PermissionVerdict(False, DENIED)
    if full_permissions_this_turn(session):
        return PermissionVerdict(True)
    return None


def plan_turn_finalizer(session) -> None:
    """Clear one-turn permission grants when the active agent turn ends."""
    state(session).pop("full_permissions_this_turn", None)


def make_plan_scope_shaper(tool_factory):
    """Build a shaper that injects the plan approval tool while Plan mode is active."""
    def plan_scope_shaper(session, registry):
        if not enabled(session):
            return registry
        from runtime.agent_scope import registry_with_tools
        return registry_with_tools(registry, [tool_factory()])
    return plan_scope_shaper


class PlanModeService(BaseService):
    """Registers Plan-mode hooks and owns Plan-mode session state."""

    model_name = "Plan Mode"
    shared = True

    def __init__(self, _config=None):
        super().__init__()
        self.runtime = None
        self._scope_shaper = None
        self._registered = False

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
        """Remove hooks and prompt overlays."""
        self._unregister()
        self.loaded = False

    def set_enabled(self, session_key: str, value: bool, message: str | None = None) -> bool:
        """Toggle Plan mode for one session."""
        runtime = self.runtime
        session = getattr(runtime, "sessions", {}).get(session_key) if runtime else None
        if runtime is None or session is None:
            return False
        old = enabled(session)
        runtime.update_session_plugin_state(session_key, PLUGIN, {"enabled": bool(value)})
        if value:
            runtime.add_system_prompt_extra(session_key, PLUGIN, PROMPT)
        else:
            runtime.remove_system_prompt_extra(session_key, PLUGIN)
        if old != bool(value):
            bus.emit(CHAT_MESSAGE_PUSHED, {"session_key": session_key, "message": message or f"Plan mode {'on' if value else 'off'}."})
        runtime.refresh_session_specs()
        return True

    def is_enabled(self, session) -> bool:
        """Whether Plan mode is active for a session."""
        return enabled(session)

    def has_full_permissions_this_turn(self, session) -> bool:
        """Whether the temporary permission bypass is active."""
        return full_permissions_this_turn(session)

    def approve(self, session_key: str, *, full_permissions: bool = False, message: str | None = None) -> bool:
        """Approve a proposed plan and optionally grant one active-turn bypass."""
        ok = self.set_enabled(session_key, False, message=message)
        if ok and full_permissions and self.runtime:
            self.runtime.update_session_plugin_state(session_key, PLUGIN, {"full_permissions_this_turn": True})
        return ok

    def debug_flags(self, session) -> list[str]:
        """Return human-readable status flags for debug surfaces."""
        return [
            label for label, active in (
                ("plan mode", enabled(session)),
                ("full permissions this turn", full_permissions_this_turn(session)),
            ) if active
        ]

    def _register(self):
        runtime = self.runtime
        hooks = getattr(runtime, "hooks", None) if runtime else None
        if hooks is None or self._registered:
            return
        hooks.add_permission_gate(plan_permission_gate)
        hooks.add_turn_finalizer(plan_turn_finalizer)
        try:
            from plugins.tools.tool_propose_plan import ProposePlan
        except ImportError:
            ProposePlan = None
        if ProposePlan is not None:
            self._scope_shaper = make_plan_scope_shaper(ProposePlan)
            hooks.add_scope_shaper(self._scope_shaper)
        self._registered = True
        for key, session in getattr(runtime, "sessions", {}).items():
            if enabled(session):
                runtime.add_system_prompt_extra(key, PLUGIN, PROMPT)
        runtime.refresh_session_specs()

    def _unregister(self):
        runtime = self.runtime
        hooks = getattr(runtime, "hooks", None) if runtime else None
        if hooks is not None:
            hooks.remove(plan_permission_gate)
            hooks.remove(plan_turn_finalizer)
            if self._scope_shaper is not None:
                hooks.remove(self._scope_shaper)
        for key in list(getattr(runtime, "sessions", {}) or {}):
            runtime.remove_system_prompt_extra(key, PLUGIN)
        if runtime is not None:
            runtime.refresh_session_specs()
        self._scope_shaper = None
        self._registered = False


def build_services(config) -> dict:
    """Build the Plan-mode service."""
    return {"plan_mode": PlanModeService(config)}
