from types import SimpleNamespace

from agent.tool_registry import ToolRegistry
from plugins.services.service_plan_mode import DENIED, PLUGIN, PlanModeService
from plugins.tools.tool_propose_plan import ProposePlan
from runtime.hooks import HookRegistry


class Runtime:
    def __init__(self):
        self.hooks = HookRegistry()
        self.sessions = {"chat": SimpleNamespace(plugin_state={}, system_prompt_extras={}, busy=False)}
        self.refreshed = 0

    def update_session_plugin_state(self, session_key, plugin, patch=None, **values):
        self.sessions[session_key].plugin_state.setdefault(plugin, {}).update({**(patch or {}), **values})
        return True

    def add_system_prompt_extra(self, session_key, key, value):
        self.sessions[session_key].system_prompt_extras[key] = value

    def remove_system_prompt_extra(self, session_key, key):
        self.sessions[session_key].system_prompt_extras.pop(key, None)

    def refresh_session_specs(self):
        self.refreshed += 1


def test_plan_service_owns_prompt_permission_and_turn_cleanup():
    runtime = Runtime()
    service = PlanModeService()
    service.bind_runtime(runtime=runtime)
    service.load()

    assert service.set_enabled("chat", True)
    session = runtime.sessions["chat"]
    assert session.plugin_state[PLUGIN]["enabled"] is True
    assert PLUGIN in session.system_prompt_extras
    verdict = runtime.hooks.vet_permission(session, "write_file", "write")
    assert verdict.allow is False
    assert verdict.reason == DENIED

    assert service.approve("chat", full_permissions=True)
    session.busy = True
    assert runtime.hooks.vet_permission(session, "write_file", "write").allow is True
    runtime.hooks.finish_turn(session)
    assert "full_permissions_this_turn" not in session.plugin_state[PLUGIN]


def test_plan_scope_shaper_exposes_propose_plan():
    runtime = Runtime()
    service = PlanModeService()
    service.bind_runtime(runtime=runtime)
    service.load()
    service.set_enabled("chat", True)
    registry = ToolRegistry(None, {"tool_timeout": 10})
    registry.visible_tool_names = set()

    shaped = runtime.hooks.shape_scope(runtime.sessions["chat"], registry)

    assert shaped is not registry
    assert shaped.tools["propose_plan"].__class__ is ProposePlan
    assert shaped.visible_tool_names == {"propose_plan"}
