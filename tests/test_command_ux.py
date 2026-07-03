"""Tests for command-UX polish: parse-error propagation, /services toggles,
config quicklinks, the Used-by map, and the session-conversation banner event.
"""

from types import SimpleNamespace

import state_machine  # noqa: F401  (import-order: break the runtime import cycle)

from events.event_bus import bus
from events.event_channels import SESSION_CONVERSATION_CHANGED
from pipeline.database import Database
from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities
from plugins.commands.command_services import ServicesCommand, _actions_for
from plugins.commands.helpers.setting_links import quicklink_run, quicklink_value_steps, quicklinks
from runtime.conversation_runtime import ConversationRuntime


# ── Invalid one-shot command args are rendered, not swallowed ────────

class _CaptureFrontend(BaseFrontend):
    name = "capture"
    capabilities = FrontendCapabilities()

    def __init__(self):
        super().__init__()
        self.rendered = []

    def render_messages(self, _key, messages):
        self.rendered.extend(messages)

    def _current_approval_request(self, _key):
        return None


def test_invalid_command_args_render_a_message():
    fe = _CaptureFrontend()
    fe.commands = SimpleNamespace(parse_args=lambda *a, **k: (_ for _ in ()).throw(
        ValueError("job_name must be one of: fifa_world_cup_daily_update, add.")))

    args, handled = fe._parse_command_args("s", "schedule", "fifa world cup daily update")

    assert args is None
    assert handled is not None and not handled.ok
    assert any("Invalid arguments for `/schedule`" in m for m in fe.rendered)
    assert any("fifa_world_cup_daily_update" in m for m in fe.rendered)


# ── /services toggles ────────────────────────────────────────────────

class _ManagedService:
    loaded = False
    lifecycle = "managed"
    config_settings = [("Embed Model", "embed_model_name", "Model.", "x", {"type": "text"})]

    def load(self):
        self.loaded = True
        return True

    def unload(self):
        self.loaded = False


def test_services_actions_are_toggles_with_quicklinks():
    svc = _ManagedService()
    context = SimpleNamespace(config={"autoload_services": []})

    actions, labels = _actions_for(context, "embedder", svc)

    assert actions[:2] == ["toggle_loaded", "toggle_autoload"]
    assert labels[:2] == ["Load it", "Autoload on startup"]
    assert actions[2] == "edit_setting:embed_model_name"
    assert labels[2] == "Edit Embed Model"

    svc.loaded = True
    context.config["autoload_services"] = ["embedder"]
    _, labels = _actions_for(context, "embedder", svc)
    assert labels[:2] == ["Unload it", "Don't autoload on startup"]


def test_toggle_loaded_flips_service_state():
    svc = _ManagedService()
    context = SimpleNamespace(services={"embedder": svc}, config={"autoload_services": []},
                              orchestrator=None)

    out = ServicesCommand().run({"service_name": "embedder", "action": "toggle_loaded"}, context)
    assert out == "Loaded service: embedder" and svc.loaded

    out = ServicesCommand().run({"service_name": "embedder", "action": "toggle_loaded"}, context)
    assert out == "Unloaded service: embedder" and not svc.loaded


def test_toggle_autoload_updates_config(monkeypatch):
    import plugins.commands.command_services as mod
    saved = {}
    monkeypatch.setattr("config.config_manager.save", lambda cfg: saved.update(cfg))
    svc = _ManagedService()
    context = SimpleNamespace(services={"embedder": svc}, config={"autoload_services": ["llm"]},
                              orchestrator=None, runtime=None)

    out = mod.ServicesCommand().run({"service_name": "embedder", "action": "toggle_autoload"}, context)

    assert "now" in out
    assert saved["autoload_services"] == ["embedder", "llm"]

    out = mod.ServicesCommand().run({"service_name": "embedder", "action": "toggle_autoload"}, context)
    assert "no longer" in out
    assert saved["autoload_services"] == ["llm"]


# ── Quicklinks ───────────────────────────────────────────────────────

def test_quicklinks_skip_hidden_and_missing_settings():
    class Tool:
        config_settings = [
            ("Visible", "vis_key", "d", 1, {"type": "text"}),
            ("Hidden", "hid_key", "d", 1, {"hidden": True}),
        ]
    values, labels = quicklinks(Tool())
    assert values == ["edit_setting:vis_key"]
    assert labels == ["Edit Visible"]
    assert quicklinks(None) == ([], [])
    assert quicklink_run("call", {}, None) is None
    assert quicklink_value_steps("call", None) == []


def test_quicklink_value_step_and_run_route_to_config(monkeypatch):
    steps = quicklink_value_steps("edit_setting:data_retention_days", None)
    assert len(steps) == 1 and steps[0].name == "value"

    monkeypatch.setattr("config.config_manager.save", lambda cfg: None)
    context = SimpleNamespace(config={"data_retention_days": 0}, db=None, user_id=1, runtime=None)
    out = quicklink_run("edit_setting:data_retention_days", {"value": "30"}, context)
    assert out == "Set data_retention_days = 30"
    assert context.config["data_retention_days"] == 30


# ── Used-by map ──────────────────────────────────────────────────────

def test_setting_plugin_names_accumulate_across_declarers():
    from plugins import plugin_discovery as pd

    class A:
        name = "tool_a"
        config_settings = [("Shared", "shared_key_ux_test", "d", 1, {"type": "text"})]

    class B:
        name = "svc_b"
        config_settings = [("Shared", "shared_key_ux_test", "d", 1, {"type": "text"})]

    pd._collect_config_settings(A(), plugin_type="tool")
    pd._collect_config_settings(B(), service_names=["svc_b"], plugin_type="service")
    try:
        assert pd.get_setting_plugin_names("shared_key_ux_test") == ["svc_b", "tool_a"]
    finally:
        pd._setting_to_plugins.pop("shared_key_ux_test", None)
        pd._setting_to_services.pop("shared_key_ux_test", None)


# ── Session conversation banner event ────────────────────────────────

def test_load_conversation_emits_session_conversation_changed(tmp_path):
    db = Database(str(tmp_path / "banner.db"))
    cid = db.create_conversation("FIFA Briefings")
    rt = ConversationRuntime(db=db, services={}, config={})
    seen = []
    unsub = bus.subscribe(SESSION_CONVERSATION_CHANGED, seen.append)
    try:
        rt.load_conversation("s", cid)
    finally:
        unsub()

    assert any(p["session_key"] == "s" and p["conversation_id"] == cid
               and p["title"] == "FIFA Briefings" for p in seen)
