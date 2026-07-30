"""Tests for the config layer (``config_manager`` + ``config_data``).

The kernel derives its defaults from the single ``SETTINGS_DATA`` source of
truth. These tests pin the kernel-minimal defaults and the load/save behaviour
that keeps an on-disk ``config.json`` in sync with the schema, using a temp
path so the real DATA_DIR config is never touched.
"""

import json

from config import config_manager
from config.config_data import DEFAULT_SCHEDULED_JOBS, SETTINGS_DATA


def _cfg(tmp_path):
    return str(tmp_path / "config.json")


# ── Kernel-minimal defaults ──────────────────────────────────────────

def test_kernel_defaults_are_minimal():
    """The kernel ships REPL plus the LLM router and Timekeeper, with no jobs."""
    assert config_manager.DEFAULTS["autoload_services"] == ["timekeeper"]
    assert config_manager.DEFAULTS["enabled_frontends"] == ["repl"]
    assert DEFAULT_SCHEDULED_JOBS == {}
    assert config_manager.DEFAULTS["scheduled_jobs"] == {}
    assert config_manager.DEFAULTS["keep_attachments_available_across_turns"] is False


def test_defaults_cover_every_settings_entry():
    names = {entry[1] for entry in SETTINGS_DATA}
    assert set(config_manager.DEFAULTS) == names


# ── load() ───────────────────────────────────────────────────────────

def test_load_creates_default_config_when_missing(tmp_path):
    path = _cfg(tmp_path)
    config = config_manager.load(path)

    assert config["enabled_frontends"] == ["repl"]
    # The file is written so subsequent loads are stable.
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["autoload_services"] == ["timekeeper"]


def test_load_merges_missing_keys_and_persists(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"max_workers": 8}))

    config = config_manager.load(path)

    assert config["max_workers"] == 8  # user value preserved
    assert config["enabled_frontends"] == ["repl"]  # default filled in
    # Schema drift is healed on disk, not just in memory.
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "enabled_frontends" in on_disk


def test_load_removes_deprecated_llm_service_autoload(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "autoload_services": ["llm", "timekeeper"],
    }))

    config = config_manager.load(path)

    assert config["autoload_services"] == ["timekeeper"]
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["autoload_services"] == ["timekeeper"]


def test_load_strips_user_config_keys_from_disk(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "last_active_conversation_id": 12,
        "active_agent_profile": "builder",
        "skip_permissions": ["run_command"],
    }))

    config = config_manager.load(path)

    assert "last_active_conversation_id" not in config
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "last_active_conversation_id" not in on_disk
    assert "active_agent_profile" not in on_disk
    assert "skip_permissions" not in on_disk


def test_load_normalizes_enabled_frontends(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "enabled_frontends": ["REPL", "telegram", "mcp_server", "repl", ""]
    }))

    config = config_manager.load(path)

    # Lowercased, deduped, empties dropped, order preserved. Unknown names
    # SURVIVE on purpose: frontends are store packages, so the kernel cannot
    # know the valid set — an installed frontend's name must not be stripped
    # by config load. Bootstrap warns and skips what discovery can't resolve.
    assert config["enabled_frontends"] == ["repl", "telegram", "mcp_server"]


def test_load_coerces_scalar_list_key_to_list(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"sync_directories": "C:/notes"}))

    config = config_manager.load(path)

    assert config["sync_directories"] == ["C:/notes"]


# ── save() ───────────────────────────────────────────────────────────

def test_save_strips_root_and_persists_known_keys(tmp_path):
    path = _cfg(tmp_path)
    config_manager.save({"max_workers": 12, "_root": "/somewhere"}, path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["max_workers"] == 12
    assert "_root" not in on_disk
    # Defaults are merged in so the file is always complete.
    assert on_disk["enabled_frontends"] == ["repl"]


def test_save_preserves_existing_unrelated_values(tmp_path):
    path = _cfg(tmp_path)
    config_manager.save({"max_workers": 8}, path)
    config_manager.save({"poll_interval": 2.0}, path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["max_workers"] == 8
    assert on_disk["poll_interval"] == 2.0


def test_save_excludes_plugin_config_keys_even_when_unregistered(tmp_path, monkeypatch):
    """Plugin values loaded into the runtime config from plugin_config.json (for
    plugins not yet discovered) must never be duplicated into core config.json."""
    plugin_path = str(tmp_path / "plugin_config.json")
    (tmp_path / "plugin_config.json").write_text(json.dumps({"telegram_bot_token": "secret"}))
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH", plugin_path)

    path = _cfg(tmp_path)
    config_manager.save({"max_workers": 8, "telegram_bot_token": "secret"}, path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["max_workers"] == 8
    assert "telegram_bot_token" not in on_disk


def test_save_strips_user_config_keys(tmp_path):
    path = _cfg(tmp_path)
    config_manager.save({
        "last_active_conversation_id": 12,
        "active_agent_profile": "builder",
        "skip_permissions": ["run_command"],
    }, path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "last_active_conversation_id" not in on_disk
    assert "active_agent_profile" not in on_disk
    assert "skip_permissions" not in on_disk


def test_load_plugin_config_repairs_trailing_data(tmp_path):
    path = str(tmp_path / "plugin_config.json")
    (tmp_path / "plugin_config.json").write_text('{"one": 1}\n{"stale": 2}', encoding="utf-8")

    loaded = config_manager.load_plugin_config(path)

    assert loaded == {"one": 1}
    assert json.loads((tmp_path / "plugin_config.json").read_text(encoding="utf-8")) == {"one": 1}


def test_save_plugin_config_uses_atomic_temp_file(tmp_path):
    path = str(tmp_path / "plugin_config.json")

    config_manager.save_plugin_config({"one": 1}, path)

    assert json.loads((tmp_path / "plugin_config.json").read_text(encoding="utf-8")) == {"one": 1}
    assert not list(tmp_path.glob("plugin_config.json.tmp-*"))


# ── Settings that graduated from a plugin into the kernel ────────────
#
# A capability absorbed into the kernel leaves its settings behind. That was
# not cosmetic: ``save`` used to treat "already in plugin_config.json" as
# ownership, so an undeclared key's home was an accident of history, and a
# key re-homed into SETTINGS_DATA could never reach the file that now owned
# it. ``llm_profiles`` lived exactly there and users lost their model config.

def test_a_kernel_declaration_beats_an_existing_plugin_config_entry(
        tmp_path, monkeypatch):
    """Otherwise the re-homed key can never reach config.json."""
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    config_manager.save_plugin_config({"llm_profiles": {"old": {}}},
                                      plugin_path)

    path = _cfg(tmp_path)
    config_manager.save({"llm_profiles": {"new": {}}}, path)

    assert json.loads(open(path).read())["llm_profiles"] == {"new": {}}


def test_rehoming_moves_the_value_and_empties_the_old_home(tmp_path,
                                                           monkeypatch):
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    config_manager.save_plugin_config(
        {"llm_profiles": {"m": {"llm_endpoint": "x"}},
         "default_llm_profile": "m",
         "some_plugin_setting": 7}, plugin_path)

    runtime = {}
    moved = config_manager.rehome_kernel_keys(runtime)

    assert moved == ["default_llm_profile", "llm_profiles"]
    # The new home has them...
    core = json.loads(open(_cfg(tmp_path)).read())
    assert core["llm_profiles"] == {"m": {"llm_endpoint": "x"}}
    assert core["default_llm_profile"] == "m"
    # ...the old one does not, and an unrelated plugin key is untouched.
    remaining = json.loads(open(plugin_path).read())
    assert remaining == {"some_plugin_setting": 7}
    # And the caller's live config can see them immediately.
    assert runtime["llm_profiles"] == {"m": {"llm_endpoint": "x"}}


def test_rehoming_is_idempotent(tmp_path, monkeypatch):
    """It runs on every boot, so a second pass must find nothing to do."""
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    config_manager.save_plugin_config({"llm_profiles": {"m": {}}}, plugin_path)

    assert config_manager.rehome_kernel_keys({}) == ["llm_profiles"]
    assert config_manager.rehome_kernel_keys({}) == []
    assert json.loads(open(_cfg(tmp_path)).read())["llm_profiles"] == {"m": {}}


def test_the_llm_settings_are_kernel_owned(tmp_path):
    """The declaration that stops it happening again.

    Talking to a model is kernel routing now — ``llm/`` owns profiles and
    brains — so nothing outside the kernel declares these.
    """
    from plugins.plugin_discovery import get_plugin_settings

    kernel = {entry[1] for entry in SETTINGS_DATA}
    plugin = {entry[1] for entry in get_plugin_settings()}
    for key in ("llm_profiles", "default_llm_profile"):
        assert key in kernel, f"{key} has no declared home"
        assert key not in plugin, f"{key} is declared in two places"


# ────────────────────────────────────────────────────────────────────
# The /config command (was test_config_command.py)
# ────────────────────────────────────────────────────────────────────

from types import SimpleNamespace
from plugins import plugin_discovery
from sandbox import Sandbox
from sandbox.handlers.kernel import _config_read


class DemoFrontend:
    name = "demo_frontend"
    config_settings = [
        (
            "Demo color",
            "demo_color_config_test",
            "Controls the demo color.",
            "blue",
            {"type": "text"},
        ),
        (
            "Hidden token",
            "demo_secret_config_test",
            "Never expose this.",
            "",
            {"hidden": True},
        ),
    ]


class SharedService:
    name = "shared_service"
    config_settings = [
        (
            "Demo color",
            "demo_color_config_test",
            "Controls the demo color.",
            "blue",
            {"type": "text"},
        ),
    ]


def _register():
    plugin_discovery._collect_config_settings(
        DemoFrontend(), plugin_type="frontend")
    plugin_discovery._collect_config_settings(
        SharedService(), service_names=["shared_service"],
        plugin_type="service")


def _remove():
    for key in ("demo_color_config_test", "demo_secret_config_test"):
        plugin_discovery._setting_to_services.pop(key, None)
        plugin_discovery._setting_to_plugins.pop(key, None)
        plugin_discovery._plugin_setting_types.pop(key, None)
        plugin_discovery._plugin_settings_keys.discard(key)
    plugin_discovery._plugin_settings[:] = [
        entry for entry in plugin_discovery._plugin_settings
        if entry[1] not in {
            "demo_color_config_test", "demo_secret_config_test"}]


def _context(monkeypatch):
    config = {
        "demo_color_config_test": "red",
        "keep_attachments_available_across_turns": True,
    }
    saved = {}
    rescans = []
    monkeypatch.setattr(
        "config.config_manager.save", lambda values: saved.update(values))
    monkeypatch.setattr(
        "config.config_manager.load_plugin_config", lambda: {})
    monkeypatch.setattr(
        "config.config_manager.save_plugin_config",
        lambda values: saved.update(values),
    )
    runtime = SimpleNamespace(
        config=config, refresh_session_specs=lambda: None)
    context = SimpleNamespace(
        config=dict(config),
        runtime=runtime,
        db=None,
        user_id=1,
        session_key="chat",
        orchestrator=SimpleNamespace(
            watcher=SimpleNamespace(rescan=lambda: rescans.append(True))),
        services={},
    )
    return context, saved, rescans


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_config.py",
            "ConfigCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_config_details_are_structured_redacted_and_hide_private_settings(
        monkeypatch):
    _register()
    context, _, _ = _context(monkeypatch)
    try:
        result = _config_read(context, {"details": True})
    finally:
        _remove()

    demo = next(
        item for item in result.data
        if item["key"] == "demo_color_config_test")
    assert demo["category"] == "plugin"
    assert demo["storage"] == "plugin_config.json"
    assert demo["owners"] == ["demo_frontend", "shared_service"]
    assert demo["restart_required"] is True
    assert not any(
        item["key"] == "demo_secret_config_test" for item in result.data)


def test_config_form_drills_through_category_plugin_and_typed_value(
        monkeypatch):
    _register()
    context, _, _ = _context(monkeypatch)
    try:
        initial = _run(context, {}, method="form")
        plugins = _run(
            context, {"category": "plugin"}, method="form")
        chosen = _run(
            context,
            {
                "category": "plugin",
                "plugin_name": "demo_frontend",
            },
            method="form",
        )
        edit = _run(
            context,
            {
                "setting_name": "demo_color_config_test",
                "action": "edit",
            },
            method="form",
        )
    finally:
        _remove()

    assert initial.data[0]["enum"] == [
        "kernel", "plugin", "user", "all"]
    assert "demo_frontend" in plugins.data[1]["enum"]
    assert "shared_service" in plugins.data[1]["enum"]
    assert chosen.data[-1]["enum"] == ["demo_color_config_test"]
    assert edit.data[-1]["type"] == "string"
    assert "Shared setting" in edit.data[-2]["prompt"]


def test_config_lists_categories_and_plugin_groups(monkeypatch):
    _register()
    context, _, _ = _context(monkeypatch)
    try:
        all_settings = _run(context, {})
        plugin_settings = _run(context, {"category": "plugin"})
        selected = _run(
            context,
            {"category": "plugin", "plugin_name": "demo_frontend"},
        )
    finally:
        _remove()

    assert "Kernel Settings (config.json):" in all_settings.data
    assert "Plugin Settings (plugin_config.json):" in all_settings.data
    assert "User Settings (per-user):" in all_settings.data
    assert "demo_frontend:" in plugin_settings.data
    assert "shared_service:" in plugin_settings.data
    assert selected.data.startswith("demo_frontend:")


def test_config_edit_preserves_output_and_restart_note(monkeypatch):
    _register()
    context, saved, _ = _context(monkeypatch)
    try:
        result = _run(
            context,
            {
                "setting_name": "demo_color_config_test",
                "action": "edit",
                "value": "green",
            },
            approve=lambda *_: True,
        )
    finally:
        _remove()

    assert result.data == (
        "Set demo_color_config_test = green. Restart required.")
    assert saved["demo_color_config_test"] == "green"


def test_config_write_rescans_watcher_settings(monkeypatch):
    context, _, rescans = _context(monkeypatch)

    result = _run(
        context,
        {
            "setting_name": "sync_directories",
            "action": "edit",
            "value": ["C:\\Notes"],
        },
        approve=lambda *_: True,
    )

    assert result.data == "Set sync_directories = C:\\Notes"
    assert rescans == [True]
