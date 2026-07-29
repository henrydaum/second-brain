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
