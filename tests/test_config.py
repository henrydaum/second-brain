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
    """The lite kernel ships only the REPL frontend, managed LLM autoload,
    and no scheduled jobs. Guard against accidental reintroduction."""
    assert config_manager.DEFAULTS["autoload_services"] == ["llm"]
    assert config_manager.DEFAULTS["enabled_frontends"] == ["repl"]
    assert DEFAULT_SCHEDULED_JOBS == {}
    assert config_manager.DEFAULTS["scheduled_jobs"] == {}


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
    assert on_disk["autoload_services"] == ["llm"]


def test_load_merges_missing_keys_and_persists(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"max_workers": 8}))

    config = config_manager.load(path)

    assert config["max_workers"] == 8  # user value preserved
    assert config["enabled_frontends"] == ["repl"]  # default filled in
    # Schema drift is healed on disk, not just in memory.
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "enabled_frontends" in on_disk


def test_load_strips_user_config_keys_from_disk(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "last_active_conversation_id": 12,
        "active_agent_profile": "builder",
        "skip_permissions": ["run_command"],
    }))

    config = config_manager.load(path)

    assert config["last_active_conversation_id"] == 12  # legacy value is still available for migration
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "last_active_conversation_id" not in on_disk
    assert "active_agent_profile" not in on_disk
    assert "skip_permissions" not in on_disk


def test_load_normalizes_enabled_frontends(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "enabled_frontends": ["REPL", "telegram", "bogus", "repl"]
    }))

    config = config_manager.load(path)

    # Lowercased, unsupported dropped, deduped, order preserved.
    assert config["enabled_frontends"] == ["repl", "telegram"]


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
