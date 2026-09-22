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
    """The kernel ships REPL plus the LLM router and Timekeeper, with no jobs.

    ``enabled_frontends`` carries ``http`` as well, and it is not a widening of
    "minimal": the HTTP frontend ships in the kernel tree and is the whole of
    what a web or native app can reach, so leaving it off by default meant a
    fresh install whose UI could not connect until somebody found the command
    for it. It binds loopback and refuses every request without the token, so
    what it costs an install that never opens a browser is one idle socket.

    Note the default reaches existing installs not at all — ``load`` merges
    ``DEFAULTS`` for *missing* keys, and anyone who has run the app has this
    one. That is the intended half: a default is what a new install starts
    from, never an opinion imposed on a configuration somebody already has.
    """
    assert config_manager.DEFAULTS["autoload_services"] == ["timekeeper"]
    assert config_manager.DEFAULTS["enabled_frontends"] == ["repl", "http"]
    assert DEFAULT_SCHEDULED_JOBS == {}
    assert config_manager.DEFAULTS["scheduled_jobs"] == {}
    assert config_manager.DEFAULTS["keep_attachments_available_across_turns"] is False


def test_an_existing_config_does_not_gain_a_new_default(tmp_path):
    """The other half of the claim above, stated where it can fail.

    Someone who enabled only the REPL meant only the REPL. ``load`` filling a
    key that is *present* would be the app overriding a choice, and the
    frontend it would switch on opens a port — which is exactly the kind of
    default nobody wants applied behind them.
    """
    path = _cfg(tmp_path)
    io_config = {"enabled_frontends": ["repl"], "sync_directories": []}
    json.dump(io_config, open(path, "w", encoding="utf-8"))

    assert config_manager.load(path)["enabled_frontends"] == ["repl"]


def test_defaults_cover_every_settings_entry():
    names = {entry[1] for entry in SETTINGS_DATA}
    assert set(config_manager.DEFAULTS) == names


# ── load() ───────────────────────────────────────────────────────────

def test_load_creates_default_config_when_missing(tmp_path):
    path = _cfg(tmp_path)
    config = config_manager.load(path)

    # Against DEFAULTS rather than a literal: what this test is about is
    # that the default is *filled in*, not which frontends it names, and
    # three copies of the list is three tests to edit for one decision.
    assert config["enabled_frontends"] == config_manager.DEFAULTS["enabled_frontends"]
    # The file is written so subsequent loads are stable.
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["autoload_services"] == ["timekeeper"]


def test_load_merges_missing_keys_and_persists(tmp_path):
    path = _cfg(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"max_workers": 8}))

    config = config_manager.load(path)

    assert config["max_workers"] == 8  # user value preserved
    assert (config["enabled_frontends"]
            == config_manager.DEFAULTS["enabled_frontends"])  # default filled in
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
        "startup_restore_conversation": False,
    }))

    config = config_manager.load(path)

    assert "last_active_conversation_id" not in config
    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "last_active_conversation_id" not in on_disk
    assert "active_agent_profile" not in on_disk
    assert "startup_restore_conversation" not in on_disk


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
    assert (on_disk["enabled_frontends"]
            == config_manager.DEFAULTS["enabled_frontends"])


def test_save_preserves_existing_unrelated_values(tmp_path):
    path = _cfg(tmp_path)
    config_manager.save({"max_workers": 8}, path)
    config_manager.save({"poll_interval": 2.0}, path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert on_disk["max_workers"] == 8
    assert on_disk["poll_interval"] == 2.0


def _announced(monkeypatch) -> list:
    """Every ``CONFIG_CHANGED`` key list emitted from here on."""
    from events.event_bus import bus
    from events.event_channels import CONFIG_CHANGED

    seen: list = []
    real = bus.emit
    monkeypatch.setattr(bus, "emit",
                        lambda channel, payload=None, **kw: (
                            seen.append(sorted((payload or {}).get("keys") or []))
                            if channel == CONFIG_CHANGED else real(channel, payload, **kw)))
    return seen


def test_materializing_a_default_is_not_a_change_anyone_made(tmp_path, monkeypatch):
    """The announcement names what the user changed, not what the file gained.

    ``save`` merges ``DEFAULTS``, so the first write after a schema addition
    persists settings nobody touched. Diffing against the *file* called every
    one of them changed, and it reached the user as
    "⚙ Settings changed: autoload_services" for deleting a scheduled job — an
    announcement naming a setting they had approved one unrelated write for.
    Diffing against the effective previous config is what makes the two agree.
    """
    path = _cfg(tmp_path)
    partial = {k: v for k, v in config_manager.DEFAULTS.items()
               if k != "autoload_services"}
    (tmp_path / "config.json").write_text(json.dumps(partial))
    seen = _announced(monkeypatch)

    config_manager.save(dict(partial), path)

    assert "autoload_services" in json.loads(open(path).read()), (
        "the file still gains the default; only the announcement changes")
    assert seen == [[]]

    partial["scheduled_jobs"] = {"nightly": {}}
    config_manager.save(dict(partial), path)
    assert seen[-1] == ["scheduled_jobs"], "a real change still announces"


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
        "startup_restore_conversation": False,
    }, path)

    on_disk = json.loads((tmp_path / "config.json").read_text())
    assert "last_active_conversation_id" not in on_disk
    assert "active_agent_profile" not in on_disk
    assert "startup_restore_conversation" not in on_disk


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
            "bundled/commands/command_config.py",
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


def test_a_kernel_setting_is_written_once_however_it_is_scoped(monkeypatch):
    """One setting, one home, one announcement.

    ``_config_write`` took the plugin branch on ``scope="plugin"`` *or* on the
    key appearing in any plugin's ``config_settings`` — and the timekeeper does
    both for ``scheduled_jobs``, which the kernel also declares. So deleting one
    scheduled job wrote the value to config.json and plugin_config.json and
    announced it twice, while ``rehome_kernel_keys`` moved the stray copy back
    at the next boot. The kernel declaration decides, and it overrides the
    caller's own ``scope``: a plugin cannot rehome a setting it does not own.
    """
    from sandbox.handlers.kernel import _config_write

    context, saved, _ = _context(monkeypatch)
    plugin_saves = []
    monkeypatch.setattr("config.config_manager.save_plugin_config",
                        lambda values, path=None: plugin_saves.append(values))
    monkeypatch.setattr("config.config_manager.plugin_setting_keys",
                        lambda: {"scheduled_jobs"})

    result = _config_write(context, {"key": "scheduled_jobs",
                                     "value": {"nightly": {}},
                                     "scope": "plugin"})

    assert result.ok
    assert saved["scheduled_jobs"] == {"nightly": {}}
    # One save, so one announcement — each ``save*`` emits its own
    # ``CONFIG_CHANGED``, which is why the second write was the second
    # notification. The emit itself is pinned in
    # ``test_materializing_a_default_is_not_a_change_anyone_made``; here
    # ``save`` is stubbed, so the count of saves is the thing to assert.
    assert plugin_saves == [], "a kernel setting never reaches plugin_config"


def test_a_stale_context_writes_its_own_key_and_nothing_else(tmp_path,
                                                              monkeypatch):
    """A resident box's config is a copy from boot; writing through it must
    not put everything changed since then back.

    ``_config_write`` saved the context's whole dict. The timekeeper's context
    was built at boot, so persisting ``scheduled_jobs`` reverted every grant
    remembered since, on disk — and the next honest save announced
    "Settings changed: autoload_services, default_llm_profile,
    net_allowed_hosts, scheduled_jobs, shell_allowed_prefixes" for one edit.
    """
    from types import SimpleNamespace

    from runtime import context as runtime_context
    from sandbox.handlers.kernel import _config_read, _config_write

    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        str(tmp_path / "plugin_config.json"))
    announced = []
    monkeypatch.setattr(config_manager, "_emit_config_changed",
                        lambda scope, keys=None: announced.append(sorted(keys or [])))

    live = config_manager.load()
    monkeypatch.setitem(runtime_context._KERNEL_PARTS, "config", live)
    stale = SimpleNamespace(config=dict(live), runtime=None, user_id=None)

    # Something changes after the resident box took its copy.
    live["shell_allowed_prefixes"] = ["git status"]
    config_manager.save({"shell_allowed_prefixes": ["git status"]})
    announced.clear()

    assert _config_read(stale, {"key": "shell_allowed_prefixes"}).data \
        == ["git status"], "a stale context reads the live value"

    assert _config_write(stale, {"key": "scheduled_jobs",
                                 "value": {"nightly": {}}}).ok

    on_disk = config_manager._stored_config()
    assert on_disk["shell_allowed_prefixes"] == ["git status"]
    assert on_disk["scheduled_jobs"] == {"nightly": {}}
    assert live["scheduled_jobs"] == {"nightly": {}}
    assert announced == [["scheduled_jobs"]]


def test_a_plugin_setting_still_reaches_plugin_config(monkeypatch):
    """The other half of the same branch, which must keep working.

    Only a *kernel* declaration wins. A setting the kernel says nothing about
    is the plugin's, and ``scope="plugin"`` is how a service persists its own
    state.
    """
    from sandbox.handlers.kernel import _config_write

    context, _, _ = _context(monkeypatch)
    plugin_saves = []
    monkeypatch.setattr("config.config_manager.save_plugin_config",
                        lambda values, path=None: plugin_saves.append(values))

    result = _config_write(context, {"key": "demo_color_config_test",
                                     "value": "green", "scope": "plugin"})

    assert result.ok
    assert plugin_saves == [{"demo_color_config_test": "green"}]


# ── Minting, and the settings the HTTP frontend shares with the kernel ──

def test_the_http_token_is_minted_once_and_never_replaced(tmp_path, monkeypatch):
    """Nobody should have to invent a bearer token, and nobody should lose one.

    The token is required — it is the whole of what stands between whoever can
    reach the port and every Request the SDK has — but it is not a decision.
    So the kernel makes one. The half that matters more is the second call:
    replacing a token on a later boot would silently break whatever is already
    holding it.
    """
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    config = {"secret_http_token": ""}

    assert config_manager.ensure_minted_secrets(config) == ["secret_http_token"]
    first = config["secret_http_token"]
    assert len(first) > 30

    assert config_manager.ensure_minted_secrets(config) == []
    assert config["secret_http_token"] == first


def test_an_existing_token_survives_minting(tmp_path, monkeypatch):
    """A token the old ``/setup`` printed is one a running UI is still using."""
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    config = {"secret_http_token": "pasted-into-an-env-file-last-month"}

    assert config_manager.ensure_minted_secrets(config) == []
    assert config["secret_http_token"] == "pasted-into-an-env-file-last-month"


def test_reconcile_leaves_kernel_settings_out_of_plugin_config(tmp_path, monkeypatch):
    """The third path that did not apply the one rule about where a key lives.

    ``save`` and ``_config_write`` both ask ``is_kernel_setting`` and both let
    a kernel declaration win. ``reconcile_plugin_config`` did not, so a setting
    declared by *both* — ``scheduled_jobs``, and now every HTTP setting — was
    seeded into plugin_config.json after discovery, moved back out by
    ``rehome_kernel_keys`` on the next boot, and seeded again. Churn is the
    mild half: the seeded copy is whatever the default was, so it also
    overwrote the real value in the runtime config on its way past.
    """
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        str(tmp_path / "plugin_config.json"))
    saved = []
    monkeypatch.setattr(config_manager, "save_plugin_config",
                        lambda values, path=None: saved.append(values))
    config = {"secret_http_token": "the-real-one"}

    config_manager.reconcile_plugin_config(config, [
        ("HTTP API token", "secret_http_token", "", "", {"type": "string"}),
        ("Demo", "demo_setting_config_test", "", "blue", {"type": "string"}),
    ])

    assert saved == [{"demo_setting_config_test": "blue"}]
    assert config["secret_http_token"] == "the-real-one"


def test_the_shutdown_snapshot_leaves_kernel_settings_alone(tmp_path,
                                                            monkeypatch):
    """The fourth path, and the one that cost a scheduled agent its home.

    Shutdown and restart snapshot the plugin half of the live config. Both
    picked their keys off ``get_plugin_settings()`` alone, so a setting the
    kernel *also* declares went into plugin_config.json on every clean exit —
    and ``rehome_kernel_keys`` moved that copy back over config.json on the
    next boot. The timekeeper pins a scheduled subagent's conversation by
    writing it into ``scheduled_jobs``; the pin therefore survived until the
    next restart and no longer, and the job opened a fresh conversation every
    single firing.
    """
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        str(tmp_path / "plugin_config.json"))
    monkeypatch.setattr(config_manager, "plugin_setting_keys",
                        lambda: {"scheduled_jobs", "secret_http_token",
                                 "demo_setting_config_test"})
    pinned = {"daily": {"channel": "subagent.spawn",
                        "payload": {"conversation_id": 587}}}

    written = config_manager.persist_plugin_settings({
        "scheduled_jobs": pinned,
        "secret_http_token": "minted",
        "demo_setting_config_test": "blue",
    })

    assert written == ["demo_setting_config_test"]
    on_disk = config_manager.load_plugin_config()
    assert on_disk == {"demo_setting_config_test": "blue"}


def test_the_composition_root_snapshots_through_the_one_rule():
    """Stated structurally, because the two copies were the whole bug.

    Shutdown and restart each had their own hand-rolled snapshot, and both
    were wrong the same way. Routing them through one function is the fix;
    this is what stops a third one being written beside it.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "main.pyw").read_text(encoding="utf-8")

    assert source.count("config_manager.persist_plugin_settings(config)") == 2
    assert "save_plugin_config" not in source, (
        "the composition root must not write plugin_config.json itself — "
        "persist_plugin_settings is where is_kernel_setting is applied")


def test_a_snapshot_adds_and_never_replaces(tmp_path, monkeypatch):
    """A plugin that failed to load must not lose its settings.

    The keys come from live discovery, so a snapshot that *replaces* the file
    erases everything belonging to something absent this boot — which is the
    boot where its settings matter most, since fixing it is how you get it
    back.
    """
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    config_manager.save_plugin_config({"absent_plugin_setting": "keep me",
                                       "demo_setting_config_test": "old"},
                                      plugin_path)
    monkeypatch.setattr(config_manager, "plugin_setting_keys",
                        lambda: {"demo_setting_config_test"})

    config_manager.persist_plugin_settings({
        "demo_setting_config_test": "new"})

    assert config_manager.load_plugin_config() == {
        "absent_plugin_setting": "keep me",
        "demo_setting_config_test": "new",
    }


def test_a_snapshot_leaves_one_users_setting_out_of_the_global_file(
        tmp_path, monkeypatch):
    """A user-scoped setting's home is that user's config blob.

    ``_config_write`` mirrors the value into the runtime dict on its way
    past, so a snapshot taken by declaration published whoever wrote last as
    everybody's default.
    """
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        str(tmp_path / "plugin_config.json"))
    monkeypatch.setattr(config_manager, "plugin_setting_keys",
                        lambda: {"per_user_setting", "demo_setting_config_test"})
    monkeypatch.setattr(config_manager, "is_user_scoped",
                        lambda key: key == "per_user_setting")

    written = config_manager.persist_plugin_settings(
        {"per_user_setting": "henry's", "demo_setting_config_test": "blue"})

    assert written == ["demo_setting_config_test"]
    assert config_manager.load_plugin_config() == {
        "demo_setting_config_test": "blue"}


def test_rehoming_never_overwrites_a_value_config_json_owns(tmp_path,
                                                            monkeypatch):
    """The other half of the same bug: a migration that keeps firing.

    Whatever puts a kernel key back into plugin_config.json, rehoming must
    not be the thing that reverts config.json on the next boot. config.json
    is the home, so its value wins and the stray is dropped.
    """
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    pinned = {"daily": {"channel": "subagent.spawn",
                        "payload": {"conversation_id": 587}}}
    stale = {"daily": {"channel": "subagent.spawn", "payload": {}}}
    config_manager.save({"scheduled_jobs": pinned}, _cfg(tmp_path))
    config_manager.save_plugin_config({"scheduled_jobs": stale}, plugin_path)

    runtime = {"scheduled_jobs": pinned}
    assert config_manager.rehome_kernel_keys(runtime) == []

    assert runtime["scheduled_jobs"] == pinned
    assert json.loads(open(_cfg(tmp_path)).read())["scheduled_jobs"] == pinned
    # ...and the stray is gone rather than left to be moved again next boot.
    assert config_manager.load_plugin_config() == {}


def test_rehoming_still_adopts_a_value_config_json_never_had(tmp_path,
                                                             monkeypatch):
    """Ownership is tested against the default, not against presence.

    ``load`` persists the merged schema, so every kernel key is *present* in
    config.json whether or not anybody set it. Reading presence as ownership
    would discard the real value of a setting that has genuinely never been
    written to its new home — the ``llm_profiles`` loss this function exists
    to repair, in reverse.
    """
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    config_manager.save(dict(config_manager.DEFAULTS), _cfg(tmp_path))
    assert "llm_profiles" in json.loads(open(_cfg(tmp_path)).read())
    config_manager.save_plugin_config({"llm_profiles": {"m": {}}}, plugin_path)

    runtime = {}
    assert config_manager.rehome_kernel_keys(runtime) == ["llm_profiles"]
    assert runtime["llm_profiles"] == {"m": {}}
    assert json.loads(
        open(_cfg(tmp_path)).read())["llm_profiles"] == {"m": {}}


def test_a_pinned_schedule_survives_a_restart(tmp_path, monkeypatch):
    """The whole round trip, which is where this was only ever visible.

    Pin, exit, boot. Each half looked correct on its own and the pair lost
    the value, so the regression is stated end to end rather than as two
    unit tests that both pass while the feature is broken.
    """
    plugin_path = str(tmp_path / "plugin_config.json")
    monkeypatch.setattr(config_manager, "_DEFAULT_PLUGIN_CONFIG_PATH",
                        plugin_path)
    monkeypatch.setattr(config_manager, "_DEFAULT_CONFIG_PATH", _cfg(tmp_path))
    monkeypatch.setattr(config_manager, "plugin_setting_keys",
                        lambda: {"scheduled_jobs"})
    pinned = {"daily": {"channel": "subagent.spawn",
                        "payload": {"conversation_id": 587}}}

    # The timekeeper pins the conversation the job ran in...
    config = config_manager.load(_cfg(tmp_path))
    config["scheduled_jobs"] = pinned
    config_manager.save(config, _cfg(tmp_path))
    # ...the process exits some time later, from a runtime dict that may be
    # older than the pin...
    config_manager.persist_plugin_settings(
        {"scheduled_jobs": {"daily": {"channel": "subagent.spawn",
                                      "payload": {}}}})
    # ...and the next boot reads its schedules back.
    booted = config_manager.load(_cfg(tmp_path))
    config_manager.load_plugin_config_early(booted)

    assert booted["scheduled_jobs"] == pinned


def test_the_http_frontend_still_declares_the_settings_it_reveals():
    """A declaration that looks like a duplicate and is a permission.

    Both the kernel and ``frontend_http`` declare ``secret_http_token``. The
    kernel declaration decides the *file*; the plugin declaration is what
    ``policy._owns_setting`` matches, and it is the only reason the frontend's
    ``secrets.reveal`` at start-up is answered instead of raising a dialog into
    a chain that is unattended by construction — which fails as a frontend that
    starts happily and answers 401 to everything.

    So this is pinned as a *negative*: tidying the apparent duplication away
    breaks start-up and nothing says so.
    """
    from pathlib import Path

    from sandbox.validator import validate

    source = (Path(__file__).resolve().parents[1]
              / "bundled/frontends/frontend_http.py").read_text(encoding="utf-8")
    declared = validate(source, filename="frontend_http.py").declarations
    settings = {entry[1] for entry in
                (declared.get("classes") or [{}])[0].get("config_settings", [])}

    assert "secret_http_token" in settings, (
        "frontend_http must keep declaring the settings it reads; the kernel "
        "declaring them too decides the file, not the permission")
    assert config_manager.is_kernel_setting("secret_http_token")
    assert config_manager.is_kernel_setting("http_client_url")


def test_no_test_can_write_the_developers_real_config():
    """The guard in ``conftest`` that makes this suite safe to run.

    ``config_manager.save`` resolves ``path=None`` to a module default, and
    plenty of code saves without naming one — ``config.write``'s handler, the
    approval dialog's "always allow". A test reaching any of those used to
    write the real file, with no ledger row to show for it, and the damage
    surfaced later inside the running app.
    """
    from pathlib import Path

    from config import config_manager
    from paths import DATA_DIR

    for attribute in ("_DEFAULT_CONFIG_PATH", "_DEFAULT_PLUGIN_CONFIG_PATH"):
        live = Path(getattr(config_manager, attribute)).resolve()
        assert DATA_DIR.resolve() not in live.parents, (
            f"{attribute} points into the real DATA_DIR during tests")


def test_saving_without_a_path_lands_in_the_scratch_file():
    """The guard has to survive the indirection, not just the constant.

    Asserting on the module attribute alone would pass while ``save`` read the
    value at import time into a default argument, which is exactly the shape
    that would leave the hole open.
    """
    from pathlib import Path

    from config import config_manager
    from paths import DATA_DIR

    config_manager.save({"sync_directories": ["/scratch-only"]})

    written = Path(config_manager._DEFAULT_CONFIG_PATH)
    assert written.exists()
    assert DATA_DIR.resolve() not in written.resolve().parents
