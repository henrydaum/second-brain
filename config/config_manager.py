"""Configuration support for config manager."""

import logging
import json
import os
import threading
from pathlib import Path

from paths import DATA_DIR

logger = logging.getLogger("Config")

"""
Config loader.

Creates a default config.json if it doesn't exist.
Loads and saves config as a plain dict.
"""


from config.config_data import SETTINGS_DATA

# Derive defaults from the single source of truth in config_data.py
DEFAULTS = {name: default for (_, name, _, default, _) in SETTINGS_DATA}
_LIST_KEYS = {name for _, name, _, default, info in SETTINGS_DATA if isinstance(default, list) or info.get("type") == "json_list"}

_DEFAULT_CONFIG_PATH = str(DATA_DIR / "config.json")
_DEFAULT_PLUGIN_CONFIG_PATH = str(DATA_DIR / "plugin_config.json")
_CONFIG_LOCK = threading.RLock()
USER_CONFIG_KEYS = {
    name for _, name, _, _, info in SETTINGS_DATA
    if isinstance(info, dict) and info.get("scope") == "user"
} | {"last_active_conversation_id"}


def _normalize_frontends(value) -> list[str]:
    """Normalize enabled_frontends: lowercase, deduplicated strings.

    Deliberately no existence whitelist — frontends are discovery-based store
    packages, so the kernel cannot know the valid set (an installed frontend's
    name must survive config load). Bootstrap warns and skips names discovery
    can't resolve."""
    if not isinstance(value, list):
        return list(DEFAULTS["enabled_frontends"])

    normalized = []
    seen = set()
    for item in value:
        name = str(item).strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    return normalized


def _normalize_list(value) -> list:
    """Internal helper to normalize list."""
    return value if isinstance(value, list) else ([value] if value not in (None, "") else [])


def load(path: str = None) -> dict:
    """Load config from JSON file. Creates default if missing."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        logger.info(f"No config found — creating default at {p}")
        save(DEFAULTS, path)
        return dict(DEFAULTS)

    with open(p, "r") as f:
        logger.info(f"Loading config from {p}")
        user_config = json.load(f)

    # If new settings are added, this adds them to the existing config.json
    merged = dict(DEFAULTS)
    merged.update(user_config)
    for key in USER_CONFIG_KEYS:
        merged.pop(key, None)
    for key in _LIST_KEYS - USER_CONFIG_KEYS:
        merged[key] = _normalize_list(merged.get(key, DEFAULTS[key]))
    # The LLM router is kernel-owned now and loads before service autoloading.
    # Remove the retired service name from existing configurations.
    merged["autoload_services"] = [
        name for name in merged.get("autoload_services", [])
        if name != "llm"
    ]
    merged["enabled_frontends"] = _normalize_frontends(
        merged.get("enabled_frontends", DEFAULTS["enabled_frontends"])
    )

    # If the schema introduced new SETTINGS_DATA keys since the file was last
    # written, persist the merged defaults now so the on-disk file no longer
    # drifts behind the schema.
    if (set(merged.keys()) - set(user_config.keys())
            or any(user_config.get(k) != merged.get(k) for k in _LIST_KEYS)
            or any(k in user_config for k in USER_CONFIG_KEYS)):
        save(merged, path)

    return merged


# Optional action-ledger hook. The bootstrap wires the live Database in so
# config saves leave an audit row; config/ itself stays db-free and fully
# usable (and silent) without it.
_LEDGER_DB = None


def set_ledger_db(db) -> None:
    """Wire the action ledger so config saves are recorded."""
    global _LEDGER_DB
    _LEDGER_DB = db


def _record_config_save(scope: str, changed_keys: list) -> None:
    """Append a config_save row to the action ledger — changed key NAMES
    only, never values (config may hold tokens/secrets)."""
    record = getattr(_LEDGER_DB, "record_action", None)
    if record is None or not changed_keys:
        return
    try:
        record(origin="system", action_type="config_save", ok=True, name=scope,
               args={"changed": sorted(changed_keys)})
    except Exception:
        pass


def _emit_config_changed(scope: str, changed_keys=None) -> None:
    """Announce a persisted config change so frontends can resync without
    polling. Defensive: a config write must never fail because of an emit, and
    the bus is imported lazily to keep this foundational module import-light.

    ``changed_keys`` carries key *names* only, never values — the same rule the
    ledger's ``config_save`` row follows, and for the same reason: config holds
    tokens. It is what lets a subscriber tell the user what changed rather than
    only that something did.
    """
    try:
        from events.event_bus import bus
        from events.event_channels import CONFIG_CHANGED
        bus.emit(CONFIG_CHANGED, {"scope": scope,
                                  "keys": sorted(changed_keys or [])})
    except Exception:
        pass


def save(config: dict, path: str = None):
    """Save config dict to JSON file."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    # Strip _root and plugin keys from persisted core config. Include keys
    # already living in plugin_config.json, not just currently-registered
    # plugin settings: the runtime config carries plugin values loaded early
    # (load_plugin_config_early) for plugins not yet discovered/installed, and
    # those must never be duplicated into core config.json.
    # Keys already living in plugin_config.json count as plugin keys, not just
    # currently-registered plugin settings: the runtime config carries values
    # loaded early for plugins not yet discovered, and those must never be
    # duplicated into core config.json.
    #
    # A *kernel* declaration always wins, though, and that exception is
    # load-bearing. Without it the rule was "whichever file already has it owns
    # it", which is not ownership at all — it made a setting's home an accident
    # of history, and left a key that had been re-homed into SETTINGS_DATA
    # permanently unable to reach the file that now owns it.
    plugin_keys = ((plugin_setting_keys() | set(load_plugin_config().keys()))
                   - set(DEFAULTS))
    existing = {}
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    merged = {**DEFAULTS, **existing, **(config or {})}
    to_save = {k: v for k, v in merged.items()
                if k != "_root" and k not in plugin_keys and k not in USER_CONFIG_KEYS}
    with open(path, "w") as f:
        json.dump(to_save, f, indent=4)
    logger.info(f"Config saved to {path}")
    changed = [k for k in to_save if to_save.get(k) != existing.get(k)]
    _record_config_save("core", changed)
    _emit_config_changed("core", changed)


# ── Plugin config ───────────────────────────────────────────────────

def plugin_setting_keys() -> set:
    """Return the set of variable_names owned by plugin config."""
    try:
        from plugins.plugin_discovery import get_plugin_settings
        return {entry[1] for entry in get_plugin_settings()}
    except ImportError:
        return set()


def migrate_secret_keys(key: str, value):
    """Rename credential fields to the ``secret_`` prefix that declares them.

    ``llm_profiles`` predates the naming rule (CLAUDE.md, "Secrets"), so a
    profile arriving with a bare ``llm_api_key`` would be stored as an ordinary
    setting and read back in plaintext. Renaming on the way in is what puts it
    behind a ``<secret:...>`` handle. Mutates and returns ``value``.
    """
    if key != "llm_profiles" or not isinstance(value, dict):
        return value
    for profile in value.values():
        if (isinstance(profile, dict) and "llm_api_key" in profile
                and "secret_llm_api_key" not in profile):
            profile["secret_llm_api_key"] = profile.pop("llm_api_key")
    return value


def is_user_scoped(key: str) -> bool:
    """Whether a setting belongs to the current user rather than the machine.

    Declared either by ``{"scope": "user"}`` in a setting's ``type_info`` or,
    for a plugin's own setting, by what discovery reports for it.
    """
    try:
        from plugins.plugin_discovery import (get_plugin_setting_scope,
                                              get_plugin_settings)
        plugin_settings = get_plugin_settings()
    except ImportError:
        plugin_settings, get_plugin_setting_scope = [], None

    entry = next((item for item in (*SETTINGS_DATA, *plugin_settings)
                  if item[1] == key), None)
    info = entry[4] if entry and isinstance(entry[4], dict) else {}
    if info.get("scope") == "user":
        return True
    if get_plugin_setting_scope is None:
        return False
    return (key in {item[1] for item in plugin_settings}
            and get_plugin_setting_scope(key) == "user")


def detect_profile_renames(old, new) -> dict:
    """Map ``{old_name: new_name}`` for profiles that were only renamed.

    A rename reads as one removal plus one addition, and is told apart from a
    genuine delete-and-create by the definition being unchanged. Sessions
    pointing at the old name have to follow it, or editing a profile's name
    silently detaches everyone using it.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return {}
    removed, added = set(old) - set(new), set(new) - set(old)
    return {source: target for source in removed for target in added
            if old[source] == new[target]}


def load_plugin_config(path: str = None) -> dict:
    """Load plugin config from JSON file. Returns empty dict if missing."""
    if path is None:
        path = _DEFAULT_PLUGIN_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        return {}
    with _CONFIG_LOCK:
        text = p.read_text(encoding="utf-8")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            value, end = json.JSONDecoder().raw_decode(text)
            if text[end:].strip():
                logger.warning(f"Plugin config had trailing data; rewriting clean JSON at {p}: {e}")
                save_plugin_config(value, path)
            return value


def save_plugin_config(plugin_values: dict, path: str = None):
    """Save plugin config dict to JSON file."""
    if path is None:
        path = _DEFAULT_PLUGIN_CONFIG_PATH
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Read before writing so the announcement can name what actually changed.
    # Deliberately *not* load_plugin_config: that one repairs a malformed file
    # by calling back into here, so reusing it would recurse forever. An
    # unreadable file simply means every key reads as new, which is a fine
    # answer for an announcement.
    previous = {}
    try:
        previous = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    if not isinstance(previous, dict):
        previous = {}
    with _CONFIG_LOCK:
        tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps(plugin_values, indent=4), encoding="utf-8")
        os.replace(tmp, p)
    logger.info(f"Plugin config saved to {p}")
    _emit_config_changed(
        "plugin",
        [k for k in (plugin_values or {})
         if (plugin_values or {}).get(k) != previous.get(k)],
    )


def rehome_kernel_keys(config: dict) -> list:
    """Move settings that have become kernel-owned into config.json.

    A capability that graduates from a plugin into the kernel leaves its
    settings behind: ``service_llm.py`` declared ``llm_profiles`` and
    ``default_llm_profile``, and absorbing it into ``llm/`` re-homed the code
    without re-homing the declaration. The values keep working — right up to
    the first write that does not carry them, after which readers look in
    config.json and the file still holding them is plugin_config.json.

    So the correction is a *move*, and the order it happens in is the whole
    of its safety: config.json gains the values first and plugin_config.json
    loses them second. Crashing in between costs a duplicated key, which the
    next boot cleans up; the other order costs the user their model
    configuration.

    Idempotent — once the keys are gone from plugin_config.json there is
    nothing left to move — so it is safe on every boot.
    """
    saved = load_plugin_config()
    moving = {key: value for key, value in saved.items() if key in DEFAULTS}
    if not moving:
        return []
    config.update(moving)
    save(config)
    save_plugin_config({k: v for k, v in saved.items() if k not in moving})
    logger.info("Moved %s from plugin_config.json to config.json: they are "
                "kernel settings now.", ", ".join(sorted(moving)))
    return sorted(moving)


def load_plugin_config_early(config: dict):
    """Phase 1 (before discovery): load existing plugin_config.json values
    into the runtime config so that build_services() etc. can see them.
    """
    rehome_kernel_keys(config)
    saved = load_plugin_config()
    if saved:
        config.update(saved)
        logger.info(f"Loaded {len(saved)} plugin config value(s) from plugin_config.json")


def reconcile_plugin_config(config: dict, plugin_settings: list):
    """Phase 2 (after all discovery): ensure every declared plugin setting
    exists in plugin_config.json with at least its default value.

    1. For each declared setting, use existing plugin_config value or the default.
    2. Write back plugin_config.json.
    3. Update the runtime config dict.
    """
    saved = load_plugin_config()
    plugin_values = dict(saved)

    for title, var_name, description, default, type_info in plugin_settings:
        # User-scoped settings live in each user's config blob, never in the
        # global plugin_config.json — their defaults apply lazily on read.
        if isinstance(type_info, dict) and type_info.get("scope") == "user":
            continue
        if var_name in plugin_values:
            continue
        plugin_values[var_name] = config.get(var_name, default)

    if plugin_values:
        save_plugin_config(plugin_values)

    config.update(plugin_values)
