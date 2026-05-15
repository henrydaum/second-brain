"""Configuration support for config manager."""

import logging
import json
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
_SUPPORTED_FRONTENDS = ("repl",)


def _normalize_frontends(value) -> list[str]:
    """Normalize enabled_frontends to the supported runtime set."""
    if not isinstance(value, list):
        return list(DEFAULTS["enabled_frontends"])

    normalized = []
    seen = set()
    for item in value:
        name = str(item).strip().lower()
        if name not in _SUPPORTED_FRONTENDS or name in seen:
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
    for key in _LIST_KEYS:
        merged[key] = _normalize_list(merged.get(key, DEFAULTS[key]))
    merged["enabled_frontends"] = _normalize_frontends(
        merged.get("enabled_frontends", DEFAULTS["enabled_frontends"])
    )

    # If the schema introduced new SETTINGS_DATA keys since the file was last
    # written, persist the merged defaults now so the on-disk file no longer
    # drifts behind the schema.
    if set(merged.keys()) - set(user_config.keys()) or any(user_config.get(k) != merged.get(k) for k in _LIST_KEYS):
        save(merged, path)

    return merged


def save(config: dict, path: str = None):
    """Save config dict to JSON file."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    # Strip _root and plugin keys from persisted core config
    plugin_keys = _get_plugin_keys()
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
                if k != "_root" and k not in plugin_keys}
    with open(path, "w") as f:
        json.dump(to_save, f, indent=4)
    logger.info(f"Config saved to {path}")


# ── Plugin config ───────────────────────────────────────────────────

def _get_plugin_keys() -> set:
    """Return the set of variable_names owned by plugin config."""
    try:
        from plugins.plugin_discovery import get_plugin_settings
        return {entry[1] for entry in get_plugin_settings()}
    except ImportError:
        return set()


def load_plugin_config(path: str = None) -> dict:
    """Load plugin config from JSON file. Returns empty dict if missing."""
    if path is None:
        path = _DEFAULT_PLUGIN_CONFIG_PATH
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


def save_plugin_config(plugin_values: dict, path: str = None):
    """Save plugin config dict to JSON file."""
    if path is None:
        path = _DEFAULT_PLUGIN_CONFIG_PATH
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(plugin_values, f, indent=4)
    logger.info(f"Plugin config saved to {p}")


def load_plugin_config_early(config: dict):
    """Phase 1 (before discovery): load existing plugin_config.json values
    into the runtime config so that build_services() etc. can see them.
    """
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
        if var_name in plugin_values:
            continue
        plugin_values[var_name] = config.get(var_name, default)

    if plugin_values:
        save_plugin_config(plugin_values)

    config.update(plugin_values)
