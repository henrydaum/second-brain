import logging
import json
from pathlib import Path

logger = logging.getLogger("Config")

"""
Config loader.

Creates a default config.json if it doesn't exist.
Loads and saves config as a plain dict.
"""


from config_data import SETTINGS_DATA

# Derive defaults from the single source of truth in config_data.py
DEFAULTS = {name: default for (_, name, _, default, _) in SETTINGS_DATA}


def load(path: str = "config.json") -> dict:
    """Load config from JSON file. Creates default if missing."""
    p = Path(path)

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

    return merged


def save(config: dict, path: str = "config.json"):
    """Save config dict to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved to {path}")