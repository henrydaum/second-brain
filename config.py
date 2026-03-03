import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

"""
Config loader.

Creates a default config.json if it doesn't exist.
Loads and saves config as a plain dict.
"""


DEFAULTS = {
    # Directories to watch
    "sync_directories": [],

    # Extensions to ignore (overrides modality mapping)
    "ignored_extensions": [],  # need to implement

    # Folders to skip
    "ignored_folders": ["node_modules", "__pycache__", ".git", ".venv", "venv"],
    "skip_hidden_folders": True,

    # Workers
    "max_workers": 4,

    # Dispatch
    "poll_interval": 1.0,

    # Timeouts
    "task_timeout": 300,

    # Reprocess interval (seconds) — skip re-queuing if recently processed
    "reprocess_interval": 300,

    # Database
    "db_path": "forge.db",
}


def load(path: str = "config.json") -> dict:
    """Load config from JSON file. Creates default if missing."""
    p = Path(path)

    if not p.exists():
        logger.info(f"No config found — creating default at {p}")
        save(DEFAULTS, path)
        return dict(DEFAULTS)

    with open(p, "r") as f:
        user_config = json.load(f)

    # Merge with defaults so new keys are always present
    merged = dict(DEFAULTS)
    merged.update(user_config)

    return merged


def save(config: dict, path: str = "config.json"):
    """Save config dict to JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved to {path}")