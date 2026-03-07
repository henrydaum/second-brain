import logging
import json
from pathlib import Path

logger = logging.getLogger("Config")

"""
Config loader.

Creates a default config.json if it doesn't exist.
Loads and saves config as a plain dict.
"""


DEFAULTS = {
    # Basic config
    "sync_directories": [],
    "db_path": "database.db",
    # Whitelist/blacklist files
    "ignored_extensions": [],
    "ignored_folders": ["node_modules", "__pycache__", ".git", ".venv", "venv"],
    "skip_hidden_folders": True,
    # Threading
    "max_workers": 4,
    "poll_interval": 1.0,
    "task_timeout": 300,
    "reprocess_interval": 300,
    # LLM
    "llm_model_name": "gemma-3-4b-it@q4_k_s",
    "llm_endpoint": "http://127.0.0.1:1234",
    "llm_api_key": "OPENAI_API_KEY",
    # Embedding
    "embed_text_model_name": "BAAI/bge-m3",
    "embed_image_model_name": "clip-ViT-L-14",
    "embed_use_cuda": False,
    "embed_chunk_size": 512,
    # Cloud Services
    "use_drive": True,
}


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