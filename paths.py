"""
Centralized path constants.

Every module that needs DATA_DIR or ROOT_DIR imports from here.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

# Project root (where main.pyw lives)
ROOT_DIR = Path(__file__).parent

# Mutable user data: database, model cache, config, credentials
_system = platform.system()
if _system == "Windows":
    DATA_DIR = Path(os.getenv("LOCALAPPDATA", "")) / "Second Brain"
elif _system == "Darwin":
    DATA_DIR = Path.home() / "Library" / "Application Support" / "Second Brain"
else:
    DATA_DIR = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "Second Brain"

# Mirrored plugin trees. ``plugins/`` in the repo is the kernel/bundled tree;
# these DATA_DIR trees hold agent drafts and installed optional plugins.
SANDBOX_PLUGINS = DATA_DIR / "sandbox_plugins"
INSTALLED_PLUGINS = DATA_DIR / "installed_plugins"

# Attachment cache: files dropped in from frontends (e.g. Telegram).
# Registered as a sync_directory by default so the Stage_2 pipeline indexes them.
ATTACHMENT_CACHE = DATA_DIR / "attachment_cache"
ATTACHMENT_CACHE.mkdir(parents=True, exist_ok=True)


def open_file(path):
    """Open a file or folder with the system's default handler."""
    path = str(path)
    if _system == "Windows":
        os.startfile(path)
    elif _system == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
