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

# Mutable demo data: database, config, credentials. Override for deployments.
_APP_DATA_NAME = os.getenv("SECOND_BRAIN_APP_DATA", "Second Brain Demo")
_DATA_OVERRIDE = os.getenv("SECOND_BRAIN_DATA_DIR")
_system = platform.system()
if _DATA_OVERRIDE:
    DATA_DIR = Path(_DATA_OVERRIDE)
elif _system == "Windows":
    DATA_DIR = Path(os.getenv("LOCALAPPDATA", "")) / _APP_DATA_NAME
elif _system == "Darwin":
    DATA_DIR = Path.home() / "Library" / "Application Support" / _APP_DATA_NAME
else:
    DATA_DIR = Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / _APP_DATA_NAME

# Sandbox directories for agent-created plugins
SANDBOX_TOOLS    = DATA_DIR / "sandbox_tools"
SANDBOX_TASKS    = DATA_DIR / "sandbox_tasks"
SANDBOX_SERVICES = DATA_DIR / "sandbox_services"
SANDBOX_COMMANDS = DATA_DIR / "sandbox_commands"
SANDBOX_FRONTENDS = DATA_DIR / "sandbox_frontends"

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
