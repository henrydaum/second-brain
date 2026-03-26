"""
Centralized path constants.

Every module that needs DATA_DIR or ROOT_DIR imports from here.
"""

import os
from pathlib import Path

# Project root (where main.pyw lives)
ROOT_DIR = Path(__file__).parent

# Mutable user data: database, model cache, config, credentials
DATA_DIR = Path(os.getenv("LOCALAPPDATA", "")) / "Second Brain"

# Sandbox directories for agent-created plugins
SANDBOX_TOOLS    = DATA_DIR / "sandbox_tools"
SANDBOX_TASKS    = DATA_DIR / "sandbox_tasks"
SANDBOX_SERVICES = DATA_DIR / "sandbox_services"
