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
