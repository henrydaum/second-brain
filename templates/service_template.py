"""
SERVICE TEMPLATE
================
This file is a self-contained reference for creating new services.
It is NOT imported by the running system — it exists for LLM consumption only.

To create a new service:
  1. Use build_plugin(plugin_type="service", file_name="<name>Service.py",
     action="create", code="...") to write the file to the sandbox.
  2. The code MUST inherit from BaseService and include:
       from Stage_0.BaseService import BaseService
  3. Implement _load(), unload(), and your service methods.
  4. Add a build_services(config) factory function at the bottom.
  5. Hot-reload picks it up automatically — no restart needed.
  6. If the service needs extra packages, install them first with
     run_command(command="pip install <pkg>", justification="...", timeout=300).

build_plugin automatically validates:
  - Correct import (from Stage_0.BaseService import BaseService)
  - Class inheriting BaseService
  - Presence of build_services() function
  - File naming conventions


AUTO-DISCOVERY RULES
--------------------
- File must be in Stage_0/services/ (baked-in) or the sandbox services dir
- File must NOT start with "_"
- Module must have a top-level build_services(config) -> dict function
- The returned dict maps service names to service instances
- Service names are how tasks/tools reference the service in requires_services


SERVICE LIFECYCLE
-----------------
  1. build_services(config) is called at startup — creates the instance
  2. load() is called when a user or the system needs the service
     - Calls your _load() implementation
     - Sets self.loaded = True on success
     - Handles timing and logging automatically
  3. The service is used by tasks/tools via context.services.get("name")
  4. unload() is called to free resources (GPU memory, connections, etc.)

Services can be loaded/unloaded at runtime from the GUI or CLI.


SHARED vs PER-CALL
------------------
  shared = True  (default) — One instance used by all threads.
                 Good for: thread-safe models (LLM, embedders).
                 Access directly: service.encode(text)

  shared = False           — Callers use get_client() for thread safety.
                 Good for: API clients with auth state (Google Drive).
                 Override get_client() to return a fresh client.


CONFIG ACCESS
-------------
The config dict (from config.json) is passed to build_services().
Use it to read settings like model names, API keys, device preferences.
Settings are defined in config_data.py.
"""

# =====================================================================
# BASE CLASS (copied from Stage_0/BaseService.py for self-containment)
# =====================================================================

import logging
import time
from abc import ABC, abstractmethod


class BaseService(ABC):
    model_name: str = ""    # human-readable name shown in CLI/GUI
    shared: bool = True     # True = one instance for all threads

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @loaded.setter
    def loaded(self, value: bool):
        self._loaded = value

    def load(self) -> bool:
        """Wraps _load() with automatic timing. Subclasses override _load()."""
        name = self.model_name or self.__class__.__name__
        logger = logging.getLogger("BaseService")
        logger.info(f"Loading model: {name}...")
        t0 = time.time()
        try:
            result = self._load()
            if result:
                logger.info(f"Model loaded: {name} ({time.time() - t0:.2f}s)")
            else:
                logger.warning(f"Model failed to load: {name} ({time.time() - t0:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Model crashed during load: {name}: {e}")
            raise

    @abstractmethod
    def _load(self) -> bool:
        """Initialize the service. Return True on success."""
        ...

    @abstractmethod
    def unload(self):
        """Release all resources. Must be safe to call even if not loaded."""
        ...

    def get_client(self):
        """Override for per-call services (shared=False)."""
        raise NotImplementedError


# =====================================================================
# EXAMPLE: A simple shared service (e.g. audio transcription)
# =====================================================================

# import gc
# import os
# from pathlib import Path
# from Stage_0.BaseService import BaseService
#
# logger = logging.getLogger("WhisperService")
#
#
# class FasterWhisperService(BaseService):
#     shared = True  # transcribe() is stateless
#
#     def __init__(self, model_name="base", device="cuda"):
#         super().__init__()
#         self.model_name = model_name
#         self.device = device
#         self.model = None
#
#     def _load(self):
#         from faster_whisper import WhisperModel
#         self.model = WhisperModel(self.model_name, device=self.device)
#         self.loaded = True
#         return True
#
#     def unload(self):
#         if self.model:
#             del self.model
#             self.model = None
#         self.loaded = False
#         gc.collect()
#         logger.info("Whisper model unloaded.")
#
#     def transcribe(self, audio_path: str) -> str:
#         """Transcribe an audio file. Returns transcript text."""
#         if not self.loaded or not self.model:
#             return ""
#         segments, info = self.model.transcribe(audio_path, beam_size=5)
#         return " ".join(seg.text.strip() for seg in segments)
#
#
# def build_services(config: dict) -> dict:
#     return {
#         "whisper": FasterWhisperService(
#             model_name=config.get("whisper_model_name", "base"),
#             device="cuda" if config.get("whisper_use_cuda", True) else "cpu",
#         ),
#     }


# =====================================================================
# EXAMPLE: A per-call service (e.g. API client with auth)
# =====================================================================

# from Stage_0.BaseService import BaseService
#
#
# class GoogleDriveService(BaseService):
#     shared = False  # each caller gets a fresh API client
#
#     def __init__(self):
#         super().__init__()
#         self.model_name = "Google Drive"
#         self.credentials = None
#
#     def _load(self):
#         # Load OAuth credentials from disk
#         self.credentials = load_credentials()
#         self.loaded = True
#         return True
#
#     def unload(self):
#         self.credentials = None
#         self.loaded = False
#
#     def get_client(self):
#         """Return a fresh Drive API client for thread-safe usage."""
#         from googleapiclient.discovery import build
#         return build("drive", "v3", credentials=self.credentials)
#
#
# def build_services(config: dict) -> dict:
#     return {"drive": GoogleDriveService()}
