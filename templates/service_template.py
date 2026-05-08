"""
SERVICE TEMPLATE
================
This file is a self-contained reference for creating new services.
It is NOT imported by the running system — it exists for LLM consumption only.

Write services so their role is obvious: what capability they provide, what
config they need, when they are loaded, and how callers should access them.

Service authoring flow:
  1. Read this template, then read one similar built-in service for style.
  2. Create sandbox_services/<name>Service.py or sandbox_services/<name>.py with edit_file.
  3. The code MUST inherit from BaseService and include:
       from plugins.BaseService import BaseService
  4. Implement _load(), unload(), and your service methods.
  5. Add a build_services(config) factory function at the bottom.
  6. Call register_plugin(plugin_type="service", file_name="<name>Service.py").
  7. If registration fails, read the error, edit the same file, and retry.
  8. Valid sandbox service files are loaded automatically on startup.
  9. To update: edit the file and call register_plugin again.
 10. To remove live only: unregister_plugin(plugin_type="service", plugin_name="<service name>").
     To remove durably: also delete the sandbox file with edit_file.
 11. If the service needs extra packages, install them first with
     run_command(command="pip install <pkg>", justification="...", timeout=300).

register_plugin validates:
  - Correct import (from plugins.BaseService import BaseService)
  - Class inheriting BaseService
  - Presence of build_services() function
  - File naming conventions


AUTO-DISCOVERY RULES
--------------------
- File must be in plugins/services/ (baked-in) or the sandbox services dir
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
     - Inside a service, use self.services.get("name") to reach peers.
  4. unload() is called to free resources (GPU memory, connections, etc.)

Services can be loaded/unloaded at runtime from the Telegram frontend or REPL.


TRIGGERING EVENT TASKS FROM A SERVICE
-------------------------------------
Services can fire event-triggered tasks by emitting on the bus. This is
how a cron-like service drives periodic work: emit on a channel the task
subscribes to, and the orchestrator enqueues a run on its next tick.

    from events.event_bus import bus

    class SchedulerService(BaseService):
        model_name = "scheduler"

        def _load(self):
            import threading
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            return True

        def _loop(self):
            while not self._stop.wait(timeout=86400):   # every 24h
                bus.emit("schedule.tick.daily", {"source": "scheduler"})

        def unload(self):
            self._stop.set()
            self.loaded = False

The service never imports the orchestrator or the tasks — it just emits.
Any task declaring trigger_channels=["schedule.tick.daily"] will fire.


SHARED vs PER-CALL
------------------
  shared = True  (default) — One instance used by all threads.
                 Good for: thread-safe models (LLM, embedders).
                 Access directly: service.encode(text)

  shared = False           — Callers use get_client() for thread safety.
                 Good for: API clients with auth state (Google Drive).
                 Override get_client() to return a fresh client.

Choose the simplest access pattern that matches the service's real concurrency
model, and document it clearly in method names and comments.


CONFIG SETTINGS
---------------
Services can declare config settings that appear in the Settings UI and are
stored in plugin_config.json. Values are passed to build_services(config).

  config_settings = [
      ("Whisper Model", "whisper_model_name",
       "Model size for transcription.",
       "base",
       {"type": "text"}),
  ]

Each entry is a tuple: (title, variable_name, description, default, type_info)

type_info controls the UI widget:
  {"type": "text"}                                          — text field
  {"type": "bool"}                                          — checkbox
  {"type": "json_list"}                                     — JSON array editor
  {"type": "slider", "range": (min, max, divs), "is_float": False} — slider

Multiple plugins can declare the same variable_name — the value is shared.
In build_services(), access via: config.get("whisper_model_name", "base")
"""

# =====================================================================
# BASE CLASS (copied from plugins/BaseService.py for self-containment)
# =====================================================================

import logging
import time
from abc import ABC, abstractmethod


class BaseService(ABC):
    model_name: str = ""    # human-readable name shown in frontends
    shared: bool = True     # True = one instance for all threads
    config_settings: list = []  # settings shown in the Settings UI

    def __init__(self):
        self._loaded = False
        self.services = {}  # Every service gets the full registry for peer access, but use it wisely to avoid tight coupling.

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

    def set_peer_services(self, services: dict):
        """Receive the live runtime service registry."""
        self.services = services


# =====================================================================
# EXAMPLE: A simple shared service (e.g. audio transcription)
# =====================================================================

# import gc
# import os
# from pathlib import Path
# from plugins.BaseService import BaseService
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

# from plugins.BaseService import BaseService
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
