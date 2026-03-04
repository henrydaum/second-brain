"""
Service Manager.

Manages shared model lifecycles (embedder, LLM, OCR, etc.)
and per-instance service factories (download clients, etc.)

Two modes:
    manual  — User controls load/unload. Tasks whose services
              aren't loaded simply wait in the queue.
    auto    — System loads on demand before dispatch, optionally
              unloads after idle timeout.

Two service types:
    shared      — One instance in memory. Load once, use everywhere.
                  (Embedding model, LLM, OCR engine)
    per-instance — Fresh instance created each time a task requests it.
                  (Download clients, API sessions)

Usage:
    manager = ServiceManager(config)

    # Shared: pass an instance
    manager.register("embedder", embedder_instance, shared=True)

    # Per-instance: pass the CLASS (not an instance)
    manager.register("gdrive", GDriveDownloader, shared=False)

    # Tasks declare what they need
    class EmbedText(BaseTask):
        requires_services = ["embedder"]

    # Orchestrator checks before dispatch
    if manager.ensure_loaded(task.requires_services):
        # dispatch
"""

import logging
import threading
import time

logger = logging.getLogger("ServiceManager")


class ServiceManager:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("service_mode", "manual")  # "manual" or "auto"
        self._lock = threading.Lock()

        # Shared services: name -> instance (with .load()/.unload()/.loaded)
        self._shared: dict[str, object] = {}

        # Per-instance factories: name -> (class, kwargs)
        self._factories: dict[str, tuple[type, dict]] = {}

        # Track all registered names and their type
        self._registry: dict[str, str] = {}  # name -> "shared" or "factory"

        # Idle tracking for auto-unload
        self._last_used: dict[str, float] = {}

    # =================================================================
    # REGISTRATION
    # =================================================================

    def register(self, name: str, service, shared: bool = True, **factory_kwargs):
        """
        Register a service.

        shared=True:  service is an instance with load()/unload()/loaded.
        shared=False: service is a CLASS. A new instance is created per get().
                      factory_kwargs are passed to the constructor.
        """
        with self._lock:
            if shared:
                self._shared[name] = service
                self._registry[name] = "shared"
                loaded = getattr(service, 'loaded', False)
                logger.info(f"Registered shared service: {name} (loaded={loaded})")
            else:
                self._factories[name] = (service, factory_kwargs)
                self._registry[name] = "factory"
                logger.info(f"Registered factory service: {name}")

    # =================================================================
    # ACCESS
    # =================================================================

    def get(self, name: str):
        """
        Get a service by name.

        Shared:       returns the singleton instance.
        Per-instance: creates and returns a new instance.
        """
        stype = self._registry.get(name)
        if stype is None:
            logger.warning(f"Unknown service requested: {name}")
            return None

        if stype == "shared":
            self._last_used[name] = time.time()
            return self._shared.get(name)

        elif stype == "factory":
            cls, kwargs = self._factories[name]
            instance = cls(**kwargs) if kwargs else cls()
            logger.debug(f"Created new instance of factory service: {name}")
            return instance

    # =================================================================
    # LIFECYCLE (shared services only)
    # =================================================================

    def is_loaded(self, name: str) -> bool:
        """Check if a shared service is loaded. Factories are always 'available'."""
        stype = self._registry.get(name)
        if stype == "factory":
            return True  # always available
        if stype == "shared":
            svc = self._shared.get(name)
            return svc is not None and getattr(svc, 'loaded', False)
        return False

    def all_loaded(self, names: list[str]) -> bool:
        """Check if all named services are ready."""
        return all(self.is_loaded(n) for n in names)

    def load(self, name: str) -> bool:
        """
        Load a shared service. Returns True if successful.
        Logs timing because model loads can take 10-60+ seconds.
        """
        with self._lock:
            svc = self._shared.get(name)
            if svc is None:
                logger.error(f"Cannot load unknown service: {name}")
                return False

            if getattr(svc, 'loaded', False):
                logger.debug(f"Service '{name}' already loaded")
                return True

            logger.info(f"Loading service '{name}'...")
            start = time.time()
            try:
                success = svc.load()
                elapsed = time.time() - start
                if success:
                    logger.info(f"Service '{name}' loaded in {elapsed:.1f}s")
                    self._last_used[name] = time.time()
                else:
                    logger.error(f"Service '{name}' failed to load ({elapsed:.1f}s)")
                return success
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"Service '{name}' load crashed after {elapsed:.1f}s: {e}")
                return False

    def unload(self, name: str):
        """Unload a shared service to free resources (VRAM, RAM)."""
        with self._lock:
            svc = self._shared.get(name)
            if svc is None:
                return

            if not getattr(svc, 'loaded', False):
                logger.debug(f"Service '{name}' already unloaded")
                return

            logger.info(f"Unloading service '{name}'...")
            try:
                svc.unload()
                logger.info(f"Service '{name}' unloaded")
            except Exception as e:
                logger.error(f"Service '{name}' unload failed: {e}")

    def ensure_loaded(self, names: list[str]) -> bool:
        """
        Make sure all named services are ready.

        auto mode:   loads anything that isn't loaded. Returns False on failure.
        manual mode: just checks. Returns False if anything isn't loaded.

        This is the gate the orchestrator calls before dispatching a task.
        """
        if not names:
            return True  # no requirements, always ready

        if self.mode == "auto":
            for name in names:
                if not self.is_loaded(name):
                    success = self.load(name)
                    if not success:
                        logger.warning(
                            f"Auto-load failed for '{name}' — "
                            f"tasks requiring this service will be skipped"
                        )
                        return False
            return True
        else:
            # Manual mode — just check, don't load
            missing = [n for n in names if not self.is_loaded(n)]
            if missing:
                logger.debug(
                    f"Services not loaded (manual mode): {missing} — "
                    f"tasks will wait"
                )
                return False
            return True

    # =================================================================
    # AUTO-UNLOAD (optional, call from a maintenance loop)
    # =================================================================

    def unload_idle(self, idle_seconds: int = 300):
        """
        Unload shared services that haven't been used recently.
        Only relevant in auto mode. Call periodically from main loop.
        """
        if self.mode != "auto":
            return

        now = time.time()
        for name in list(self._shared.keys()):
            if not self.is_loaded(name):
                continue
            last = self._last_used.get(name, 0)
            if now - last > idle_seconds:
                logger.info(f"Auto-unloading idle service '{name}' "
                            f"(idle {now - last:.0f}s)")
                self.unload(name)

    # =================================================================
    # INFO
    # =================================================================

    def status(self) -> dict:
        """Return status of all services. Useful for GUI and debugging."""
        result = {}
        for name, stype in self._registry.items():
            if stype == "shared":
                svc = self._shared[name]
                result[name] = {
                    "type": "shared",
                    "loaded": getattr(svc, 'loaded', False),
                    "model_name": getattr(svc, 'model_name', ''),
                    "last_used": self._last_used.get(name),
                }
            else:
                result[name] = {
                    "type": "factory",
                    "loaded": True,  # always available
                }
        return result

    def list_services(self) -> list[str]:
        return list(self._registry.keys())

    def shutdown(self):
        """Unload all shared services. Call on app exit."""
        logger.info("Shutting down all services...")
        for name in list(self._shared.keys()):
            if self.is_loaded(name):
                self.unload(name)