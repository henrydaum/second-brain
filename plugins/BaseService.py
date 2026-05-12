"""
Service interface.

Services are long-lived capabilities shared across tools and tasks.
They wrap models, schedulers, external APIs, or other reusable runtime
state and give the rest of the system a consistent lifecycle:
build, load, use, and unload.
"""

import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger("BaseService")


class BaseService(ABC):
    """
    The contract every service implements.

    Class attributes:
        model_name:
            Human-readable name shown in frontends and service listings.
        shared:
            If True, one instance is shared across threads.
            If False, callers should use get_client() for thread-safe access.

    Lifecycle:
        load():
            Initialize the service. Returns True on success. Timing and basic
            logging are handled by the base class wrapper.
        unload():
            Release resources. Must be safe to call repeatedly.
        loaded:
            Property indicating whether the service is ready for use.

    Per-call services (shared = False):
        Override get_client() to return a fresh client for each caller. The
        base implementation raises NotImplementedError to make misuse obvious.
    """

    model_name: str = ""
    shared: bool = True

    # --- Config settings this plugin needs ---
    # Each entry is a tuple:
    # (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []

    def __init_subclass__(cls, **kwargs):
        """Internal helper to handle init subclass."""
        super().__init_subclass__(**kwargs)
        for attr in ("config_settings",):
            value = getattr(cls, attr)
            if isinstance(value, (dict, list)):
                setattr(cls, attr, value.copy())

    def __init__(self):
        """Initialize the base service."""
        self._loaded = False
        self.services = {}

    @property
    def loaded(self) -> bool:
        """Handle loaded."""
        return self._loaded

    @loaded.setter
    def loaded(self, value: bool):
        """Handle loaded."""
        self._loaded = value

    def load(self) -> bool:
        """Wraps _load() with automatic timing. Subclasses override _load()."""
        name = self.model_name or self.__class__.__name__
        logger.info(f"Loading service: {name}...")
        t0 = time.time()
        try:
            result = self._load()
            elapsed = time.time() - t0
            if result:
                logger.info(f"Service loaded: {name} ({elapsed:.2f}s)")
            else:
                logger.warning(f"Service failed to load: {name} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Service crashed during load: {name} ({time.time() - t0:.2f}s): {e}")
            raise

    @abstractmethod
    def _load(self) -> bool:
        """Initialize the service. Return True on success, False on failure."""
        ...

    @abstractmethod
    def unload(self):
        """Release all resources. Must be safe to call even if not loaded."""
        ...

    def get_client(self):
        """
        Return a fresh client instance for thread-safe per-call usage.

        This only makes sense for services where shared = False. Shared services
        should be accessed directly (e.g. service.encode(), service.invoke()).

        Raises NotImplementedError when a shared service is used incorrectly,
        or when a per-call service forgets to implement get_client().
        """
        if self.shared:
            raise NotImplementedError(
                f"Service '{self.model_name}' is shared — access it directly, "
                f"don't call get_client()."
            )
        raise NotImplementedError(
            f"Service '{self.model_name}' is per-call (shared=False) but "
            f"doesn't implement get_client()."
        )

    def set_peer_services(self, services: dict):
        """Receive the live runtime service registry."""
        self.services = services
