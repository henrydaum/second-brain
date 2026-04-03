import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger("BaseService")


class BaseService(ABC):
    """
    The contract every service implements.

    Class attributes:
        model_name    Human-readable name shown in `services` command.
        shared        If True, one instance is used by all threads (LLM, embedder).
                      If False, callers should use get_client() for thread safety.

    Lifecycle:
        load()        Initialize the service. Returns True on success.
                      Timing is logged automatically by the base class wrapper.
        unload()      Release resources. Safe to call multiple times.
        loaded        Property indicating whether the service is ready.

    Per-call services (shared = False):
        Override get_client() to return a fresh client for each caller.
        The base implementation raises NotImplementedError as a reminder.
    """

    model_name: str = ""
    shared: bool = True

    # --- Config settings this plugin needs ---
    # List of tuples: (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr in ("config_settings",):
            value = getattr(cls, attr)
            if isinstance(value, (dict, list)):
                setattr(cls, attr, value.copy())

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
        logger.info(f"Loading model: {name}...")
        t0 = time.time()
        try:
            result = self._load()
            elapsed = time.time() - t0
            if result:
                logger.info(f"Model loaded: {name} ({elapsed:.2f}s)")
            else:
                logger.warning(f"Model failed to load: {name} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"Model crashed during load: {name} ({time.time() - t0:.2f}s): {e}")
            raise

    @abstractmethod
    def _load(self) -> bool:
        """Initialize the service model. Return True on success, False on failure."""
        ...

    @abstractmethod
    def unload(self):
        """Release all resources. Must be safe to call even if not loaded."""
        ...

    def get_client(self):
        """
        Return a fresh client instance for thread-safe per-call usage.

        Only meaningful for services where shared = False. Shared services
        should be accessed directly (e.g. service.encode(), service.invoke()).

        Raises NotImplementedError if the service is shared but someone
        tries to call get_client() anyway — that's a bug.
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