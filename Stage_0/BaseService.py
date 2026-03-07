import logging
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
        unload()      Release resources. Safe to call multiple times.
        loaded        Property indicating whether the service is ready.

    Per-call services (shared = False):
        Override get_client() to return a fresh client for each caller.
        The base implementation raises NotImplementedError as a reminder.
    """

    model_name: str = ""
    shared: bool = True

    def __init__(self):
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @loaded.setter
    def loaded(self, value: bool):
        self._loaded = value

    @abstractmethod
    def load(self) -> bool:
        """Initialize the service. Return True on success, False on failure."""
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