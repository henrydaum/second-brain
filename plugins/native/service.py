"""The native face of a service adapter.

Nothing subclasses this by hand. A service is sandboxed code, and
``sandbox.bridge`` builds a subclass of this class at load whose ``_load``
opens a persistent box and whose ``unload`` closes it. What lives here is the
half the *kernel* needs to see: the lifecycle it drives, and the declarations
it reads. The behaviour is on the other side of the boundary, in the box.

Services are long-lived capabilities shared across tools and tasks. They wrap
models, schedulers, external APIs, runtime extensions, or other reusable state
and give the rest of the system a consistent lifecycle.
"""

import logging
import time
from abc import ABC

logger = logging.getLogger("BaseService")

MANAGED = "managed"
EXTENSION = "extension"


class BaseService(ABC):
    """
    What the kernel sees of a service.

    Class attributes:
        model_name:
            Human-readable name shown in frontends and service listings.
        lifecycle:
            "managed" services are user-loadable backends. "extension" services
            are runtime hook carriers that auto-load whenever installed.

    Lifecycle:
        load():
            Initialize the service. Returns True on success. Timing and basic
            logging are handled by the base class wrapper.
        unload():
            Release resources. Must be safe to call repeatedly.
        loaded:
            Property indicating whether the service is ready for use.
    """

    model_name: str = ""
    lifecycle: str = MANAGED

    # --- Config settings this plugin needs ---
    # Each entry is a tuple:
    # (title, variable_name, description, default, type_info)
    # Same format as SETTINGS_DATA in config_data.py.
    config_settings: list = []
    dependencies_files: list[str] = []
    dependencies_pip: list[str] = []

    def __init_subclass__(cls, **kwargs):
        """Internal helper to handle init subclass."""
        super().__init_subclass__(**kwargs)
        for attr in ("config_settings", "dependencies_files", "dependencies_pip"):
            value = getattr(cls, attr)
            if isinstance(value, (dict, list)):
                setattr(cls, attr, value.copy())

    # --- Agent system-prompt contribution ---
    # Guidance injected into the agent's system prompt when this service is loaded.
    # Declare a plain string, or override with ``def agent_prompt(self, ctx)``
    # when the text depends on the session (``ctx`` is a PromptContext:
    # db/services/orchestrator/config/scope/...). The collector accepts either.
    agent_prompt: str = ""

    def __init__(self):
        """Initialize the base service."""
        self._loaded = False

    @property
    def loaded(self) -> bool:
        """Handle loaded."""
        return self._loaded

    @loaded.setter
    def loaded(self, value: bool):
        """Handle loaded."""
        self._loaded = value

    def load(self) -> bool:
        """Wraps _load() with timing and logging. Subclasses override _load().

        There is no wall-clock timeout here. This used to run ``_load`` on a
        daemon worker and abandon it after ``load_timeout`` seconds, which
        mattered while a service could be arbitrary in-process code that hung.
        A service is a box now, and the box owns its own start deadline — two
        timers racing over one load is a worse answer than one.
        """
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

    def _load(self) -> bool:
        """Initialize the service. Return True on success, False on failure."""
        self.loaded = True
        return True

    def unload(self):
        """Release all resources. Must be safe to call even if not loaded."""
        self.loaded = False


def service_lifecycle(svc) -> str:
    """Return a service lifecycle, defaulting to managed."""
    return getattr(svc, "lifecycle", MANAGED) or MANAGED


def is_extension_service(svc) -> bool:
    """Whether a service is an installed runtime extension."""
    return service_lifecycle(svc) == EXTENSION


def is_user_managed_service(svc) -> bool:
    """Whether /services should offer load/unload controls."""
    return service_lifecycle(svc) == MANAGED


def should_autoload_service(name: str, svc, config: dict) -> bool:
    """Whether startup should load a service."""
    return is_extension_service(svc) or name in (config.get("autoload_services") or [])
