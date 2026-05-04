import logging
import threading

from frontend.platforms.platform_telegram import TelegramPlatformAdapter
from frontend.runtime import FrontendRuntime

logger = logging.getLogger("Frontends")

_ADAPTERS = {
    "telegram": TelegramPlatformAdapter,
}


def start_frontends(frontends: set[str], ctrl, shutdown_fn, shutdown_event,
                    tool_registry, services, config, root_dir):
    runtime = FrontendRuntime(ctrl, services, config, tool_registry, root_dir)
    threads = []
    adapters = {}
    for name in sorted(frontends):
        adapter_cls = _ADAPTERS.get(name)
        if adapter_cls is None:
            logger.warning(f"Unknown frontend '{name}' — skipping.")
            continue
        try:
            adapter = adapter_cls(
                ctrl, shutdown_fn, shutdown_event, tool_registry, services, config, root_dir
            )
        except Exception as e:
            logger.warning(f"Failed to initialize frontend '{name}': {e}")
            continue
        runtime.register_adapter(adapter)
        thread = threading.Thread(target=adapter.start, daemon=True, name=f"{name}-frontend")
        thread.start()
        adapters[name] = adapter
        threads.append(thread)
    return runtime, adapters, threads
