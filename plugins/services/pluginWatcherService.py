"""Plugin watcher service for discovery, diagnostics, and hot reloads."""

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
from plugins.BaseService import BaseService
from plugins.helpers.plugin_paths import iter_plugin_dirs, plugin_info
from plugins.plugin_discovery import get_plugin_settings, load_single_plugin, unload_plugin, wire_peer_services

logger = logging.getLogger("PluginWatcher")


class PluginWatcherService(BaseService):
    """Plugin watcher service."""
    model_name = "Plugin Watcher"
    shared = True

    def __init__(self, config: dict):
        """Initialize the plugin watcher service."""
        super().__init__()
        self.config = config
        self.observer = None
        self._handler = None
        self._known_mtimes: dict[str, float] = {}
        self._runtime = {}
        self._lock = threading.RLock()

    def bind_runtime(self, *, tool_registry=None, orchestrator=None, command_registry=None, frontend_manager=None):
        """Bind runtime."""
        self._runtime.update({
            "tool_registry": tool_registry,
            "orchestrator": orchestrator,
            "command_registry": command_registry,
            "frontend_manager": frontend_manager,
        })

    def _load(self) -> bool:
        """Internal helper to load plugin watcher service."""
        self.observer = Observer()
        handler = _PluginEventHandler(self)
        self._handler = handler
        watched = 0
        for _plugin_type, directory in iter_plugin_dirs():
            directory.mkdir(parents=True, exist_ok=True)
            self.observer.schedule(handler, str(directory), recursive=False)
            watched += 1
        self._scan_existing()
        self.observer.start()
        self.loaded = True
        logger.info(f"Plugin watcher started on {watched} folder(s).")
        return True

    def unload(self):
        """Handle unload."""
        if self._handler:
            self._handler.cancel_pending()
        observer = self.observer
        if observer and observer.is_alive():
            observer.stop()
            observer.join(timeout=5.0)
        self.observer = None
        self._handler = None
        self.loaded = False

    def _scan_existing(self):
        """Internal helper to handle scan existing."""
        with self._lock:
            self._known_mtimes.clear()
            for _plugin_type, directory in iter_plugin_dirs():
                if not directory.exists():
                    continue
                for path in directory.glob("*.py"):
                    try:
                        self._known_mtimes[str(path.resolve())] = path.stat().st_mtime
                    except OSError:
                        pass

    def handle_create_or_modify(self, raw_path: str):
        """Handle create or modify."""
        path = Path(raw_path).resolve()
        if not path.exists() or path.suffix != ".py":
            return
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        key = str(path)
        with self._lock:
            old = self._known_mtimes.get(key)
            if old is not None and abs(mtime - old) < 0.1:
                return
            self._known_mtimes[key] = mtime
        self._load_plugin(path, edited=old is not None)

    def handle_delete(self, raw_path: str):
        """Handle delete."""
        path = Path(raw_path).resolve()
        key = str(path)
        with self._lock:
            known = self._known_mtimes.pop(key, None)
        if known is not None or path.suffix == ".py":
            self._unload_plugin(path)

    def _load_plugin(self, path: Path, edited: bool = False):
        """Internal helper to load plugin."""
        info, err = plugin_info(path)
        if err:
            logger.warning(f"Plugin watcher skipped {path}: {err}")
            self._notify(f"Plugin registration failed: {path.name}")
            return
        logger.info(f"Plugin watcher loading {info.plugin_type}: {path.name}")
        try:
            name, error = load_single_plugin(
                info.plugin_type, path,
                tool_registry=self._runtime.get("tool_registry"),
                orchestrator=self._runtime.get("orchestrator"),
                services=self.services,
                config=self.config,
                command_registry=self._runtime.get("command_registry"),
                frontend_manager=self._runtime.get("frontend_manager"),
            )
        except Exception as e:
            name, error = None, str(e)
        if error:
            logger.warning(f"Plugin watcher failed to load {path.name}: {error}")
            self._notify(f"Plugin registration failed: {name or path.name}")
            return
        if info.plugin_type == "service":
            wire_peer_services(self.services)
        self._reconcile_plugin_config()
        self._notify(f"Registered plugin{' edit' if edited else ''}: {name}")
        logger.info(f"Plugin watcher loaded {info.plugin_type}: {name}")

    def _unload_plugin(self, path: Path):
        """Internal helper to handle unload plugin."""
        info, err = plugin_info(path)
        if err:
            logger.warning(f"Plugin watcher could not infer deleted plugin {path}: {err}")
            return
        names = self._names_registered_from(info.plugin_type, path)
        unload_plugin(
            info.plugin_type, "",
            tool_registry=self._runtime.get("tool_registry"),
            orchestrator=self._runtime.get("orchestrator"),
            services=self.services,
            source_path=str(path),
            command_registry=self._runtime.get("command_registry"),
            frontend_manager=self._runtime.get("frontend_manager"),
        )
        self._reconcile_plugin_config()
        for name in names:
            self._notify(f"Unregistered plugin: {name}")
        logger.info(f"Plugin watcher unloaded deleted {info.plugin_type}: {path.name}")

    def _notify(self, message: str):
        """Internal helper to handle notify."""
        bus.emit(CHAT_MESSAGE_PUSHED, {"message": message, "kind": "plugin", "source": "plugin_watcher"})

    def _names_registered_from(self, plugin_type: str, path: Path) -> list[str]:
        """Internal helper to handle names registered from."""
        source = str(path.resolve())
        if plugin_type == "tool":
            items = getattr(self._runtime.get("tool_registry"), "tools", {})
        elif plugin_type == "task":
            items = getattr(self._runtime.get("orchestrator"), "tasks", {})
        elif plugin_type == "command":
            registry = self._runtime.get("command_registry") or getattr(self._runtime.get("tool_registry"), "command_registry", None)
            items = getattr(registry, "_commands", {})
        elif plugin_type == "service":
            items = self.services
        elif plugin_type == "frontend":
            items = {k: v.__class__ for k, v in getattr(self._runtime.get("frontend_manager"), "adapters", {}).items()}
        else:
            items = {}
        return [name for name, item in items.items() if getattr(item, "_source_path", "") == source]

    def _reconcile_plugin_config(self):
        """Internal helper to handle reconcile plugin config."""
        try:
            import config.config_manager as cm
            cm.reconcile_plugin_config(self.config, get_plugin_settings())
        except Exception as e:
            logger.warning(f"Plugin watcher config reconcile failed: {e}")


class _PluginEventHandler(FileSystemEventHandler):
    """Plugin event handler."""
    def __init__(self, watcher: PluginWatcherService):
        """Initialize the plugin event handler."""
        self.watcher = watcher
        self.pending: dict[str, threading.Timer] = {}
        self.lock = threading.Lock()
        self.debounce_interval = 1.0

    def _debounce(self, path: str):
        """Internal helper to handle debounce."""
        with self.lock:
            timer = self.pending.pop(path, None)
            if timer:
                timer.cancel()
            timer = threading.Timer(self.debounce_interval, self._fire, [path])
            timer.daemon = True
            self.pending[path] = timer
            timer.start()

    def cancel_pending(self):
        """Cancel pending."""
        with self.lock:
            for timer in self.pending.values():
                timer.cancel()
            self.pending.clear()

    def _fire(self, path: str):
        """Internal helper to handle fire."""
        with self.lock:
            self.pending.pop(path, None)
        self.watcher.handle_create_or_modify(path)

    def on_created(self, event):
        """Handle on created."""
        if not event.is_directory:
            self._debounce(event.src_path)

    def on_modified(self, event):
        """Handle on modified."""
        if not event.is_directory:
            self._debounce(event.src_path)

    def on_moved(self, event):
        """Handle on moved."""
        if not event.is_directory:
            self.watcher.handle_delete(event.src_path)
            self._debounce(event.dest_path)

    def on_deleted(self, event):
        """Handle on deleted."""
        if not event.is_directory:
            self.watcher.handle_delete(event.src_path)


def build_services(config: dict) -> dict:
    """Build services."""
    return {"plugin_watcher": PluginWatcherService(config)}
