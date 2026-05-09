import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from plugins.BaseService import BaseService
from plugins.helpers.plugin_paths import iter_plugin_dirs, plugin_info
from plugins.plugin_discovery import get_plugin_settings, load_single_plugin, unload_plugin, wire_peer_services

logger = logging.getLogger("PluginWatcher")


class PluginWatcherService(BaseService):
    model_name = "Plugin Watcher"
    shared = True

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.observer = None
        self._known_mtimes: dict[str, float] = {}
        self._runtime = {}
        self._lock = threading.RLock()

    def bind_runtime(self, *, tool_registry=None, orchestrator=None, command_registry=None, frontend_manager=None):
        self._runtime.update({
            "tool_registry": tool_registry,
            "orchestrator": orchestrator,
            "command_registry": command_registry,
            "frontend_manager": frontend_manager,
        })

    def _load(self) -> bool:
        self.observer = Observer()
        handler = _PluginEventHandler(self)
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
        observer = self.observer
        if observer and observer.is_alive():
            observer.stop()
            observer.join(timeout=5.0)
        self.observer = None
        self.loaded = False

    def _scan_existing(self):
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
        self._load_plugin(path)

    def handle_delete(self, raw_path: str):
        path = Path(raw_path).resolve()
        key = str(path)
        with self._lock:
            known = self._known_mtimes.pop(key, None)
        if known is not None or path.suffix == ".py":
            self._unload_plugin(path)

    def _load_plugin(self, path: Path):
        info, err = plugin_info(path)
        if err:
            logger.warning(f"Plugin watcher skipped {path}: {err}")
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
            return
        if info.plugin_type == "service":
            wire_peer_services(self.services)
        self._reconcile_plugin_config()
        logger.info(f"Plugin watcher loaded {info.plugin_type}: {name}")

    def _unload_plugin(self, path: Path):
        info, err = plugin_info(path)
        if err:
            logger.warning(f"Plugin watcher could not infer deleted plugin {path}: {err}")
            return
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
        logger.info(f"Plugin watcher unloaded deleted {info.plugin_type}: {path.name}")

    def _reconcile_plugin_config(self):
        try:
            import config.config_manager as cm
            cm.reconcile_plugin_config(self.config, get_plugin_settings())
        except Exception as e:
            logger.warning(f"Plugin watcher config reconcile failed: {e}")


class _PluginEventHandler(FileSystemEventHandler):
    def __init__(self, watcher: PluginWatcherService):
        self.watcher = watcher
        self.pending: dict[str, threading.Timer] = {}
        self.lock = threading.Lock()
        self.debounce_interval = 1.0

    def _debounce(self, path: str):
        with self.lock:
            timer = self.pending.pop(path, None)
            if timer:
                timer.cancel()
            timer = threading.Timer(self.debounce_interval, self._fire, [path])
            self.pending[path] = timer
            timer.start()

    def _fire(self, path: str):
        with self.lock:
            self.pending.pop(path, None)
        self.watcher.handle_create_or_modify(path)

    def on_created(self, event):
        if not event.is_directory:
            self._debounce(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._debounce(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.watcher.handle_delete(event.src_path)
            self._debounce(event.dest_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.watcher.handle_delete(event.src_path)


def build_services(config: dict) -> dict:
    return {"plugin_watcher": PluginWatcherService(config)}
