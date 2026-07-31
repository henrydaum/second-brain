"""Kernel-owned plugin discovery, diagnostics, and hot reload coordination."""

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
import trees
from plugins.plugin_paths import plugin_info
from plugins.plugin_discovery import get_plugin_settings, load_single_plugin, unload_plugin

logger = logging.getLogger("PluginWatcher")


class PluginWatcher:
    """Watch plugin trees and coordinate changes across live registries."""

    def __init__(
        self,
        config: dict,
        *,
        services=None,
        tool_registry=None,
        orchestrator=None,
        command_registry=None,
        frontend_manager=None,
        runtime=None,
    ):
        """Bind the kernel objects whose registries hot reload mutates."""
        self.config = config
        self.services = services if services is not None else {}
        self.observer = None
        self._handler = None
        self._known_mtimes: dict[str, float] = {}
        self._runtime = {
            "tool_registry": tool_registry,
            "orchestrator": orchestrator,
            "command_registry": command_registry,
            "frontend_manager": frontend_manager,
            "runtime": runtime,
        }
        self._lock = threading.RLock()
        self.started = False

    def bind_runtime(self, *, tool_registry=None, orchestrator=None, command_registry=None, frontend_manager=None, runtime=None, **_):
        """Refresh live kernel bindings before the watcher starts."""
        self._runtime.update({
            "tool_registry": tool_registry,
            "orchestrator": orchestrator,
            "command_registry": command_registry,
            "frontend_manager": frontend_manager,
            "runtime": runtime,
        })

    def start(self) -> bool:
        """Start watching after initial plugin discovery is complete."""
        if self.started:
            return True
        self.observer = Observer()
        handler = _PluginEventHandler(self)
        self._handler = handler
        watched = 0
        for tree, _root, directory in trees.iter_root_dirs(watched_only=True):
            if tree.builtin:
                # Never create a directory in the source tree. The app's own
                # roots ship in the repo, so an absent one means the app ships
                # nothing of that kind — materializing it puts empty folders in
                # a checkout at every boot, and git does not track them, so the
                # next person to delete them watches them come back.
                if not directory.is_dir():
                    continue
            else:
                directory.mkdir(parents=True, exist_ok=True)
            self.observer.schedule(handler, str(directory), recursive=False)
            watched += 1
        self._scan_existing()
        self.observer.start()
        self.started = True
        logger.info(f"Plugin watcher started on {watched} folder(s).")
        return True

    def stop(self):
        """Stop observation and discard pending reload batches."""
        if self._handler:
            self._handler.cancel_pending()
        observer = self.observer
        if observer and observer.is_alive():
            observer.stop()
            observer.join(timeout=5.0)
        self.observer = None
        self._handler = None
        self.started = False

    def _scan_existing(self):
        """Internal helper to handle scan existing."""
        with self._lock:
            self._known_mtimes.clear()
            # Every ``.py`` in every watched root, not just the ones matching a
            # prefix. Seeding only the registrable files meant anything else had
            # no baseline mtime, so its first save after startup looked like a
            # brand-new file.
            for _tree, _root, directory in trees.iter_root_dirs(watched_only=True):
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
        self.register(path, edited=old is not None)

    def handle_delete(self, raw_path: str):
        """Handle delete."""
        path = Path(raw_path).resolve()
        key = str(path)
        with self._lock:
            known = self._known_mtimes.pop(key, None)
        if known is not None or path.suffix == ".py":
            self.unregister(path)

    def register(self, raw_path, *, edited: bool = False) -> dict:
        """Load or reload one recognized plugin source file."""
        path = Path(raw_path).resolve()
        if not path.exists() or path.suffix != ".py":
            error = f"Plugin file does not exist: {path}"
            self._notify(f"✕ Plugin registration failed: {path.name}\n{error}")
            return {"ok": False, "error": error}
        root = self._root_of(path)
        if root == "llm":
            self._refresh_llm_backends()
            self._notify(
                f"✓ Registered LLM backend{' edit' if edited else ''}: "
                f"{path.stem}"
            )
            return {
                "ok": True, "name": path.stem, "family": "llm_backend",
                "path": str(path),
            }
        if root == "parsers":
            self._refresh_parsers()
            self._notify(
                f"✓ Registered parser{' edit' if edited else ''}: {path.stem}"
            )
            return {
                "ok": True, "name": path.stem, "family": "parser",
                "path": str(path),
            }
        info, err = plugin_info(path)
        if err:
            logger.warning(f"Plugin watcher skipped {path}: {err}")
            self._notify(f"✕ Plugin registration failed: {path.name}\n{err}")
            return {"ok": False, "error": err}
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
                runtime=self._runtime.get("runtime"),
            )
        except Exception as e:
            name, error = None, str(e)
        if error:
            logger.warning(f"Plugin watcher failed to load {path.name}: {error}")
            self._notify(f"✕ Plugin registration failed: {name or path.name}\n{error}")
            return {"ok": False, "error": error}
        if info.plugin_type == "command":
            self._refresh_commands()
        self._reconcile_plugin_config()
        self._notify(f"✓ Registered plugin{' edit' if edited else ''}: {name}")
        logger.info(f"Plugin watcher loaded {info.plugin_type}: {name}")
        return {
            "ok": True, "name": name, "family": info.plugin_type,
            "path": str(path),
        }

    def reload(self, raw_path) -> dict:
        """Reload one recognized plugin source file."""
        return self.register(raw_path, edited=True)

    def unregister(self, raw_path) -> dict:
        """Unload everything registered from one recognized source file."""
        path = Path(raw_path).resolve()
        root = self._root_of(path)
        if root == "llm":
            self._refresh_llm_backends()
            self._notify(f"Deregistered LLM backend: {path.stem}")
            return {
                "ok": True, "names": [path.stem], "family": "llm_backend",
                "path": str(path),
            }
        if root == "parsers":
            self._refresh_parsers()
            self._notify(f"Deregistered parser: {path.stem}")
            return {
                "ok": True, "names": [path.stem], "family": "parser",
                "path": str(path),
            }
        info, err = plugin_info(path)
        if err:
            logger.warning(f"Plugin watcher could not infer deleted plugin {path}: {err}")
            return {"ok": False, "error": err}
        if info.plugin_type == "frontend" and info.builtin:
            # git pull (e.g. /update) replaces files as delete+create; tearing
            # down a kernel frontend mid-churn kills the surface the user is
            # typing into. Built-in frontends only change on restart anyway.
            logger.info(f"Plugin watcher ignoring delete of built-in frontend {path.name} (restart applies changes).")
            return {
                "ok": True, "names": [], "family": info.plugin_type,
                "path": str(path), "restart_required": True,
            }
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
        if info.plugin_type == "service":
            self._refresh_llm_backends()
        if info.plugin_type == "command":
            self._refresh_commands()
        self._reconcile_plugin_config()
        for name in names:
            self._notify(f"Deregistered plugin: {name}")
        logger.info(f"Plugin watcher unloaded deleted {info.plugin_type}: {path.name}")
        return {
            "ok": True, "names": names, "family": info.plugin_type,
            "path": str(path),
        }

    def resolve_registered(self, name: str, family: str = "") -> tuple:
        """Resolve a registered name to one unambiguous source path."""
        normalized = (family or "").strip().lower().rstrip("s")
        families = (
            [normalized]
            if normalized
            else ["tool", "task", "service", "command", "frontend"]
        )
        matches = []
        for plugin_type in families:
            if plugin_type not in {
                "tool", "task", "service", "command", "frontend"
            }:
                return None, f"Unknown plugin family: {family}"
            item = self._items(plugin_type).get(name)
            source = getattr(item, "_source_path", "") if item else ""
            if source:
                matches.append((plugin_type, Path(source).resolve()))
        if not matches:
            return None, f"No registered plugin named '{name}'."
        if len(matches) > 1:
            kinds = ", ".join(kind for kind, _path in matches)
            return None, (
                f"Plugin name '{name}' is ambiguous across: {kinds}. "
                "Specify family."
            )
        return matches[0][1], None

    def _notify(self, message: str):
        """Internal helper to handle notify."""
        bus.emit(CHAT_MESSAGE_PUSHED, {"message": message, "kind": "plugin", "source": "plugin_watcher"})

    def _names_registered_from(self, plugin_type: str, path: Path) -> list[str]:
        """Internal helper to handle names registered from."""
        source = str(path.resolve())
        items = self._items(plugin_type)
        return [name for name, item in items.items() if getattr(item, "_source_path", "") == source]

    def _items(self, plugin_type: str) -> dict:
        """Return the live registry mapping for one plugin family."""
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
        return items

    def _reconcile_plugin_config(self):
        """Internal helper to handle reconcile plugin config."""
        try:
            import config.config_manager as cm
            cm.reconcile_plugin_config(self.config, get_plugin_settings())
        except Exception as e:
            logger.warning(f"Plugin watcher config reconcile failed: {e}")

    def _refresh_commands(self):
        runtime = self._runtime.get("runtime")
        registry = self._runtime.get("command_registry")
        if runtime and registry and hasattr(registry, "to_callable_specs"):
            runtime.commands = registry.to_callable_specs()
        if runtime and hasattr(runtime, "refresh_session_specs"):
            runtime.refresh_session_specs()

    def _refresh_llm_backends(self):
        """Rescan backends after a file changed.

        A backend is found by scanning declarations, so ``discover`` is the
        whole update. ``refresh(force=True)`` is what makes it take effect:
        the unforced path short-circuits on an unchanged *profile dict*, so a
        backend whose source changed would leave every open brain running the
        old code and report success.
        """
        try:
            import llm

            llm.discover()
            llm.refresh(self.config, force=True)
        except Exception:
            logger.exception("LLM backend rescan failed")

    def _refresh_parsers(self):
        """Rescan parsers after a file changed.

        The counterpart to ``_refresh_llm_backends``, and the same full
        idempotent rebuild ``package_manager`` runs on install — the watcher
        only ever did the LLM half, so editing a parser did nothing until
        restart.
        """
        try:
            import parsing

            parsing.discover()
        except Exception:
            logger.exception("parser rescan failed")

    @staticmethod
    def _root_of(path: Path) -> str | None:
        """Which tree root a changed file sits directly in, or None.

        The directory is the answer. This used to sniff the filename stem,
        because everything that was not one of the five families lived in a
        single ``helpers/`` folder and only the prefix could tell an LLM
        backend from a parser from an ordinary library — so a plain library
        file came back as "not in a known plugin folder", which put a red ✕ in
        the user's chat for saving one.

        Top level only: a family-local ``tools/helpers/x.py`` belongs to its
        plugin, not to a root, and is not watched at all.
        """
        if path.suffix != ".py":
            return None
        found = trees.locate(path)
        if found is None or found.root is None or len(found.rel.parts) != 1:
            return None
        return found.root.name


# Load priority: services must register before the tasks that require them, so
# a batch install (many files landing at once) doesn't leave tasks warning about
# missing services. Lower number = loaded first. Unknown types load last.
_LOAD_PRIORITY = {"service": 0, "task": 1, "tool": 2, "command": 3, "frontend": 4}

#: Roots a kernel registry scans rather than discovery, which is why a file in
#: one is hoisted to the front of a batch: it belongs to no family, so
#: ``plugin_info`` cannot rank it at all.
_RESCANNED_ROOTS = ("parsers", "llm")


class _PluginEventHandler(FileSystemEventHandler):
    """Plugin event handler.

    Filesystem events are coalesced into a single debounced batch. When the
    batch fires, all pending paths are loaded **on one thread, in priority
    order** (services first) — never concurrently. Serializing the loads is
    what keeps registry mutation off the dispatch/registration critical paths,
    and ordering is what spares dependent tasks the 'missing service' churn.
    """
    def __init__(self, watcher: PluginWatcher):
        """Initialize the plugin event handler."""
        self.watcher = watcher
        self.pending: set[str] = set()
        self.lock = threading.Lock()
        self.debounce_interval = 1.0
        self._timer: threading.Timer | None = None

    def _debounce(self, path: str):
        """Coalesce a create/modify into the next batch, resetting the timer."""
        with self.lock:
            self.pending.add(path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_interval, self._fire_batch)
            self._timer.daemon = True
            self._timer.start()

    def cancel_pending(self):
        """Cancel pending."""
        with self.lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            self.pending.clear()

    def _fire_batch(self):
        """Drain the pending set and load each plugin in priority order."""
        with self.lock:
            paths = list(self.pending)
            self.pending.clear()
            self._timer = None

        def _priority(raw_path: str) -> int:
            path = Path(raw_path)
            # Parsers and backends load first: either is what a plugin arriving
            # in the same batch may be about to look for.
            if self.watcher._root_of(path) in _RESCANNED_ROOTS:
                return _LOAD_PRIORITY["service"]
            info, err = plugin_info(path)
            return _LOAD_PRIORITY.get(info.plugin_type, len(_LOAD_PRIORITY)) if info and not err else len(_LOAD_PRIORITY)

        # Stable sort: services before tasks before everything else, ties keep
        # discovery order. Loads run one at a time on this single thread.
        for path in sorted(paths, key=_priority):
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
