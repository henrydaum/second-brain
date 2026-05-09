"""
Plugin discovery — unified loader for tools, tasks, services, and commands.

Handles both baked-in (read-only, in source tree) and sandbox (mutable,
in DATA_DIR) plugins. Used at startup for bulk discovery and by
build_plugin for single-file load/unload at runtime.

Public API:
    discover_all()          — startup convenience, discovers everything
    discover_tools()        — tools only
    discover_tasks()        — tasks only
    discover_services()     — services only, returns dict
    discover_commands()     — commands only
    load_single_plugin()    — load one sandbox file and register it
    unload_plugin()         — unregister a plugin by name
"""

import importlib
import importlib.util
import inspect
import logging
import sys
import time
from pathlib import Path

from plugins.helpers.plugin_paths import PLUGIN_CONFIG, plugin_info

logger = logging.getLogger("Discovery")


# ── Plugin config settings accumulator ──────────────────────────────

_plugin_settings: list = []        # collected (title, var, desc, default, type_info) tuples
_plugin_settings_keys: set = set() # variable_names already seen (first-wins dedup)
_plugin_setting_types: dict[str, str] = {}  # variable_name -> plugin type that first declared it

# Reverse map: setting variable_name -> set of service names that declared it.
# Only populated for services (not tools/tasks), enabling targeted reloads.
_setting_to_services: dict[str, set[str]] = {}


def get_plugin_settings() -> list:
    """Return the accumulated plugin config settings (read-only copy)."""
    return list(_plugin_settings)


def get_setting_service_map() -> dict[str, set[str]]:
    """Return a copy of the setting_key -> {service_names} map."""
    return {k: set(v) for k, v in _setting_to_services.items()}


def _collect_config_settings(source, service_names: list[str] | None = None,
                             plugin_type: str | None = None):
    """Extract config_settings from a plugin instance or module and accumulate.
    Deduplicates by variable_name — first plugin to declare a key wins.

    If *service_names* is provided, each setting key is also recorded in
    the _setting_to_services reverse map so we know which services to
    rebuild when a setting changes.
    """
    settings = getattr(source, "config_settings", None)
    if not settings:
        return
    for entry in settings:
        if not isinstance(entry, (list, tuple)) or len(entry) != 5:
            continue
        var_name = entry[1]
        if var_name not in _plugin_settings_keys:
            _plugin_settings_keys.add(var_name)
            _plugin_settings.append(tuple(entry))
            if plugin_type:
                _plugin_setting_types[var_name] = plugin_type
        # Always record the service mapping (even if settings deduped)
        if service_names:
            _setting_to_services.setdefault(var_name, set()).update(service_names)


def _purge_plugin_settings(plugin_types: set[str]):
    """Remove accumulated settings for the given plugin types.

    Used before a full rediscovery so deleted plugins don't leave stale
    settings behind in the runtime config UI.
    """
    if not plugin_types:
        return

    kept = []
    kept_keys = set()
    kept_types = {}

    for entry in _plugin_settings:
        var_name = entry[1]
        owner_type = _plugin_setting_types.get(var_name)
        if owner_type in plugin_types:
            _setting_to_services.pop(var_name, None)
            continue
        kept.append(entry)
        kept_keys.add(var_name)
        if owner_type:
            kept_types[var_name] = owner_type

    _plugin_settings[:] = kept
    _plugin_settings_keys.clear()
    _plugin_settings_keys.update(kept_keys)
    _plugin_setting_types.clear()
    _plugin_setting_types.update(kept_types)


# ── Per-type configuration ───────────────────────────────────────────

def _discovery_config(plugin_type: str) -> dict:
    built_dir, sandbox_dir, prefix, namespaces = PLUGIN_CONFIG[plugin_type]
    return {
        "baked_in_dir": built_dir,
        "sandbox_dir": sandbox_dir,
        "glob": f"{prefix}*.py" if prefix else "*.py",
        "baked_in_ns": namespaces[0],
        "sandbox_ns": namespaces[1],
    }


_TOOL_CONFIG = _discovery_config("tool")
_TASK_CONFIG = _discovery_config("task")
_SERVICE_CONFIG = _discovery_config("service")
_COMMAND_CONFIG = _discovery_config("command")
_FRONTEND_CONFIG = _discovery_config("frontend")


# ── Bulk discovery (startup) ─────────────────────────────────────────

def discover_all(root_dir: Path, tool_registry, orchestrator, config: dict) -> dict:
    """Discover all plugins. Returns the services dict."""
    discover_tools(root_dir, tool_registry, config)
    discover_tasks(root_dir, orchestrator, config)
    return discover_services(root_dir, config)


def discover_commands(root_dir: Path, command_registry, config: dict | None = None, reload: bool = False):
    """Discover and register all slash commands (baked-in + sandbox)."""
    from plugins.BaseCommand import BaseCommand
    cfg = _COMMAND_CONFIG
    t0 = time.time()
    count = 0
    baked_in_names = set()

    if reload:
        _purge_plugin_settings({"command"})

    if cfg["baked_in_dir"].exists():
        for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
            module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
            module = _load_baked_in(module_name, reload)
            if module is None:
                continue
            for instance in _find_subclass_instances(module, BaseCommand, module_name):
                if not getattr(instance, "name", ""):
                    continue
                instance._source_path = _source_path(py_file)
                command_registry.register(instance)
                _collect_config_settings(instance, plugin_type="command")
                baked_in_names.add(instance.name)
                count += 1

    if cfg["sandbox_dir"].exists():
        for py_file in sorted(cfg["sandbox_dir"].glob(cfg["glob"])):
            module_name = cfg["sandbox_ns"].format(stem=py_file.stem)
            module = _load_sandbox(module_name, py_file, reload)
            if module is None:
                continue
            for instance in _find_subclass_instances(module, BaseCommand, module_name):
                if not getattr(instance, "name", ""):
                    continue
                if instance.name in baked_in_names:
                    logger.warning(f"Sandbox command '{instance.name}' collides with baked-in — skipped")
                    continue
                instance._source_path = _source_path(py_file)
                command_registry.register(instance)
                _collect_config_settings(instance, plugin_type="command")
                count += 1

    logger.info(f"Discovered {count} command(s) in {time.time() - t0:.2f}s")


def discover_frontends(root_dir: Path, config: dict | None = None, reload: bool = False) -> dict[str, type]:
    """Discover frontend plugin classes (baked-in + sandbox).

    Returns ``{frontend_name: cls}``. Frontends are instantiated by the
    bootstrap layer (which supplies transport-specific constructor args)
    rather than at discovery time, so this returns classes — unlike the
    other discoverers which return instances.
    """
    from plugins.BaseFrontend import BaseFrontend
    cfg = _FRONTEND_CONFIG
    t0 = time.time()
    found: dict[str, type] = {}
    baked_in_names: set[str] = set()

    if reload:
        _purge_plugin_settings({"frontend"})

    if cfg["baked_in_dir"].exists():
        for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
            module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
            module = _load_baked_in(module_name, reload)
            if module is None:
                continue
            for cls in _find_subclasses(module, BaseFrontend, module_name):
                name = getattr(cls, "name", "") or ""
                if not name:
                    continue
                cls._source_path = _source_path(py_file)
                found[name] = cls
                _collect_config_settings(cls, plugin_type="frontend")
                baked_in_names.add(name)

    if cfg["sandbox_dir"].exists():
        for py_file in sorted(cfg["sandbox_dir"].glob(cfg["glob"])):
            module_name = cfg["sandbox_ns"].format(stem=py_file.stem)
            module = _load_sandbox(module_name, py_file, reload)
            if module is None:
                continue
            for cls in _find_subclasses(module, BaseFrontend, module_name):
                name = getattr(cls, "name", "") or ""
                if not name:
                    continue
                if name in baked_in_names:
                    logger.warning(f"Sandbox frontend '{name}' collides with baked-in — skipped")
                    continue
                cls._source_path = _source_path(py_file)
                found[name] = cls
                _collect_config_settings(cls, plugin_type="frontend")

    logger.info(f"Discovered {len(found)} frontend(s) in {time.time() - t0:.2f}s")
    return found


def discover_tools(root_dir: Path, tool_registry, config: dict, reload: bool = False):
    """Discover and register all tools (baked-in + sandbox)."""
    from plugins.BaseTool import BaseTool
    cfg = _TOOL_CONFIG
    t0 = time.time()
    count = 0
    baked_in_names = set()

    if reload:
        _purge_plugin_settings({"tool"})

    # Baked-in
    for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
        module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
        module = _load_baked_in(module_name, reload)
        if module is None:
            continue
        for instance in _find_subclass_instances(module, BaseTool, module_name):
            instance._source_path = _source_path(py_file)
            tool_registry.register(instance)
            _collect_config_settings(instance, plugin_type="tool")
            baked_in_names.add(instance.name)
            count += 1

    # Sandbox
    if cfg["sandbox_dir"].exists():
        for py_file in sorted(cfg["sandbox_dir"].glob(cfg["glob"])):
            module_name = cfg["sandbox_ns"].format(stem=py_file.stem)
            module = _load_sandbox(module_name, py_file, reload)
            if module is None:
                continue
            for instance in _find_subclass_instances(module, BaseTool, module_name):
                if instance.name in baked_in_names:
                    logger.warning(f"Sandbox tool '{instance.name}' collides with baked-in — skipped")
                    continue
                instance._source_path = _source_path(py_file)
                tool_registry.register(instance)
                _collect_config_settings(instance, plugin_type="tool")
                count += 1

    logger.info(f"Discovered {count} tool(s) in {time.time() - t0:.2f}s")


def discover_tasks(root_dir: Path, orchestrator, config: dict, reload: bool = False):
    """Discover and register all tasks (baked-in + sandbox)."""
    from plugins.BaseTask import BaseTask
    cfg = _TASK_CONFIG
    t0 = time.time()
    count = 0
    baked_in_names = set()

    if reload:
        _purge_plugin_settings({"task"})

    # Baked-in
    for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
        module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
        module = _load_baked_in(module_name, reload)
        if module is None:
            continue
        for instance in _find_subclass_instances(module, BaseTask, module_name):
            instance._source_path = _source_path(py_file)
            orchestrator.register_task(instance)
            _collect_config_settings(instance, plugin_type="task")
            baked_in_names.add(instance.name)
            count += 1

    # Sandbox
    if cfg["sandbox_dir"].exists():
        for py_file in sorted(cfg["sandbox_dir"].glob(cfg["glob"])):
            module_name = cfg["sandbox_ns"].format(stem=py_file.stem)
            module = _load_sandbox(module_name, py_file, reload)
            if module is None:
                continue
            for instance in _find_subclass_instances(module, BaseTask, module_name):
                if instance.name in baked_in_names:
                    logger.warning(f"Sandbox task '{instance.name}' collides with baked-in — skipped")
                    continue
                instance._source_path = _source_path(py_file)
                orchestrator.register_task(instance)
                _collect_config_settings(instance, plugin_type="task")
                count += 1

    logger.info(f"Discovered {count} task(s) in {time.time() - t0:.2f}s")


def discover_services(root_dir: Path, config: dict) -> dict:
    """Discover all services (baked-in + sandbox). Returns {name: instance}."""
    _setting_to_services.clear()
    _purge_plugin_settings({"service"})
    cfg = _SERVICE_CONFIG
    t0 = time.time()
    services = {}
    baked_in_names = set()

    # Baked-in
    for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
        if py_file.stem.startswith("_"):
            continue
        module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
        module = _load_baked_in(module_name, reload=False)
        if module is None:
            continue
        built = _call_build_services(module, module_name, config)
        built_names = list(built.keys())
        for svc_name, svc in built.items():
            svc._source_path = _source_path(py_file)
            _collect_config_settings(svc, service_names=built_names, plugin_type="service")
            services[svc_name] = svc
            baked_in_names.add(svc_name)

    # Sandbox
    if cfg["sandbox_dir"].exists():
        for py_file in sorted(cfg["sandbox_dir"].glob(cfg["glob"])):
            if py_file.stem.startswith("_"):
                continue
            module_name = cfg["sandbox_ns"].format(stem=py_file.stem)
            module = _load_sandbox(module_name, py_file, reload=False)
            if module is None:
                continue
            built = _call_build_services(module, module_name, config)
            built_names = [n for n in built if n not in baked_in_names]
            for svc_name, svc in built.items():
                if svc_name in baked_in_names:
                    logger.warning(f"Sandbox service '{svc_name}' collides with baked-in — skipped")
                    continue
                svc._source_path = _source_path(py_file)
                _collect_config_settings(svc, service_names=built_names, plugin_type="service")
                services[svc_name] = svc

    wire_peer_services(services)
    logger.info(f"Discovered {len(services)} service(s) in {time.time() - t0:.2f}s")
    return services


def wire_peer_services(services: dict):
    """Inject the live service registry into every service."""
    for svc in list(services.values()):
        if hasattr(svc, "set_peer_services"):
            svc.set_peer_services(services)


# ── Single-plugin load/unload (used by build_plugin) ────────────────

def load_single_plugin(plugin_type: str, file_path: Path,
                       tool_registry=None, orchestrator=None,
                       services: dict = None, config: dict = None,
                       command_registry=None, frontend_manager=None) -> tuple[str | None, str | None]:
    """
    Load a single sandbox plugin file and register it.

    Returns (plugin_name, error_message).
    On success: (name, None). On failure: (None, error_string).
    """
    if plugin_type == "tool":
        return _load_single_tool(file_path, tool_registry)
    elif plugin_type == "task":
        return _load_single_task(file_path, orchestrator, config)
    elif plugin_type == "service":
        return _load_single_service(file_path, services, config)
    elif plugin_type == "command":
        return _load_single_command(file_path, command_registry or getattr(tool_registry, "command_registry", None))
    elif plugin_type == "frontend":
        return _load_single_frontend(file_path, frontend_manager)
    else:
        return None, f"Unknown plugin_type: {plugin_type}"


def unload_plugin(plugin_type: str, plugin_name: str,
                  tool_registry=None, orchestrator=None,
                  services: dict = None, source_path: str = None,
                  command_registry=None, frontend_manager=None):
    """Unregister a plugin. For services, uses source_path to find all
    service names registered from that file."""
    if plugin_type == "tool" and tool_registry:
        for name in _names_by_source(getattr(tool_registry, "tools", {}), plugin_name, source_path):
            tool_registry.unregister(name)
    elif plugin_type == "task" and orchestrator:
        for name in _names_by_source(getattr(orchestrator, "tasks", {}), plugin_name, source_path):
            orchestrator.unregister_task(name)
    elif plugin_type == "command" and (command_registry or getattr(tool_registry, "command_registry", None)):
        registry = command_registry or tool_registry.command_registry
        for name in _names_by_source(getattr(registry, "_commands", {}), plugin_name, source_path):
            registry.unregister(name)
    elif plugin_type == "service" and services:
        if source_path:
            _unload_services_by_source(services, source_path)
        else:
            _unload_service_by_name(services, plugin_name)
    elif plugin_type == "frontend" and frontend_manager:
        adapters = getattr(frontend_manager, "adapters", {})
        for name in _names_by_source({k: v.__class__ for k, v in adapters.items()}, plugin_name, source_path):
            frontend_manager.unregister(name)


def _names_by_source(items: dict, plugin_name: str, source_path: str | None) -> list[str]:
    if source_path:
        source = _source_path(source_path)
        return [name for name, item in items.items() if _source_path(getattr(item, "_source_path", "")) == source]
    return [plugin_name] if plugin_name else []


def _unload_services_by_source(services: dict, source_path: str):
    """Find all services registered from a source file, unload and remove them."""
    source = _source_path(source_path)
    to_remove = [
        name for name, svc in services.items()
        if _source_path(getattr(svc, "_source_path", "")) == source
    ]
    for name in to_remove:
        svc = services.pop(name)
        if hasattr(svc, "unload") and getattr(svc, "loaded", False):
            try:
                svc.unload()
                logger.info(f"Unloaded service: {name}")
            except Exception as e:
                logger.error(f"Error unloading service '{name}': {e}")
        logger.info(f"Unregistered service: {name}")


def _unload_service_by_name(services: dict, plugin_name: str):
    svc = services.pop(plugin_name, None)
    if svc and hasattr(svc, "unload") and getattr(svc, "loaded", False):
        try:
            svc.unload()
            logger.info(f"Unloaded service: {plugin_name}")
        except Exception as e:
            logger.error(f"Error unloading service '{plugin_name}': {e}")
    if svc:
        logger.info(f"Unregistered service: {plugin_name}")


def _load_single_tool(file_path: Path, tool_registry) -> tuple[str | None, str | None]:
    from plugins.BaseTool import BaseTool
    info, err = plugin_info(file_path)
    if err:
        return None, err
    module_name = info.module_name

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    instances = _find_subclass_instances(module, BaseTool, module_name)
    if not instances:
        return None, f"No BaseTool subclass found in {file_path.name}"

    instance = next((item for item in instances if getattr(item, "name", "")), None)
    if instance is None:
        return None, f"No named BaseTool subclass found in {file_path.name}"
    instance._source_path = _source_path(file_path)
    tool_registry.register(instance)
    _collect_config_settings(instance, plugin_type="tool")
    return instance.name, None


def _load_single_frontend(file_path: Path, frontend_manager) -> tuple[str | None, str | None]:
    from plugins.BaseFrontend import BaseFrontend
    info, err = plugin_info(file_path)
    if err:
        return None, err
    module_name = info.module_name
    if frontend_manager is None:
        return None, "No frontend manager available"

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    classes = _find_subclasses(module, BaseFrontend, module_name)
    classes = [cls for cls in classes if getattr(cls, "name", "")]
    if not classes:
        return None, f"No named BaseFrontend subclass found in {file_path.name}"

    cls = classes[0]
    cls._source_path = _source_path(file_path)
    _collect_config_settings(cls, plugin_type="frontend")
    err = frontend_manager.register(cls)
    if err:
        return None, err
    return cls.name, None


def _load_single_command(file_path: Path, command_registry) -> tuple[str | None, str | None]:
    from plugins.BaseCommand import BaseCommand
    info, err = plugin_info(file_path)
    if err:
        return None, err
    module_name = info.module_name
    if command_registry is None:
        return None, "No command registry available"

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    instances = _find_subclass_instances(module, BaseCommand, module_name)
    if not instances:
        return None, f"No BaseCommand subclass found in {file_path.name}"

    instance = instances[0]
    instance._source_path = _source_path(file_path)
    command_registry.register(instance)
    _collect_config_settings(instance, plugin_type="command")
    return instance.name, None


def _load_single_task(file_path: Path, orchestrator, config: dict) -> tuple[str | None, str | None]:
    from plugins.BaseTask import BaseTask
    info, err = plugin_info(file_path)
    if err:
        return None, err
    module_name = info.module_name

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    instances = _find_subclass_instances(module, BaseTask, module_name)
    if not instances:
        return None, f"No BaseTask subclass found in {file_path.name}"

    instance = instances[0]
    instance._source_path = _source_path(file_path)
    orchestrator.register_task(instance)
    _collect_config_settings(instance, plugin_type="task")
    return instance.name, None


def _load_single_service(file_path: Path, services: dict, config: dict) -> tuple[str | None, str | None]:
    info, err = plugin_info(file_path)
    if err:
        return None, err
    module_name = info.module_name

    # Unload any existing services from this file first (frees models/GPU)
    source = _source_path(file_path)
    was_loaded = {
        name for name, svc in services.items()
        if getattr(svc, "_source_path", None) == source and getattr(svc, "loaded", False)
    }
    runtime_bindings = {
        name: dict(getattr(svc, "_runtime", {}) or {})
        for name, svc in services.items()
        if getattr(svc, "_source_path", None) == source
    }
    _unload_services_by_source(services, _source_path(file_path))

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    built = _call_build_services(module, module_name, config)
    if not built:
        return None, f"build_services() returned nothing in {file_path.name}"

    names = list(built.keys())
    for svc_name, svc in built.items():
        svc._source_path = _source_path(file_path)
        _collect_config_settings(svc, service_names=names, plugin_type="service")
        services[svc_name] = svc
        if runtime_bindings.get(svc_name) and hasattr(svc, "bind_runtime"):
            svc.bind_runtime(**runtime_bindings[svc_name])
        if svc_name in was_loaded:
            try:
                svc.load()
            except Exception as e:
                return None, f"Reloaded service '{svc_name}' failed to load: {e}"

    wire_peer_services(services)
    return ", ".join(names), None


def _source_path(path) -> str:
    return str(Path(path).resolve()) if path else ""


# ── Internal helpers ─────────────────────────────────────────────────

def _load_baked_in(module_name: str, reload: bool):
    """Load a baked-in module via importlib.import_module."""
    try:
        if reload and module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)
    except ImportError as e:
        logger.warning(f"Could not import {module_name}: {e}")
    except Exception as e:
        logger.error(f"Failed to load {module_name}: {e}", exc_info=True)
    return None


def _load_sandbox(module_name: str, file_path: Path, reload: bool):
    """Load a sandbox module via spec_from_file_location.

    Always uses spec_from_file_location (never importlib.reload) because
    reload() can't re-find specs for modules loaded this way.
    """
    try:
        if reload:
            sys.modules.pop(module_name, None)
        elif module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            logger.error(f"Failed to load sandbox plugin {file_path.name}: spec not found")
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to load sandbox plugin {file_path.name}: {e}")
        sys.modules.pop(module_name, None)
    return None


def _find_subclass_instances(module, base_class, module_name: str) -> list:
    """Find and instantiate all subclasses of base_class in a module.

    Skips classes with ``auto_register = False`` — these are special tools
    that carry per-call construction state and are instantiated manually.
    """
    instances = []
    for cls in _find_subclasses(module, base_class, module_name):
        if getattr(cls, "auto_register", True) is False:
            continue
        try:
            instances.append(cls())
        except Exception as e:
            logger.error(f"Could not instantiate {cls.__name__}: {e}", exc_info=True)
    return instances


def _find_subclasses(module, base_class, module_name: str) -> list:
    """Find all concrete subclasses of base_class declared in a module."""
    found = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, base_class) and cls is not base_class and cls.__module__ == module_name:
            found.append(cls)
    return found


def _call_build_services(module, module_name: str, config: dict) -> dict:
    """Call build_services(config) on a module, return resulting dict."""
    build_fn = getattr(module, "build_services", None)
    if build_fn is None:
        return {}
    try:
        built = build_fn(config)
        return built if built else {}
    except Exception as e:
        logger.error(f"build_services() in {module_name} failed: {e}. Check config settings for this service with /config.", exc_info=True)
        return {}
