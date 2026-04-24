"""
Plugin discovery — unified loader for tools, tasks, and services.

Handles both baked-in (read-only, in source tree) and sandbox (mutable,
in DATA_DIR) plugins. Used at startup for bulk discovery and by
build_plugin for single-file load/unload at runtime.

Public API:
    discover_all()          — startup convenience, discovers everything
    discover_tools()        — tools only
    discover_tasks()        — tasks only
    discover_services()     — services only, returns dict
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

from paths import ROOT_DIR, SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES

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

_TOOL_CONFIG = {
    "baked_in_dir":       ROOT_DIR / "Stage_3" / "tools",
    "sandbox_dir":        SANDBOX_TOOLS,
    "glob":               "tool_*.py",
    "baked_in_ns":        "Stage_3.tools.{stem}",
    "sandbox_ns":         "sandbox_tools_{stem}",
    "base_module":        "Stage_3.BaseTool",
    "base_class_name":    "BaseTool",
}

_TASK_CONFIG = {
    "baked_in_dir":       ROOT_DIR / "Stage_2" / "tasks",
    "sandbox_dir":        SANDBOX_TASKS,
    "glob":               "task_*.py",
    "baked_in_ns":        "Stage_2.tasks.{stem}",
    "sandbox_ns":         "sandbox_tasks_{stem}",
    "base_module":        "Stage_2.BaseTask",
    "base_class_name":    "BaseTask",
}

_SERVICE_CONFIG = {
    "baked_in_dir":       ROOT_DIR / "Stage_1" / "services",
    "sandbox_dir":        SANDBOX_SERVICES,
    "glob":               "*.py",
    "baked_in_ns":        "Stage_1.services.{stem}",
    "sandbox_ns":         "sandbox_services_{stem}",
}


# ── Bulk discovery (startup) ─────────────────────────────────────────

def discover_all(root_dir: Path, tool_registry, orchestrator, config: dict) -> dict:
    """Discover all plugins. Returns the services dict."""
    discover_tools(root_dir, tool_registry, config)
    discover_tasks(root_dir, orchestrator, config)
    return discover_services(root_dir, config)


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
            instance._mutable = False
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
                instance._mutable = True
                instance._source_path = str(py_file)
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
            instance._mutable = False
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
                instance._mutable = True
                instance._source_path = str(py_file)
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
            svc._mutable = False
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
                svc._mutable = True
                svc._source_path = str(py_file)
                _collect_config_settings(svc, service_names=built_names, plugin_type="service")
                services[svc_name] = svc

    logger.info(f"Discovered {len(services)} service(s) in {time.time() - t0:.2f}s")
    return services


# ── Single-plugin load/unload (used by build_plugin) ────────────────

def load_single_plugin(plugin_type: str, file_path: Path,
                       tool_registry=None, orchestrator=None,
                       services: dict = None, config: dict = None) -> tuple[str | None, str | None]:
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
    else:
        return None, f"Unknown plugin_type: {plugin_type}"


def unload_plugin(plugin_type: str, plugin_name: str,
                  tool_registry=None, orchestrator=None,
                  services: dict = None, source_path: str = None):
    """Unregister a plugin. For services, uses source_path to find all
    service names registered from that file."""
    if plugin_type == "tool" and tool_registry:
        tool_registry.unregister(plugin_name)
    elif plugin_type == "task" and orchestrator:
        orchestrator.unregister_task(plugin_name)
    elif plugin_type == "service" and services:
        _unload_services_by_source(services, source_path or plugin_name)


def _unload_services_by_source(services: dict, source_path: str):
    """Find all services registered from a source file, unload and remove them."""
    to_remove = [
        name for name, svc in services.items()
        if getattr(svc, "_source_path", None) == source_path
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


def _load_single_tool(file_path: Path, tool_registry) -> tuple[str | None, str | None]:
    from plugins.BaseTool import BaseTool
    cfg = _TOOL_CONFIG
    module_name = cfg["sandbox_ns"].format(stem=file_path.stem)

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    instances = _find_subclass_instances(module, BaseTool, module_name)
    if not instances:
        return None, f"No BaseTool subclass found in {file_path.name}"

    instance = instances[0]
    instance._mutable = True
    instance._source_path = str(file_path)
    tool_registry.register(instance)
    _collect_config_settings(instance, plugin_type="tool")
    return instance.name, None


def _load_single_task(file_path: Path, orchestrator, config: dict) -> tuple[str | None, str | None]:
    from plugins.BaseTask import BaseTask
    cfg = _TASK_CONFIG
    module_name = cfg["sandbox_ns"].format(stem=file_path.stem)

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    instances = _find_subclass_instances(module, BaseTask, module_name)
    if not instances:
        return None, f"No BaseTask subclass found in {file_path.name}"

    instance = instances[0]
    instance._mutable = True
    instance._source_path = str(file_path)
    orchestrator.register_task(instance)
    _collect_config_settings(instance, plugin_type="task")
    return instance.name, None


def _load_single_service(file_path: Path, services: dict, config: dict) -> tuple[str | None, str | None]:
    cfg = _SERVICE_CONFIG
    module_name = cfg["sandbox_ns"].format(stem=file_path.stem)

    # Unload any existing services from this file first (frees models/GPU)
    _unload_services_by_source(services, str(file_path))

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    built = _call_build_services(module, module_name, config)
    if not built:
        return None, f"build_services() returned nothing in {file_path.name}"

    names = list(built.keys())
    for svc_name, svc in built.items():
        svc._mutable = True
        svc._source_path = str(file_path)
        _collect_config_settings(svc, service_names=names, plugin_type="service")
        services[svc_name] = svc

    return ", ".join(names), None


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
    """Find and instantiate all subclasses of base_class in a module."""
    instances = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, base_class) and cls is not base_class and cls.__module__ == module_name:
            try:
                instances.append(cls())
            except Exception as e:
                logger.error(f"Could not instantiate {cls.__name__}: {e}", exc_info=True)
    return instances


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
