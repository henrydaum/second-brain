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
    "baked_in_dir":       ROOT_DIR / "Stage_0" / "services",
    "sandbox_dir":        SANDBOX_SERVICES,
    "glob":               "*.py",
    "baked_in_ns":        "Stage_0.services.{stem}",
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
    from Stage_3.BaseTool import BaseTool
    cfg = _TOOL_CONFIG
    t0 = time.time()
    count = 0
    baked_in_names = set()

    # Baked-in
    for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
        module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
        module = _load_baked_in(module_name, reload)
        if module is None:
            continue
        for instance in _find_subclass_instances(module, BaseTool, module_name):
            instance._mutable = False
            tool_registry.register(instance)
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
                count += 1

    logger.info(f"Discovered {count} tool(s) in {time.time() - t0:.2f}s")


def discover_tasks(root_dir: Path, orchestrator, config: dict, reload: bool = False):
    """Discover and register all tasks (baked-in + sandbox)."""
    from Stage_2.BaseTask import BaseTask
    cfg = _TASK_CONFIG
    t0 = time.time()
    count = 0
    baked_in_names = set()

    # Baked-in
    for py_file in sorted(cfg["baked_in_dir"].glob(cfg["glob"])):
        module_name = cfg["baked_in_ns"].format(stem=py_file.stem)
        module = _load_baked_in(module_name, reload)
        if module is None:
            continue
        for instance in _find_subclass_instances(module, BaseTask, module_name):
            instance._mutable = False
            orchestrator.register_task(instance)
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
                count += 1

    logger.info(f"Discovered {count} task(s) in {time.time() - t0:.2f}s")


def discover_services(root_dir: Path, config: dict) -> dict:
    """Discover all services (baked-in + sandbox). Returns {name: instance}."""
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
        for svc_name, svc in built.items():
            svc._mutable = False
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
            for svc_name, svc in built.items():
                if svc_name in baked_in_names:
                    logger.warning(f"Sandbox service '{svc_name}' collides with baked-in — skipped")
                    continue
                svc._mutable = True
                svc._source_path = str(py_file)
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
                  services: dict = None):
    """Unregister a plugin by name. Also cleans up sys.modules."""
    if plugin_type == "tool" and tool_registry:
        tool_registry.unregister(plugin_name)
    elif plugin_type == "task" and orchestrator:
        orchestrator.unregister_task(plugin_name)
    elif plugin_type == "service" and services:
        svc = services.pop(plugin_name, None)
        if svc and hasattr(svc, "unload") and getattr(svc, "loaded", False):
            try:
                svc.unload()
            except Exception as e:
                logger.error(f"Error unloading service '{plugin_name}': {e}")
        if svc:
            logger.info(f"Unregistered service: {plugin_name}")


def _load_single_tool(file_path: Path, tool_registry) -> tuple[str | None, str | None]:
    from Stage_3.BaseTool import BaseTool
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
    return instance.name, None


def _load_single_task(file_path: Path, orchestrator, config: dict) -> tuple[str | None, str | None]:
    from Stage_2.BaseTask import BaseTask
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
    return instance.name, None


def _load_single_service(file_path: Path, services: dict, config: dict) -> tuple[str | None, str | None]:
    cfg = _SERVICE_CONFIG
    module_name = cfg["sandbox_ns"].format(stem=file_path.stem)

    module = _load_sandbox(module_name, file_path, reload=True)
    if module is None:
        return None, f"Failed to import {file_path.name}"

    built = _call_build_services(module, module_name, config)
    if not built:
        return None, f"build_services() returned nothing in {file_path.name}"

    names = []
    for svc_name, svc in built.items():
        svc._mutable = True
        svc._source_path = str(file_path)
        services[svc_name] = svc
        names.append(svc_name)

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
    """Load a sandbox module via spec_from_file_location."""
    try:
        if reload and module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Failed to load sandbox plugin {file_path.name}: {e}")
        # Clean up failed module from sys.modules
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
        logger.error(f"build_services() in {module_name} failed: {e}", exc_info=True)
        return {}
