"""Support code for plugin paths."""

from dataclasses import dataclass
from pathlib import Path

from paths import ROOT_DIR, DATA_DIR, SANDBOX_TOOLS, SANDBOX_TASKS, SANDBOX_SERVICES, SANDBOX_COMMANDS, SANDBOX_FRONTENDS


@dataclass(frozen=True)
class PluginPathInfo:
    """Plugin path info."""
    plugin_type: str
    path: Path
    built_in: bool
    module_name: str


PLUGIN_CONFIG = {
    "tool": (ROOT_DIR / "plugins" / "tools", SANDBOX_TOOLS, "tool_", ("plugins.tools.{stem}", "sandbox_tools_{stem}")),
    "task": (ROOT_DIR / "plugins" / "tasks", SANDBOX_TASKS, "task_", ("plugins.tasks.{stem}", "sandbox_tasks_{stem}")),
    "service": (ROOT_DIR / "plugins" / "services", SANDBOX_SERVICES, "service_", ("plugins.services.{stem}", "sandbox_services_{stem}")),
    "command": (ROOT_DIR / "plugins" / "commands", SANDBOX_COMMANDS, "command_", ("plugins.commands.{stem}", "sandbox_commands_{stem}")),
    "frontend": (ROOT_DIR / "plugins" / "frontends", SANDBOX_FRONTENDS, "frontend_", ("plugins.frontends.{stem}", "sandbox_frontends_{stem}")),
}
ALLOWED_ROOTS = tuple(p.resolve() for p in (ROOT_DIR, DATA_DIR))


def resolve_plugin_path(raw: str) -> tuple[Path | None, str | None]:
    """Resolve plugin path."""
    if not raw:
        return None, "plugin_path is required."
    p = Path(raw)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        first = p.parts[0] if p.parts else ""
        root_path = (ROOT_DIR / p).resolve()
        data_path = (DATA_DIR / p).resolve()
        if first.startswith("sandbox_"):
            resolved = data_path
        elif first == "plugins":
            resolved = root_path
        else:
            resolved = root_path if root_path.exists() or not data_path.exists() else data_path
    if not any(resolved == root or root in resolved.parents for root in ALLOWED_ROOTS):
        return None, f"Path is outside allowed roots: {resolved}"
    return resolved, None


def plugin_info(path: Path) -> tuple[PluginPathInfo | None, str | None]:
    """Handle plugin info."""
    path = path.resolve()
    name = path.name
    if path.suffix != ".py":
        return None, f"File name must end with .py, got '{name}'."
    for plugin_type, (built_dir, sandbox_dir, prefix, namespaces) in PLUGIN_CONFIG.items():
        in_built = path.parent == built_dir.resolve()
        in_sandbox = path.parent == sandbox_dir.resolve()
        if not (in_built or in_sandbox):
            continue
        if prefix and not name.startswith(prefix):
            return None, f"{plugin_type.title()} files must start with '{prefix}', got '{name}'."
        module_template = namespaces[0] if in_built else namespaces[1]
        return PluginPathInfo(plugin_type, path, in_built, module_template.format(stem=path.stem)), None
    inferred = _infer_type(name)
    if inferred:
        built_dir, sandbox_dir, *_ = PLUGIN_CONFIG[inferred]
        return None, f"{inferred.title()} plugin '{name}' must live in {built_dir.resolve()} or {sandbox_dir.resolve()}, got {path.parent}."
    return None, f"Plugin file '{name}' is not in a known plugin folder."


def iter_plugin_dirs():
    """Handle iter plugin dirs."""
    for plugin_type, (built_dir, sandbox_dir, *_rest) in PLUGIN_CONFIG.items():
        yield plugin_type, built_dir
        yield plugin_type, sandbox_dir


def _infer_type(file_name: str) -> str | None:
    """Internal helper to handle infer type."""
    for plugin_type, (_, _, prefix, _) in PLUGIN_CONFIG.items():
        if prefix and file_name.startswith(prefix):
            return plugin_type
    return None
