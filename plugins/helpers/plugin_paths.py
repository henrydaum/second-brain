"""Support code for plugin paths."""

from dataclasses import dataclass
from pathlib import Path

from paths import DATA_DIR, INSTALLED_PLUGINS, ROOT_DIR, SANDBOX_PLUGINS


@dataclass(frozen=True)
class PluginRoot:
    """A physical tree that can contain mirrored plugin family folders."""
    name: str
    path: Path
    module: str
    built_in: bool = False


@dataclass(frozen=True)
class PluginDir:
    """A concrete plugin family directory under one root."""
    root: PluginRoot
    plugin_type: str
    family: str
    prefix: str

    @property
    def path(self) -> Path:
        return self.root.path / self.family

    def module_name(self, stem: str) -> str:
        return f"{self.root.module}.{self.family}.{stem}"


@dataclass(frozen=True)
class PluginPathInfo:
    """Plugin path info."""
    plugin_type: str
    path: Path
    built_in: bool
    module_name: str
    root_name: str


PLUGIN_ROOTS = (
    PluginRoot("built_in", ROOT_DIR / "plugins", "plugins", True),
    PluginRoot("sandbox", SANDBOX_PLUGINS, "sandbox_plugins"),
    PluginRoot("installed", INSTALLED_PLUGINS, "installed_plugins"),
)

PLUGIN_FAMILIES = {
    "tool": ("tools", "tool_"),
    "task": ("tasks", "task_"),
    "service": ("services", "service_"),
    "command": ("commands", "command_"),
    "frontend": ("frontends", "frontend_"),
}

PLUGIN_CONFIG = {
    plugin_type: tuple(PluginDir(root, plugin_type, family, prefix) for root in PLUGIN_ROOTS)
    for plugin_type, (family, prefix) in PLUGIN_FAMILIES.items()
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
        if first in {"sandbox_plugins", "installed_plugins"}:
            resolved = (DATA_DIR / p).resolve()
        elif first == "plugins":
            resolved = (ROOT_DIR / p).resolve()
        else:
            root_path = (ROOT_DIR / p).resolve()
            data_path = (DATA_DIR / p).resolve()
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
    for plugin_type, dirs in PLUGIN_CONFIG.items():
        for plugin_dir in dirs:
            if path.parent != plugin_dir.path.resolve():
                continue
            if not name.startswith(plugin_dir.prefix):
                return None, f"{plugin_type.title()} files must start with '{plugin_dir.prefix}', got '{name}'."
            return PluginPathInfo(plugin_type, path, plugin_dir.root.built_in, plugin_dir.module_name(path.stem), plugin_dir.root.name), None
    inferred = _infer_type(name)
    if inferred:
        locations = ", ".join(str(d.path.resolve()) for d in PLUGIN_CONFIG[inferred])
        return None, f"{inferred.title()} plugin '{name}' must live in one of: {locations}. Got {path.parent}."
    return None, f"Plugin file '{name}' is not in a known plugin folder."


def iter_plugin_dirs():
    """Yield concrete plugin family directories."""
    for plugin_type, dirs in PLUGIN_CONFIG.items():
        for plugin_dir in dirs:
            yield plugin_type, plugin_dir.path


def plugin_dirs(plugin_type: str) -> tuple[PluginDir, ...]:
    """Return plugin directories for one family in precedence order."""
    return PLUGIN_CONFIG[plugin_type]


def _infer_type(file_name: str) -> str | None:
    """Internal helper to handle infer type."""
    for plugin_type, (_family, prefix) in PLUGIN_FAMILIES.items():
        if file_name.startswith(prefix):
            return plugin_type
    return None
