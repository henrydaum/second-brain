"""Where the five plugin families live, and what a file sitting in one is.

Every fact here is derived from :mod:`trees`, which is the single declaration
of the layout. This module is the *discovery-facing* view of it: the five
families ``plugin_discovery`` registers, expanded across the trees into the
concrete directories it globs. Roots the kernel routes some other way —
``parsers/``, ``llm/``, ``scripts/`` — are not families and are reached
through ``trees.dirs_for`` by whoever owns them.
"""

from dataclasses import dataclass
from pathlib import Path

import trees
from paths import DATA_DIR, ROOT_DIR
from trees import Tree

#: A tree, under the name discovery has always called it. One object, one
#: spelling — the old ``PluginRoot`` carried a name, a path and a module string
#: that were three renamings of the same word.
PluginRoot = Tree


@dataclass(frozen=True)
class PluginDir:
    """A concrete plugin family directory under one tree."""
    root: Tree
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
    builtin: bool
    module_name: str
    root_name: str


PLUGIN_ROOTS = trees.TREES

#: ``plugin_type -> (folder, prefix)``, the shape discovery and the package
#: manager both read.
PLUGIN_FAMILIES = {root.family: (root.name, root.prefix) for root in trees.FAMILIES}

PLUGIN_CONFIG = {
    root.family: tuple(PluginDir(tree, root.family, root.name, root.prefix)
                       for tree in PLUGIN_ROOTS)
    for root in trees.FAMILIES
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
        tree = trees.tree(first)
        if tree is not None and tree.local:
            # A path already naming its tree resolves against that tree's
            # parent, so "workspace/tools/tool_x.py" means the one place it can.
            resolved = (tree.path.parent / p).resolve()
        else:
            root_path = (ROOT_DIR / p).resolve()
            data_path = (DATA_DIR / p).resolve()
            resolved = root_path if root_path.exists() or not data_path.exists() else data_path
    if not any(resolved == root or root in resolved.parents for root in ALLOWED_ROOTS):
        return None, f"Path is outside allowed roots: {resolved}"
    return resolved, None


def plugin_info(path: Path) -> tuple[PluginPathInfo | None, str | None]:
    """Identify a file as one of the five families, or say why it is not.

    The error half matters as much as the answer: this is what a person sees
    after saving a file in the wrong place, so it names where the file should
    have gone rather than reporting only that it is unrecognised.
    """
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
            return PluginPathInfo(plugin_type, path, plugin_dir.root.builtin,
                                  plugin_dir.module_name(path.stem),
                                  plugin_dir.root.name), None
    inferred = _infer_type(name)
    if inferred:
        locations = ", ".join(str(d.path.resolve()) for d in PLUGIN_CONFIG[inferred])
        return None, f"{inferred.title()} plugin '{name}' must live in one of: {locations}. Got {path.parent}."
    found = trees.locate(path)
    if found is not None and found.root is not None:
        return None, (f"'{name}' is in {found.tree.name}/{found.root.name}/, "
                      f"which is not one of the five plugin families.")
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
