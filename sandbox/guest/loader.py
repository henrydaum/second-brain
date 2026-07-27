"""Loading a box's members so they can import each other.

A box is one execution context, and files inside it are expected to import
their siblings — a plugin and its helpers are written as a unit. That only
works if they are loaded as a *package*: a bare
``spec_from_file_location`` gives a module with no parent, and every
``from .helpers import x`` fails.

So the box becomes a synthetic package whose ``__path__`` is the box root, and
members are imported as ``box_<name>.<stem>``. Relative imports then resolve
exactly as they do in the repo today, which is what lets a file move between
the built-in, sandbox, and installed trees unchanged.

The rule this enforces is the same one that defines a box: a relative import
reaches a sibling, and there is no relative path out of the package — so a
file in another box is not reachable by import at all. Getting there costs a
Request.

Pure importlib. Used by the child, and by the host when it runs a box
in-process, so both load code the same way.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

from .bases import entry_for

PACKAGE_PREFIX = "box_"


def install_box(root, box_name: str, extra_roots=()):
    """Register the synthetic package a box's members are imported into.

    Idempotent: loading a second member of the same box reuses the package, so
    siblings share one namespace and one module cache — which is what "same
    box" means at the language level.

    ``extra_roots`` are the directories holding files the plugin *declared* in
    ``dependencies_files``. They join ``__path__``, so a declared helper is a
    sibling by import even though it lives in another folder — a tool can
    reach ``helpers/parse_image.py`` as ``.parse_image``. Declaring is what
    makes a file importable; the plugin still writes the import, exactly as
    ``dependencies_pip`` installs a library and the plugin still imports it.

    A box therefore stays *flat*: one namespace, no package structure to
    reason about, and no relative path that climbs out of it.
    """
    package = f"{PACKAGE_PREFIX}{box_name}"
    existing = sys.modules.get(package)
    if existing is not None:
        return package
    spec = importlib.machinery.ModuleSpec(package, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    # The plugin's own directory first, so a sibling always wins a name clash
    # with a declared dependency.
    module.__path__ = [str(root), *(str(p) for p in extra_roots)]
    sys.modules[package] = module
    return package


def load_member(module_path, box_name: str = "", root=None, extra_roots=()):
    """Import one file as a member of its box and return the module."""
    path = Path(module_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {module_path}")
    root = Path(root) if root else path.parent
    package = install_box(root, box_name or path.stem, extra_roots)

    qualified = f"{package}.{path.stem}"
    cached = sys.modules.get(qualified)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(qualified, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that a sibling importing back into this
    # module during load finds it, rather than re-executing it.
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified, None)
        raise
    return module


def load_entry(module_path, func_name: str = "", box_name: str = "",
               root=None, bound: bool = True, method: str = "run",
               extra_roots=()):
    """Import a box member and resolve what the runner should hold.

    ``bound`` decides *what* comes back, because the two lifetimes need
    different things:

    - ``True`` (ephemeral) — a callable to invoke once. A plugin class is
      instantiated and its ``run`` returned.
    - ``False`` (persistent) — the object the kernel will call *into*
      repeatedly. A plugin class becomes an instance; a bare script stays a
      module, so its functions are the methods and its globals are the state.

    That second case is the whole of what makes a persistent script work: an
    agent's scratchpad server is a module whose globals outlive each call.
    """
    module = load_member(module_path, box_name=box_name, root=root,
                         extra_roots=extra_roots)
    if not func_name:
        # No entry named: the module itself is the object. Only meaningful
        # for a persistent box, where calls resolve to module-level functions.
        return module

    target = getattr(module, func_name, None)
    if target is None:
        raise AttributeError(f"{module_path} has no {func_name!r}")

    if not bound:
        return target() if isinstance(target, type) else target

    fn = entry_for(target, method)
    if not callable(fn):
        raise AttributeError(f"{module_path}.{func_name} is not callable")
    return fn


def unload_box(box_name: str):
    """Forget a box and every member loaded into it.

    Tearing down an ephemeral box has to clear the module cache, or the next
    run of the same box silently reuses stale code.
    """
    package = f"{PACKAGE_PREFIX}{box_name}"
    for key in [k for k in sys.modules
                if k == package or k.startswith(package + ".")]:
        sys.modules.pop(key, None)
