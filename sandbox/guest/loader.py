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
        # Merge rather than return early. Members of one box may declare
        # *different* ``dependencies_files``, and returning here dropped every
        # root but the first file's — so whichever sibling loaded second could
        # not import the helper it had declared, with no error until the
        # import failed somewhere unrelated.
        for root in (str(root), *(str(p) for p in extra_roots)):
            if root not in existing.__path__:
                existing.__path__.append(root)
        return package
    spec = importlib.machinery.ModuleSpec(package, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    # The plugin's own directory first, so a sibling always wins a name clash
    # with a declared dependency.
    module.__path__ = [str(root), *(str(p) for p in extra_roots)]
    sys.modules[package] = module
    return package


class StaleSource(ImportError):
    """The file on disk is not the file that was validated."""


def _verify(path: Path, digest: str) -> None:
    """Refuse to execute bytes nobody checked.

    Validation reads a path and execution opens it again, so without this the
    two can disagree — the file may have been edited, or swapped, in between.
    The window is small and the check is a hash, so it is bought cheaply.

    Scope worth being honest about: this covers the *entry* file, which is what
    the report describes. Siblings pulled in by ordinary imports off the box's
    ``__path__`` are not re-checked here.
    """
    if not digest:
        return
    from hashlib import sha256

    actual = sha256(
        path.read_text(encoding="utf-8", errors="replace").encode("utf-8")
    ).hexdigest()
    if actual != digest:
        raise StaleSource(
            f"{path.name} changed after it was validated; it was not loaded")


def load_member(module_path, box_name: str = "", root=None, extra_roots=(),
                digest: str = ""):
    """Import one file as a member of its box and return the module."""
    path = Path(module_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {module_path}")
    _verify(path, digest)
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


def install_parsers(paths, box_name: str = "", root=None) -> int:
    """Import the parser modules a box was provisioned with.

    A plugin declares ``parse_modalities`` and the host resolves that against
    the live parser registry; this is where the resolved files actually arrive.
    Each is imported as an ordinary box member, so its module-level
    ``register(...)`` calls fire and :func:`guest.parsing.adopt_registrations`
    collects them into the box's own routing table.

    The declaration is what makes this different from the plugin importing the
    file itself. A relative import is invisible to the isolation decision —
    the entry file's AST shows a sibling, not the foreign library behind it —
    whereas a declaration is read before anything runs, so the kernel knows
    foreign code is being provisioned and contains the box accordingly.

    One broken parser does not sink the others: it is logged past and the rest
    still load, which matches how the kernel's own discovery treats them.
    Returns the number of routes gained.
    """
    from . import parsing as guest_parsing

    gained = 0
    for module_path in paths or ():
        try:
            load_member(module_path, box_name=box_name, root=root)
        except BaseException as exc:      # noqa: BLE001 - one bad parser only
            print(f"[guest] parser {module_path} did not load: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            guest_parsing.drain_registrations()   # drop a partial declaration
            continue
        gained += guest_parsing.adopt_registrations()
    return gained


def load_entry(module_path, func_name: str = "", box_name: str = "",
               root=None, bound: bool = True, method: str = "run",
               extra_roots=(), digest: str = ""):
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
                         extra_roots=extra_roots, digest=digest)
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


def load_entries(module_path, names, box_name: str = "", root=None,
                 extra_roots=(), digest: str = "") -> dict:
    """Instantiate several plugin classes out of **one** module import.

    The point is the single import. Two services in a file share it because
    they share something expensive — a machine-learning library, an
    accelerator context, a connection pool — and loading the module twice
    would pay for it twice and leave the two halves unable to see each other's
    state. ``load_member`` caches by qualified name, so this is one execution
    of the file no matter how many classes come out of it.

    Keyed by each class's declared ``name`` rather than by its class name,
    because that is the handle every caller already has: the service name is
    what ``build_services`` registers, what a chain link is written with, and
    what arrives on the wire as a call's target.
    """
    module = load_member(module_path, box_name=box_name, root=root,
                         extra_roots=extra_roots, digest=digest)
    instances = {}
    for entry in names:
        target = getattr(module, entry, None)
        if target is None:
            raise AttributeError(f"{module_path} has no {entry!r}")
        instance = target() if isinstance(target, type) else target
        instances[getattr(instance, "name", "") or entry] = instance
    return instances


def unload_box(box_name: str):
    """Forget a box and every member loaded into it.

    Tearing down an ephemeral box has to clear the module cache, or the next
    run of the same box silently reuses stale code.
    """
    package = f"{PACKAGE_PREFIX}{box_name}"
    for key in [k for k in sys.modules
                if k == package or k.startswith(package + ".")]:
        sys.modules.pop(key, None)
