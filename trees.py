"""Which trees exist, and what kinds of file a tree may contain.

Two facts decide where any piece of extension code lives, and until this module
existed neither was written down in one place.

**Who finds it.** A root whose files carry a *prefix* is scanned: something
globs ``f"{prefix}*.py"`` and builds an index. A root with no prefix is only
ever reached by something naming the file — imported, or run. That is the whole
distinction between ``tools/`` and ``scripts/``, and it is why a script needs no
``script_`` prefix: the directory is already the declaration, and taxing the
cheapest capability only pushes work back through ``proc.run``.

**Who put it here.** Three local trees — the app's own (``bundled``), the
store's (``installed``), the agent's (``workspace``) — plus the store branch
itself, which is the same shape reached over git rather than a filesystem.
Precedence runs bundled → installed → workspace, matching discovery order.

A root is declared here **only when the kernel itself routes it**. That test
admits ``parsers/`` and ``llm/`` (kernel registries live in ``parsing/`` and
``llm/``) and ``scripts/`` (``script.run``, ``isolation.is_script``,
``policy._classify_script``) beside the five plugin families. It excludes
``bundles/``, which exists because of store packages and which the package
layer keeps handling on its own. It also excludes ``workspace/memory/``, which
the kernel *does* name — ``agent.system_prompt`` inlines its ``MEMORY.md`` —
but only ever in the one tree the agent writes: a root is a shape every tree
holds, and a bundled or installed ``memory/`` would mean nothing. The store's
memory tool owns everything inside it. Adding a root here is a claim that core code needs standing
knowledge of it — the same question CLAUDE.md asks before widening the kernel
boundary.

**There is no top-level ``helpers/``.** A helper exists to help a plugin, so it
lives inside the family it helps (``<tree>/tools/helpers/x.py``). The root used
to exist because parsers and LLM backends had nowhere else to go, which made
"not a plugin" the definition of a folder that two kernel registries were
scanning.

This module lives at the repo root rather than under ``plugins/`` because
``sandbox/isolation.py`` and ``sandbox/policy.py`` both need it and both refuse
to import ``plugins.*`` — the sandbox may not depend on the plugin substrate to
answer a question about containment. ``paths.py`` is already the shared home for
exactly this kind of knowledge; this is its companion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from paths import DATA_DIR, ROOT_DIR

logger = logging.getLogger("Trees")

# ──────────────────────────────────────────────────────────────────────
# Trees
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Tree:
    """One materialization of the root layout.

    ``module`` is the ``sys.modules`` namespace plugins from this tree load
    under, and it is deliberately the same word as the folder: three names for
    one thing is how the old ``PluginRoot("sandbox", SANDBOX_PLUGINS,
    "sandbox_plugins")`` drifted.

    A remote tree carries ``ref`` instead of ``path`` and cannot be walked,
    imported from, or watched — only listed and read through
    ``store_backend``. It is here so the store's layout is checked against the
    same table as everywhere else rather than by borrowing the install
    target's rules.
    """
    name: str
    path: Path | None = None
    module: str | None = None
    ref: str | None = None
    builtin: bool = False

    @property
    def local(self) -> bool:
        return self.path is not None


BUNDLED = Tree("bundled", ROOT_DIR / "bundled", "bundled", builtin=True)
INSTALLED = Tree("installed", DATA_DIR / "installed", "installed")
WORKSPACE = Tree("workspace", DATA_DIR / "workspace", "workspace")

#: The store branch. Not in :data:`TREES` — discovery, the watcher and
#: isolation all iterate that tuple and none of them can reach a git ref.
STORE = Tree("store", ref="origin/store")

#: Local trees in discovery precedence order. **First match wins**: every
#: discoverer (``plugin_discovery``, ``parsing.discover``, ``llm.discover``)
#: keeps a seen-set and skips a later collision with a warning, so a bundled
#: capability shadows an installed one of the same name and both shadow a
#: workspace draft. Resolution *by filename* runs the other way — see
#: ``isolation.resolve_script``.
TREES: tuple[Tree, ...] = (BUNDLED, INSTALLED, WORKSPACE)

def tree(name: str) -> Tree | None:
    """Look a tree up by name, or None.

    A function rather than a dict built at import, because ``TREES`` is the one
    thing in this module that gets *replaced* — a test pointing the layout at a
    tmp_path swaps the tuple, and anything that had already snapshotted it into
    a mapping would go on answering with the real paths.
    """
    for candidate in (*TREES, STORE):
        if candidate.name == name:
            return candidate
    return None


# ──────────────────────────────────────────────────────────────────────
# Roots
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Root:
    """One folder that may appear at the top of any tree.

    ``family`` is set for the five kinds ``plugin_discovery`` registers, and is
    what ``plugin_info`` reports. ``parsers`` and ``llm`` are registered too,
    but by their own kernel registries, so they carry a prefix and no family.
    """
    name: str
    prefix: str = ""
    watched: bool = True
    family: str | None = None

    @property
    def registered(self) -> bool:
        """Whether a scanner globs this folder, rather than a caller naming a file."""
        return bool(self.prefix)

    @property
    def glob(self) -> str:
        return f"{self.prefix}*.py" if self.prefix else "*.py"


ROOTS: tuple[Root, ...] = (
    Root("tools", "tool_", family="tool"),
    Root("tasks", "task_", family="task"),
    Root("services", "service_", family="service"),
    Root("commands", "command_", family="command"),
    Root("frontends", "frontend_", family="frontend"),
    Root("parsers", "parse_"),
    Root("llm", "llm_"),
    # Nothing registers a script, so nothing needs to hear about one changing;
    # it is read from disk at the moment it runs.
    Root("scripts", watched=False),
)

#: Where a plugin keeps code that is not itself a plugin. Not a root — it is
#: the one nested folder the layout allows, directly under a family.
HELPERS_DIRNAME = "helpers"

roots_by_name: dict[str, Root] = {root.name: root for root in ROOTS}

#: The five ``plugin_discovery`` families, in discovery order.
FAMILIES: tuple[Root, ...] = tuple(r for r in ROOTS if r.family)

families_by_type: dict[str, Root] = {r.family: r for r in FAMILIES}


# ──────────────────────────────────────────────────────────────────────
# Lookups
# ──────────────────────────────────────────────────────────────────────


class Located(NamedTuple):
    """Where a file sits: its tree, the root it is under, and the rest.

    ``root`` is None when the path is inside a tree but not inside any declared
    root — a store-shipped ``bundles/`` folder, say. That is a real answer, not a
    failure: the tree is still known, which is all isolation needs.
    """
    tree: Tree
    root: Root | None
    rel: Path


#: ``_ordered_trees``' answer, and the ``TREES`` it was computed from.
_ordered: tuple[tuple[Tree, Path], ...] = ()
_ordered_for: tuple[Tree, ...] | None = None


def _ordered_trees() -> tuple[tuple[Tree, Path], ...]:
    """Local trees with their resolved paths, deepest first.

    Ordering is load-bearing rather than cosmetic: a checkout with DATA_DIR
    inside it puts one tree under another, and the more specific must answer
    first or every workspace file reads as a bundled one. Computed from the
    resolved paths so it holds however the two roots are configured.

    Resolving is why this is memoized. A tree root is fixed for the life of
    the process, but this was recomputing every one of them on every call —
    and ``locate`` then resolved them *again*, once per tree it tried. On
    Windows each ``resolve`` is a real filesystem call, so placing one file
    cost eighteen of them, and ``required_isolation`` asks twice per plugin
    load. That was 79% of the cost of validating a tool before running it.

    Keyed on the identity of ``TREES`` because that tuple is the one thing in
    this module that gets *replaced* — a test pointing the layout at a
    tmp_path swaps it wholesale and monkeypatch puts the original object back
    afterwards, so an identity check invalidates in both directions.
    """
    global _ordered, _ordered_for
    if _ordered_for is not TREES:
        found = []
        for candidate in TREES:
            if not candidate.local:
                continue
            try:
                found.append((candidate, candidate.path.resolve()))
            except (OSError, ValueError):
                found.append((candidate, candidate.path))
        _ordered = tuple(sorted(found, key=lambda pair: len(pair[1].parts),
                                reverse=True))
        _ordered_for = TREES
    return _ordered


def locate(path) -> Located | None:
    """Place a file in the layout, or None if it is outside every local tree."""
    if not path:
        return None
    try:
        target = Path(path).resolve()
    except (OSError, ValueError, TypeError):
        return None
    for tree, root_path in _ordered_trees():
        try:
            rel = target.relative_to(root_path)
        except ValueError:
            continue
        head = rel.parts[0] if rel.parts else ""
        root = roots_by_name.get(head)
        return Located(tree, root, Path(*rel.parts[1:]) if root else rel)
    return None


def tree_of(path) -> Tree | None:
    """Which local tree a file belongs to, or None."""
    found = locate(path)
    return found.tree if found else None


def attachment_cache() -> Path:
    """Where frontends drop files a person just handed the agent.

    **Inside the agent's own tree**, not beside it. A file arriving over a
    transport is the one kind of user data the agent is unambiguously meant to
    work on — read it, rename it, convert it, delete it once it has been dealt
    with — and every one of those verbs was a permission dialog while the cache
    sat at ``DATA_DIR/attachment_cache``. ``sandbox/policy.py`` had already
    conceded the point halfway by naming the folder a scratch root, so writes
    were free and everything else was not; putting it under ``workspace`` makes
    the whole grant one rule instead of a rule plus an exception.

    **Not a root**, and the module docstring is the test: nothing globs this
    folder and the kernel reaches it by name, so declaring it would put an
    empty ``attachments/`` in ``bundled/`` and ``installed/`` claiming those
    trees might hold one. It is the same kind of directory as
    ``workspace/temp`` — a place inside a tree rather than a shape the layout
    repeats.

    Created here rather than in :func:`materialize`, for the same reason it is
    not a root: the layout is one shape across trees and this belongs to
    exactly one. Failure to create it is left to the caller's own write to
    report — a read-only install should still boot.
    """
    workspace = tree("workspace")
    path = workspace.path / "attachments"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("could not create %s", path)
    return path


def dirs_for(root_name: str) -> tuple[tuple[Tree, Path], ...]:
    """Every local tree's ``<root_name>/`` directory, in precedence order.

    The one way to ask "where do parsers live?" — used by discovery, the
    watcher, ``parsing.discover`` and ``llm.discover`` alike, so none of them
    has to know how many trees there are.
    """
    return tuple((tree, tree.path / root_name) for tree in TREES if tree.local)


def iter_root_dirs(watched_only: bool = False):
    """Yield ``(tree, root, path)`` for every root of every local tree."""
    for tree in TREES:
        if not tree.local:
            continue
        for root in ROOTS:
            if watched_only and not root.watched:
                continue
            yield tree, root, tree.path / root.name


def materialize() -> tuple[Path, ...]:
    """Create every declared root in every local tree. Returns what was made.

    The layout is a *claim about where things go*, and a folder that only
    appears once something lands in it does not make that claim to anybody.
    Three consequences, all of which were live: ``/locations`` showed three
    trees with three different folder lists and no way to tell which
    difference was meaningful; an agent writing its first tool had to know to
    create ``workspace/tools/`` first; and ``scripts/`` — the safe alternative
    to ``proc.run`` — existed in no tree at all, because the only code that
    ever made a directory was the watcher, which skips unwatched roots.

    The built-in tree is included. It was excluded on the reasoning that the
    source tree is the developer's, but a root missing from ``bundled/`` reads
    as "the kernel cannot hold one of these", which is false and is the whole
    thing this fixes.

    Idempotent, and never destructive: an existing directory is left exactly
    as it is. Anything that cannot be created is reported rather than raised —
    a read-only install should still boot.
    """
    made = []
    for _tree, _root, directory in iter_root_dirs():
        if directory.is_dir():
            continue
        try:
            directory.mkdir(parents=True, exist_ok=True)
            made.append(directory)
        except OSError:
            logger.warning("could not create %s", directory)
    return tuple(made)


def is_root_dir(path) -> bool:
    """Whether this directory *is* a declared root of a local tree.

    Asked by anything that prunes empty directories: a root is empty on
    purpose most of the time, and deleting it un-does ``materialize`` on the
    next uninstall.
    """
    candidate = Path(path)
    return any(candidate == directory
               for _tree, _root, directory in iter_root_dirs())


def module_name(tree: Tree, rel: Path | str) -> str:
    """The dotted module name a file in ``tree`` loads under."""
    parts = Path(rel).with_suffix("").parts
    return ".".join((tree.module, *parts))
