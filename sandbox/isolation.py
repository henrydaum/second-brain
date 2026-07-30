"""How isolated a file runs — decided by where it lives, never by what it says.

Why the tree decides and not the file — and why foreign-library detection is
computed from the AST rather than read off ``dependencies_pip`` — is argued in
CLAUDE.md under "Isolation is provenance, not declaration". The short form:
code being contained may never be the authority on its own containment, and a
file cannot assert which tree it is in.

Three trees, three answers:

- **``sandbox_plugins/``** — agent-authored, always a subprocess. This is the
  tree the security boundary exists for, and it is what buys the agent free
  rein inside it (see ``policy.py``): code it writes is contained before it
  runs, so writing it needs no dialog.
- **``plugins/``** — the kernel's own tree, always in-process. First-party
  code that ships with the system, on the trusted side by definition; putting
  it behind a pipe would buy nothing and cost every call.
- **``installed_plugins/``** — store packages, subprocess *if* they reach for
  something that cannot be mediated. A package that is pure computation over
  the SDK is as inspectable as kernel code; one that imports a foreign library
  has a component the validator cannot see inside, which is exactly the case
  the security contract says to put in a subprocess.

Anything else — a temporary file, a template, a path outside every tree — is
of unknown provenance and gets the subprocess. Failing closed is the only
defensible default for "I do not know what this is".
"""

from __future__ import annotations

from pathlib import Path

from .guest.box import IN_PROCESS, SUBPROCESS

# Tree names. Also the box namespace (see ``box_namespace``), so two files in
# different trees can never resolve into the same box.
KERNEL = "built_in"
SANDBOX = "sandbox"
INSTALLED = "installed"
UNKNOWN = "unknown"

# The fourth tree root, beside the five plugin families and ``helpers/``. A
# script is SDK code that nothing registers: no base class, no entry point, no
# discovery. See :func:`is_script`.
SCRIPTS_DIRNAME = "scripts"


def _roots() -> tuple:
    """(tree, root) pairs, most specific first.

    Read from ``paths`` rather than the plugin tree metadata: this is kernel
    path knowledge, and reaching into ``plugins.*`` from here would widen what
    the sandbox depends on for no gain. Order matters — the data trees live
    under DATA_DIR and the kernel tree under ROOT_DIR, but a checkout with
    DATA_DIR inside it must still resolve the more specific root first.
    """
    try:
        from paths import INSTALLED_PLUGINS, ROOT_DIR, SANDBOX_PLUGINS
    except Exception:
        return ()
    return (
        (SANDBOX, Path(SANDBOX_PLUGINS)),
        (INSTALLED, Path(INSTALLED_PLUGINS)),
        (KERNEL, Path(ROOT_DIR) / "plugins"),
    )


def _within(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def tree_of(source) -> str:
    """Which plugin tree a file belongs to, or ``UNKNOWN``.

    The whole of what the kernel trusts about a file, and the only input to
    :func:`required_isolation` that the file cannot influence.
    """
    if not source:
        return UNKNOWN
    path = Path(source)
    for tree, root in _roots():
        if _within(path, root):
            return tree
    return UNKNOWN


def is_script(source) -> bool:
    """Whether a file is a *script* — SDK code nothing registers.

    True for ``<tree>/scripts/<name>.py`` at any of the three tree roots. The
    directory is the whole declaration, exactly as ``helpers/`` is: a script
    has no base class, no entry point and no family prefix, so there is nothing
    else about the file that could say what it is.

    Top level only, matching ``helpers/``. A ``scripts/`` folder nested inside
    a family is not a tree root and is not treated as one.
    """
    if not source:
        return False
    path = Path(source)
    for _tree, root in _roots():
        try:
            relative = path.resolve().relative_to(Path(root).resolve())
        except (ValueError, OSError):
            continue
        return relative.parts[:1] == (SCRIPTS_DIRNAME,) and len(
            relative.parts) == 2
    return False


def resolve_script(raw) -> Path | None:
    """Find a script named the way an agent naturally names one, or None.

    ``resolve_plugin_path`` resolves a bare relative path against the project
    root and prefers it whenever it exists, which is right for a plugin — every
    plugin lives under a family directory that is named in the path. A script is
    named by its own filename, and the answer was consistently wrong in both
    directions: ``scripts/demo.py`` resolved into the *checkout*, which is not a
    tree root and so refused as "not in a scripts/ directory", and a bare
    ``demo.py`` resolved to a project-root file that was never there.

    So the two shapes an agent actually writes — the bare name, and one already
    relative to ``scripts/`` — are resolved against the script directories
    first, in tree precedence order. Anything else (an absolute path, or a
    relative path pointing somewhere else entirely) is left alone for the
    ordinary resolver and judged on its merits by :func:`is_script`.

    Answering None is not a refusal. It means "this is not a script reference I
    recognise", and the caller falls back.
    """
    if not raw:
        return None
    path = Path(str(raw).strip())
    if path.is_absolute():
        return None
    parts = path.parts
    if len(parts) == 1:
        relative = Path(SCRIPTS_DIRNAME) / parts[0]
    elif len(parts) == 2 and parts[0] == SCRIPTS_DIRNAME:
        relative = path
    else:
        return None
    for _tree, root in _roots():
        candidate = root / relative
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def required_isolation(source, report=None) -> str:
    """The isolation the kernel requires for this file. Not negotiable.

    ``report`` is a :class:`~sandbox.validator.Report`; it is consulted only
    for installed packages, and only to ask whether the validator found an
    import it cannot mediate. Absent report means unknown content, which is
    treated the same way as unknown provenance.
    """
    # Scripts are subprocessed wherever they live, which is the one place the
    # per-tree answer below is deliberately not consulted. An installed plugin
    # that is pure computation over the SDK earns in-process execution because
    # somebody approved it at ``plugin.install`` and it is a declared,
    # registered capability. A script is neither: nothing registers it, nothing
    # reviewed it, and the only thing that has ever been said about it is that
    # it parsed. Containment is the whole of what makes running one cheap, so
    # it is not something the file's address can buy its way out of.
    if is_script(source):
        return SUBPROCESS

    tree = tree_of(source)
    if tree == KERNEL:
        return IN_PROCESS
    if tree == SANDBOX:
        return SUBPROCESS
    if tree == INSTALLED:
        if report is None or getattr(report, "unmediated", None) is None:
            return SUBPROCESS
        return SUBPROCESS if report.unmediated else IN_PROCESS
    return SUBPROCESS


# ──────────────────────────────────────────────────────────────────────
# A note on boxes spanning trees.
#
# Files group into a shared box by declaring ``box = "name"``, and a box takes
# the *tightest* isolation any member asked for. The obvious worry is a file in
# ``sandbox_plugins`` naming the kernel's box to be loaded in-process beside
# it — the exact escape the tree rule exists to prevent.
#
# It cannot happen, and the reason is worth writing down rather than
# rediscovering. Isolation is computed per file, from that file's own path,
# before any grouping occurs. A sandbox file resolves to ``SUBPROCESS`` no
# matter which box it names, and tightest-wins can only ever *tighten* a box
# from there. The worst a mislabelled file achieves is dragging its box into a
# subprocess it did not need — a performance mistake, not an escalation.
#
# So no namespacing is applied to box names, which keeps them readable in
# provenance chains and ledger rows. If grouping ever becomes multi-file at the
# resolve site, the invariant to assert is that a box's members share a tree.
# ──────────────────────────────────────────────────────────────────────
