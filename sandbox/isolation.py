"""How isolated a file runs — decided by where it lives, never by what it says.

``isolation = "subprocess"`` used to be a module-level declaration, read off
the file by AST alongside ``exports`` and ``requests``. That put the choice of
containment in the hands of the code being contained, which is the one thing a
sandbox may never delegate: an agent authoring a plugin could author its own
escape from the process boundary by leaving a line out.

So isolation joins provenance rather than declaration. A file's tree is not
something it can assert — writing into ``sandbox_plugins/`` is what *makes*
something an agent-authored plugin — so the kernel reads the answer off the
path and the file gets no vote.

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

**Foreign-library detection is computed, not declared**, for the same reason
the tree is. ``dependencies_pip`` is a declaration and would reintroduce the
bug one level down; the validator's import walk already answers the question
from the AST, and ``report.unmediated`` is that answer.
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


def required_isolation(source, report=None) -> str:
    """The isolation the kernel requires for this file. Not negotiable.

    ``report`` is a :class:`~sandbox.validator.Report`; it is consulted only
    for installed packages, and only to ask whether the validator found an
    import it cannot mediate. Absent report means unknown content, which is
    treated the same way as unknown provenance.
    """
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
