"""How isolated a file runs — decided by where it lives, never by what it says.

Why the tree decides and not the file — and why foreign-library detection is
computed from the AST rather than read off ``dependencies_pip`` — is argued in
CLAUDE.md under "Isolation is provenance, not declaration". The short form:
code being contained may never be the authority on its own containment, and a
file cannot assert which tree it is in.

Three trees, three answers:

- **``workspace/``** — agent-authored, always a subprocess. This is the
  tree the security boundary exists for, and it is what buys the agent free
  rein inside it (see ``policy.py``): code it writes is contained before it
  runs, so writing it needs no dialog.
- **``bundled/``** — the app's own tree, always in-process. First-party
  code that ships with the system, on the trusted side by definition; putting
  it behind a pipe would buy nothing and cost every call.
- **``installed/``** — store packages, subprocess *if* they reach for
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
# different trees can never resolve into the same box. Taken from ``trees``
# rather than restated: these strings are compared against ``tree_of`` results
# all over the sandbox, and a fourth spelling of "which tree" is exactly what
# this module used to be.
KERNEL = "bundled"
SANDBOX = "workspace"
INSTALLED = "installed"
UNKNOWN = "unknown"

#: The root a script lives in. One name, read from the table.
SCRIPTS_DIRNAME = "scripts"


def _locate(source):
    """``trees.locate``, tolerant of a kernel that is not importable.

    Read from ``trees`` rather than the plugin substrate: this is kernel path
    knowledge, and reaching into ``plugins.*`` from here would widen what the
    sandbox depends on to answer a question about containment. An absent
    kernel (tests, a bare container) yields None, which every caller below
    treats as unknown provenance and therefore as a subprocess.
    """
    try:
        import trees
    except Exception:
        return None
    return trees.locate(source)


def tree_of(source) -> str:
    """Which plugin tree a file belongs to, or ``UNKNOWN``.

    The whole of what the kernel trusts about a file, and the only input to
    :func:`required_isolation` that the file cannot influence.
    """
    found = _locate(source)
    return found.tree.name if found is not None else UNKNOWN


def is_script(source) -> bool:
    """Whether a file is a *script* — SDK code nothing registers.

    True for ``<tree>/scripts/<name>.py`` in any tree. The directory is the
    whole declaration: a script has no base class, no entry point and no
    family prefix, so there is nothing else about the file that could say what
    it is.

    Top level only. A ``scripts/`` folder nested inside a family is not a tree
    root and is not treated as one.
    """
    found = _locate(source)
    if found is None or found.root is None:
        return False
    return found.root.name == SCRIPTS_DIRNAME and len(found.rel.parts) == 1


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
        name = parts[0]
    elif len(parts) == 2 and parts[0] == SCRIPTS_DIRNAME:
        name = parts[1]
    else:
        return None
    try:
        import trees
        # Reversed against discovery order on purpose. Discovery resolves a
        # *capability name* and lets the bundled tree win, so the app's own
        # tool is not shadowed by a draft. This resolves a *filename* an agent
        # typed, and the agent means the one it wrote — so the workspace is
        # searched first and the bundled tree last.
        script_dirs = tuple(reversed(trees.dirs_for(SCRIPTS_DIRNAME)))
    except Exception:
        return None
    for _tree, directory in script_dirs:
        candidate = directory / name
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
        if report.unmediated or _imports_foreign_code(source, report):
            return SUBPROCESS
        return IN_PROCESS
    return SUBPROCESS


def _imports_foreign_code(source, report) -> bool:
    """Whether this file pulls foreign libraries in behind its own AST.

    ``report.unmediated`` is built from the entry file's imports alone, and
    two things get past that:

    - a **declared helper** reached by relative import. ``from . import
      parse_pdf`` reads as an ordinary sibling, so a task whose own source is
      pure stdlib resolved IN_PROCESS while PyMuPDF loaded into the kernel's
      process. The file it names is right there in ``dependencies_files``, so
      the imports behind it are askable rather than unknowable.
    - a **declared modality**. ``parse_modalities = ["image"]`` asks the
      kernel to load parser files into this box; the parsers behind it are
      foreign by construction, and the plugin's own source says nothing about
      them at all.

    Both are declarations, which is exactly why they can be checked before
    anything runs — and why this direction is safe. It can only ever *tighten*
    a file's isolation, never loosen it, the same property the note below
    relies on for box grouping.
    """
    declarations = getattr(report, "declarations", None) or {}
    if declarations.get("parse_modalities"):
        return True

    declared = list(declarations.get("dependencies_files") or ())
    if not declared:
        return False

    # ``dependencies_files`` carries two jobs at once, and only one of them
    # puts code in this box. It tells the package manager what else to install
    # — which is why a tool declares the *service* it calls over the wire —
    # and it puts the file on the box's import path. The loader is explicit
    # that the second is only permission: "Declaring is what makes a file
    # importable; the plugin still writes the import."
    #
    # So a declared file nobody imports is never loaded, and its foreign
    # libraries never run here. Counting it anyway would subprocess most of
    # the store's tools for a packaging relationship, which is a real cost for
    # no containment. What is followed is the *import*, transitively, because
    # a helper may reach another declared helper of its own.
    from .validator import validate_file

    try:
        import trees
        roots = [tree.path for tree in trees.TREES if tree.local]
    except Exception:
        return True
    roots.insert(0, Path(source).parent.parent)   # <tree>/<family>/plugin.py

    def _resolve(relative):
        for root in roots:
            candidate = Path(root) / str(relative)
            if candidate.is_file():
                return candidate
        return None

    by_stem = {Path(str(rel)).stem: rel for rel in declared}
    pending = _relative_imports(report.source) & set(by_stem)
    seen = set()
    while pending:
        stem = pending.pop()
        if stem in seen:
            continue
        seen.add(stem)

        resolved = _resolve(by_stem[stem])
        if resolved is None:
            # Imported, declared, and not on disk. The import will fail at
            # load; until then, a file the kernel cannot find is one it cannot
            # vouch for, so the answer is the contained one.
            return True
        try:
            helper = validate_file(resolved)
        except Exception:
            return True
        if helper.unmediated:
            return True
        pending |= (_relative_imports(helper.source) & set(by_stem)) - seen

    return False


def _relative_imports(source_text: str) -> set:
    """Module names this source reaches for as box siblings.

    ``from . import x`` and ``from .x import y`` are the two spellings a box
    member uses, since a box is a flat synthetic package. Anything else is
    either absolute — and therefore already counted by ``unmediated`` — or not
    an import at all.
    """
    import ast

    names = set()
    try:
        tree = ast.parse(source_text or "")
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module:
            names.add(node.module.split(".")[0])     # from .x import y
        else:
            names.update(alias.name for alias in node.names)   # from . import x
    return names


# ──────────────────────────────────────────────────────────────────────
# A note on boxes spanning trees.
#
# Files group into a shared box by declaring ``box = "name"``, and a box takes
# the *tightest* isolation any member asked for. The obvious worry is a file in
# a ``workspace`` file naming the bundled tree's box to be loaded in-process beside
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
