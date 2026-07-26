"""The parity harness — does the migrated plugin still answer the same?

Migrating a plugin means rewriting it against the SDK. The question that
matters afterwards is narrow: *given the same arguments, does it return the
same thing?* This runs both versions and says.

**Both versions without duplicate files.** A migrated file replaces its own
predecessor, so the two versions are the working tree and git. The harness
materializes the old one with ``git show <ref>:<path>`` — the same technique
the package manager uses against the store branch — and runs it in-process
against the same context the sandbox gets. There is nothing to name, nothing
to register twice, and nothing to clean up afterwards.

**One context, two paths.** The native plugin receives the context directly;
the sandboxed one receives an ``sdk`` whose handlers are backed by *that same
object*. Same database, same services, same config, so a difference in the
result is a difference in the plugin.

**Return value only.** Filesystem and database effects are deliberately not
compared. Store plugins are meant to be customised and will drift; what has
to hold is the answer. Kernel plugins — commands especially — are where a
difference is worth stopping for.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .facade import Sandbox
from .guest.loader import load_member, unload_box
from .guest.requests import Result

logger = logging.getLogger("Sandbox")

REPO = Path(__file__).resolve().parent.parent

# What each family is handed, and in what order. The native contract differs
# per family and the sandboxed one mirrors it with ``sdk`` in place of
# ``context``, so the harness needs to know both shapes.
FAMILY_CALLS = {
    "tool": "kwargs",       # native run(context, **kwargs)
    "command": "args",      # native run(args, context)
    "task": "paths",        # native run(paths, context)
}

# Fields worth comparing, and what they are called on each side.
COMPARED = (
    ("ok", "success"),
    ("data", "data"),
    ("error", "error"),
    ("llm_summary", "llm_summary"),
    ("attachment_paths", "attachment_paths"),
    ("also_contains", "also_contains"),
    ("discovered_paths", "discovered_paths"),
)


@dataclass
class Parity:
    """The verdict on one plugin's migration."""

    name: str
    native: dict = field(default_factory=dict)
    sandboxed: dict = field(default_factory=dict)
    differences: list = field(default_factory=list)
    error: str = ""

    @property
    def matched(self) -> bool:
        """Whether both versions answered the same."""
        return not self.differences and not self.error

    def render(self) -> str:
        """A report worth reading when it fails."""
        if self.error:
            return f"{self.name}: could not compare — {self.error}"
        if self.matched:
            return f"{self.name}: identical."
        lines = [f"{self.name}: {len(self.differences)} difference(s)"]
        for field_name, native, sandboxed in self.differences:
            lines.append(f"  {field_name}:")
            lines.append(f"    native:    {native!r}")
            lines.append(f"    sandboxed: {sandboxed!r}")
        return "\n".join(lines)


def previous_source(path, ref: str = "HEAD") -> str | None:
    """The file as it was before migration, or None if git has no such copy."""
    path = Path(path).resolve()
    try:
        relative = path.relative_to(REPO).as_posix()
    except ValueError:
        return None
    done = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{ref}:{relative}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", check=False)
    return done.stdout if done.returncode == 0 else None


def _load_native(source: str, stem: str, workspace: Path, box_name: str):
    """Write the previous version somewhere and import it.

    The box name is passed in rather than derived, because deriving it here
    and unloading a differently-derived one in the caller means the module
    cache is never cleared — and a stale previous version silently compares
    the wrong code.
    """
    path = workspace / f"{stem}.py"
    path.write_text(source, encoding="utf-8")
    return load_member(path, box_name=box_name, root=workspace)


def _plugin_class(module):
    """The one plugin class in a module, native or sandboxed."""
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        bases = {base.__name__ for base in value.__mro__[1:]}
        if bases & {"BaseTool", "BaseTask", "BaseCommand", "BaseService",
                    "BaseFrontend", "BasePlugin"}:
            return value
    return None


def _invoke_native(module, family: str, payload, context):
    """Call the pre-migration plugin the way the kernel used to."""
    cls = _plugin_class(module)
    if cls is None:
        raise LookupError("no plugin class in the previous version")
    instance = cls()
    if family == "command":
        return instance.run(payload or {}, context)
    if family == "task":
        return instance.run(payload or [], context)
    return instance.run(context, **(payload or {}))


def _normalize(outcome) -> dict:
    """Reduce either result type to one comparable shape.

    ``ToolResult`` and ``TaskResult`` carry fields the sandbox's ``Result``
    now mirrors, so the mapping is a rename rather than a translation. A
    command returns a bare string, which is treated as its data.
    """
    if outcome is None:
        return {"ok": True, "data": None, "error": ""}
    if isinstance(outcome, str):
        return {"ok": True, "data": outcome, "error": ""}

    shape = {}
    for ours, theirs in COMPARED:
        if hasattr(outcome, ours):
            value = getattr(outcome, ours)
        elif hasattr(outcome, theirs):
            value = getattr(outcome, theirs)
        else:
            continue
        shape[ours] = list(value) if isinstance(value, (list, tuple)) else value
    return shape


def compare(path, entry: str = "", *, payload=None, context=None,
            family: str = "tool", ref: str = "HEAD",
            sandbox: Sandbox | None = None, workspace=None) -> Parity:
    """Run a plugin's pre-migration and migrated versions, and diff them.

    ``payload`` is whatever the family takes: kwargs for a tool, an args dict
    for a command, a list of paths for a task.
    """
    path = Path(path)
    name = path.stem
    previous = previous_source(path, ref)
    if previous is None:
        return Parity(name, error=f"no {ref} copy of {name}.py to compare with")

    owned = sandbox is None
    box = sandbox or Sandbox(context=context)
    native_box = f"parity_{name}"

    # Never beside the plugin. A file called ``tool_x_previous.py`` sitting in
    # a plugin directory carries the ``tool_`` prefix, so discovery finds it,
    # registers it, and collides on the very name we just migrated.
    space = Path(workspace) if workspace else Path(
        tempfile.mkdtemp(prefix="sb-parity-"))
    disposable = workspace is None

    try:
        module = _load_native(previous, f"{name}_previous", space,
                              native_box)
        native = _normalize(_invoke_native(module, family, payload, context))
    except Exception as exc:
        return Parity(name, error=f"previous version failed: {exc}")
    finally:
        unload_box(native_box)
        if disposable:
            shutil.rmtree(space, ignore_errors=True)

    try:
        if entry:
            outcome = box.run(path, entry, kwargs=_as_kwargs(family, payload))
        else:
            outcome = box.run(path, kwargs=_as_kwargs(family, payload))
        sandboxed = _normalize(outcome)
    except Exception as exc:
        return Parity(name, native=native,
                      error=f"migrated version failed: {exc}")
    finally:
        unload_box(name)
        if owned:
            box.shutdown()

    return Parity(name, native=native, sandboxed=sandboxed,
                  differences=_diff(native, sandboxed))


def _as_kwargs(family: str, payload):
    """Shape the payload the way the migrated entry point expects it."""
    if family == "command":
        return {"args": payload or {}}
    if family == "task":
        return {"paths": payload or []}
    return dict(payload or {})


def _diff(native: dict, sandboxed: dict) -> list:
    """Fields present on either side that do not agree.

    A field only one side reports is not a difference: the pre-migration
    types do not all carry every field, and absence is not disagreement.
    """
    differences = []
    for key in sorted(set(native) | set(sandboxed)):
        if key not in native or key not in sandboxed:
            continue
        if native[key] != sandboxed[key]:
            differences.append((key, native[key], sandboxed[key]))
    return differences


def compare_many(cases, *, context=None, sandbox: Sandbox | None = None):
    """Run several comparisons against one sandbox.

    ``cases`` are dicts of :func:`compare` keyword arguments. Sharing the
    sandbox means sharing one context and one interpreter, which is what makes
    a whole-suite run cheap.
    """
    owned = sandbox is None
    box = sandbox or Sandbox(context=context)
    try:
        return [compare(sandbox=box, context=context, **case)
                for case in cases]
    finally:
        if owned:
            box.shutdown()
