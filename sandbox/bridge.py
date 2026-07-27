"""The dual-mode loader — migrated and unmigrated plugins side by side.

A migration that made every unmigrated plugin inert would stop the app working
until the last one was done, which is the way rewrites die. They do not have to
coexist awkwardly: which contract a file uses is *readable from the file*, so
the loader can simply route.

**Detection is by import, read without importing.** A file that imports
``guest.bases`` is sandboxed; one that imports ``plugins.BaseTool`` is native.
No manifest, no naming convention, no registry of who has been migrated — the
same AST pass that already reads declarations answers this too.

**The adapter is the whole trick.** A sandboxed plugin cannot be registered
directly: its ``run`` wants an ``sdk``, and the registry will hand it a
context. So the bridge builds a *native* subclass whose ``run`` forwards into
the sandbox and translates the answer back. To the tool registry, the agent
loop, and the frontends, it is an ordinary plugin. Nothing downstream changes.

That is what makes migration reversible: one file, one commit, and
``git checkout`` puts it back.

Store plugins come along for free — discovery already scans the built-in,
sandbox, and installed roots through the same code path, so an installed
package migrates exactly like a kernel one.
"""

from __future__ import annotations

import ast
import logging
import types
from pathlib import Path

from .facade import Sandbox
from .policy import Chain
from .validator import FAMILIES, validate_file

logger = logging.getLogger("Sandbox")

# Modules whose import means "this file is written against the SDK".
SANDBOX_MODULES = {"guest", "guest.bases", "guest.box", "guest.sdk",
                   "sandbox.guest", "sandbox.guest.bases",
                   "sandbox.guest.box"}

# The native base class each family's adapter must subclass, so the kernel
# keeps seeing what it expects.
NATIVE_BASES = {
    "tool": ("plugins.BaseTool", "BaseTool"),
    "task": ("plugins.BaseTask", "BaseTask"),
    "command": ("plugins.BaseCommand", "BaseCommand"),
    "service": ("plugins.BaseService", "BaseService"),
    "frontend": ("plugins.BaseFrontend", "BaseFrontend"),
}

# Declarations are copied onto the adapter wholesale, minus the few that mean
# something to the *sandbox* rather than to the kernel. A denylist rather than
# an allowlist because the base classes will grow, and an allowlist that
# drifts silently drops a plugin's schema.
NOT_CARRIED = {"family", "box", "isolation", "lifetime", "timeout",
               "memory_mb", "requests", "exports"}

_SANDBOX: Sandbox | None = None


def configure(sandbox: Sandbox | None):
    """Give the bridge the sandbox migrated plugins should run in.

    Called once at bootstrap. Without it the bridge builds one on demand,
    which works but shares no boxes with the rest of the kernel.
    """
    global _SANDBOX
    _SANDBOX = sandbox


def get_sandbox() -> Sandbox:
    """The sandbox migrated plugins run in."""
    global _SANDBOX
    if _SANDBOX is None:
        _SANDBOX = Sandbox()
    return _SANDBOX


def is_sandboxed(path) -> bool:
    """Whether a plugin file is written against the SDK.

    Parses rather than imports, so asking costs nothing and cannot run
    anything — and parses rather than greps, because a docstring mentioning
    ``guest.bases`` would otherwise route a perfectly ordinary plugin into the
    bridge and break it at load.
    """
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return False
    return imports_sdk(tree)


def imports_sdk(tree) -> bool:
    """Whether a parsed module imports the SDK contract."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name in SANDBOX_MODULES for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if (node.module or "") in SANDBOX_MODULES:
                return True
    return False


def family_of(path) -> str:
    """Which plugin family a file belongs to, by filename."""
    stem = Path(path).stem
    prefix = stem.split("_")[0] if "_" in stem else ""
    return prefix if prefix in FAMILIES else ""


def _native_base(family: str):
    """Import the native base class for a family."""
    module_name, attr = NATIVE_BASES[family]
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def _result_to_native(family: str, result, native_module):
    """Translate a sandbox Result back into what the kernel expects."""
    if family == "command":
        # Commands return markdown, or None.
        return result.data if result.ok else f"Error: {result.error}"

    if family == "task":
        task_result = getattr(native_module, "TaskResult", None)
        if task_result is None:
            return result
        return task_result(success=result.ok, error=result.error,
                           data=result.data or [],
                           also_contains=list(result.also_contains),
                           discovered_paths=list(result.discovered_paths))

    tool_result = getattr(native_module, "ToolResult", None)
    if tool_result is None:
        return result
    return tool_result(success=result.ok, error=result.error,
                       data=result.data, llm_summary=result.llm_summary,
                       attachment_paths=list(result.attachment_paths))


def adapt(path, entry: str = "", family: str = "") -> types.ModuleType | None:
    """Build a synthetic module holding a native-looking adapter.

    Returns None when the file is not a migrated plugin, so callers can fall
    through to loading it the ordinary way.
    """
    path = Path(path)
    family = family or family_of(path)
    if not family or family not in NATIVE_BASES:
        return None
    if not is_sandboxed(path):
        return None

    report = validate_file(path)
    if not report.ok:
        logger.error("migrated plugin %s will not load:\n%s",
                     path.name, report.render())
        return None

    declarations = report.declarations
    entry = entry or _entry_from(report.source)
    if not entry:
        logger.error("migrated plugin %s declares no plugin class", path.name)
        return None

    base = _native_base(family)
    native_module = __import__(NATIVE_BASES[family][0],
                               fromlist=["ToolResult"])
    source_path = str(path)
    box_name = declarations.get("box") or path.stem

    def _forward(self, context, payload, method: str = "run"):
        """Run the migrated plugin and translate the answer back.

        The context is passed *per call* rather than held, because two calls
        can be in flight with different sessions and users behind them.
        """
        # Only the root is set here. ``Sandbox.start`` pushes the execution's
        # own name, and pushing it here too would put the plugin in the chain
        # twice — which the cycle detector correctly refuses.
        chain = Chain(root=_root_for(context))
        result = get_sandbox().run(
            source_path, entry, kwargs=payload, chain=chain, context=context,
            name=self.name or path.stem, method=method)
        if method != "run":
            return result.data if result.ok else None
        return _result_to_native(family, result, native_module)

    # The families disagree about argument order, and the adapter is the one
    # place that has to know. A generic ``run(self, context, *args)`` would
    # silently bind a task's ``paths`` to its ``context``.
    def run_tool(self, context, **kwargs):
        """Native tool contract."""
        return _forward(self, context, dict(kwargs))

    def run_task(self, paths, context):
        """Native task contract."""
        return _forward(self, context, {"paths": list(paths or [])})

    def run_command(self, args, context):
        """Native command contract."""
        return _forward(self, context, {"args": dict(args or {})})

    def form_command(self, args, context):
        """Native command form contract.

        A command whose form vanished would silently stop collecting its
        arguments, which is worse than not bridging commands at all — so the
        second entry point is forwarded too, and only when the migrated file
        actually defines one.

        Form steps cross the boundary as plain data, because a ``FormStep``
        is a live kernel object and sandboxed code cannot hold one. The
        registry does want the real thing — it reads ``step.name`` and calls
        ``step.coerce`` — so the dicts are rebuilt into ``FormStep``s here.
        """
        steps = _forward(self, context, {"args": dict(args or {})},
                         method="form") or []
        return [s for s in map(_form_step, steps) if s is not None]

    run = {"tool": run_tool, "task": run_task,
           "command": run_command}.get(family)
    if run is None:
        logger.info("%s: %s plugins are not bridged yet", path.name, family)
        return None

    attributes = {
        "__doc__": f"Sandboxed {family} loaded from {path.name}.",
        **({"form": form_command} if family == "command"
           and _defines(report.source, entry, "form") else {}),
        "_source_path": source_path,
        "_sandboxed": True,
        "_box": box_name,
        "_entry": entry,
        "run": run,
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value
    attributes.setdefault("name", path.stem.split("_", 1)[-1])

    adapter = type(f"Sandboxed{entry}", (base,), attributes)

    module = types.ModuleType(f"sandboxed_{path.stem}")
    module.__file__ = source_path
    setattr(module, adapter.__name__, adapter)
    return module


def _form_step(step):
    """Rebuild one form step from the plain data a sandboxed command returned.

    Unknown keys are dropped rather than raising: a command that names a field
    the kernel does not have should lose that field, not its whole form. The
    ``validator`` field is deliberately unreachable — it is a callable, so it
    could never have crossed the boundary in the first place.
    """
    if not isinstance(step, dict):
        return step
    from state_machine.conversation import FormStep

    allowed = set(FormStep.__dataclass_fields__) - {"validator"}
    fields = {k: v for k, v in step.items() if k in allowed}
    if not fields.get("name"):
        # Nothing can be collected under no name, and raising here would cost
        # the command its whole form over one bad step.
        logger.warning("dropping a form step with no 'name': %r", step)
        return None
    return FormStep(**fields)


def _root_for(context) -> str:
    """What caused this call, for the chain of provenance."""
    if context is None:
        return "kernel"
    if getattr(context, "user_initiated", False):
        return "user"
    session = getattr(context, "session_key", None)
    return str(session) if session else "agent"


def _entry_from(source: str) -> str:
    """The migrated plugin class's name, read out of the source."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    wanted = set(FAMILIES.values())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in wanted:
                    return node.name
    return ""


def _defines(source: str, class_name: str, method: str) -> bool:
    """Whether a class in the source defines a given method."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(isinstance(item, ast.FunctionDef)
                       and item.name == method for item in node.body)
    return False
