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

import logging
import types
from pathlib import Path

from .facade import Sandbox
from .policy import Chain
from .validator import FAMILIES, validate_file

logger = logging.getLogger("Sandbox")

# Imports that mean "this file is written against the SDK".
SANDBOX_MARKERS = ("guest.bases", "guest.box", "sandbox.guest.bases",
                   "from guest import", "import guest\n")

# The native base class each family's adapter must subclass, so the kernel
# keeps seeing what it expects.
NATIVE_BASES = {
    "tool": ("plugins.BaseTool", "BaseTool"),
    "task": ("plugins.BaseTask", "BaseTask"),
    "command": ("plugins.BaseCommand", "BaseCommand"),
    "service": ("plugins.BaseService", "BaseService"),
    "frontend": ("plugins.BaseFrontend", "BaseFrontend"),
}

# Declarations that mean something to the *kernel* and must be copied onto the
# adapter, or the registry will advertise a plugin with no schema and no name.
CARRIED = ("name", "description", "parameters", "requires_services",
           "dependencies_files", "dependencies_pip", "dependencies_tools",
           "max_calls", "background_safe", "auto_register", "category",
           "hide_from_help", "require_approval", "config_settings",
           "agent_prompt", "trigger", "trigger_channels", "modalities",
           "reads", "writes", "output_schema", "batch_size", "max_workers",
           "default_jobs", "require_all_inputs", "shared", "lifecycle",
           "model_name", "user_binding", "default_user_id")

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

    Reads the source rather than importing it, so asking the question costs
    nothing and cannot run anything.
    """
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in source for marker in SANDBOX_MARKERS)


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

    def _forward(self, context, payload):
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
            name=self.name or path.stem)
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

    run = {"tool": run_tool, "task": run_task,
           "command": run_command}.get(family)
    if run is None:
        logger.info("%s: %s plugins are not bridged yet", path.name, family)
        return None

    attributes = {
        "__doc__": f"Sandboxed {family} loaded from {path.name}.",
        "_source_path": source_path,
        "_sandboxed": True,
        "_box": box_name,
        "_entry": entry,
        "run": run,
    }
    for key in CARRIED:
        if key in declarations:
            attributes[key] = declarations[key]
    attributes.setdefault("name", path.stem.split("_", 1)[-1])

    adapter = type(f"Sandboxed{entry}", (base,), attributes)

    module = types.ModuleType(f"sandboxed_{path.stem}")
    module.__file__ = source_path
    setattr(module, adapter.__name__, adapter)
    return module


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
