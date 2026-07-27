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
               "memory_mb", "requests", "exports", "hooks"}

_SANDBOX: Sandbox | None = None


def configure(sandbox: Sandbox | None):
    """Give the bridge the sandbox migrated plugins should run in.

    Called once at bootstrap. Without it the bridge builds one on demand,
    which works but shares no boxes with the rest of the kernel.

    Also hands over the plugin tree roots, so a plugin's declared
    ``dependencies_files`` resolve across trees — an installed tool can
    declare a helper that only ships with the kernel. The bridge does this
    because it is the one part of the sandbox that knows about plugin layout.
    """
    global _SANDBOX
    _SANDBOX = sandbox
    if sandbox is not None:
        try:
            from plugins.helpers.plugin_paths import PLUGIN_ROOTS
            sandbox.plugin_roots = [root.path for root in PLUGIN_ROOTS]
        except Exception:
            logger.debug("could not resolve plugin roots for dependencies")


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

    # A service is not a call, it is a residency: it opens a box and stays.
    # Different enough that it gets its own builder rather than a branch in
    # the per-call machinery below.
    if family == "service":
        return _adapt_service(path, entry, base, declarations, box_name)

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


def _make_response(answer: dict):
    """Build an ``LLMResponse`` for an escort that answered without dialing.

    Lives here rather than in ``sandbox.hooks`` because that type is a plugin
    type, and the bridge is the one part of the sandbox sanctioned to import
    across that line — the same reason it holds the native base classes.
    """
    from plugins.services.service_llm import LLMResponse

    return LLMResponse(content=str(answer.get("content") or ""),
                       tool_calls=list(answer.get("tool_calls") or []),
                       error=(answer.get("error") or None))


def _sync_hooks(service) -> None:
    """Stand this service's declared hooks at their doorways, exactly once.

    Called from both ``bind_runtime`` and ``_load`` because the two arrive in
    different orders at boot and on reload, and neither alone is enough: a
    hook needs a runtime to register with and a box to call into.
    """
    declared = getattr(service, "_hooks", None) or {}
    runtime = getattr(service, "_runtime", None)
    registry = getattr(runtime, "hooks", None)
    if not declared or registry is None or service._shims is not None:
        return

    from .hooks import build_shim

    shims = []
    for moment, method in declared.items():
        try:
            shim = build_shim(service, moment, method, _make_response)
        except ValueError as exc:
            logger.error("service %s: %s", service.name, exc)
            continue
        registry.add(moment, shim)
        shims.append(shim)
    service._shims = shims
    if shims:
        logger.info("service %s stands at %s", service.name,
                    ", ".join(sorted(declared)))


def _unhook(service) -> None:
    """Walk this service away from every doorway.

    A hook outliving its plugin is a leak with no symptom — it keeps being
    consulted and keeps abstaining — so unload has to be thorough.
    """
    registry = getattr(getattr(service, "_runtime", None), "hooks", None)
    for shim in getattr(service, "_shims", None) or []:
        if registry is not None:
            try:
                registry.remove(shim)
            except Exception:
                logger.exception("could not remove a hook for %s", service.name)
    service._shims = None


def _listen(service) -> None:
    """Subscribe this service to every channel it declared.

    Called only from ``_load``, unlike ``_sync_hooks``: a subscription needs a
    box to deliver into but no runtime, so there is only one moment it can
    happen and no ordering problem to be idempotent about.
    """
    channels = getattr(service, "_channels", None) or []
    if not channels or service._listeners is not None:
        return

    from .events import subscribe_all

    def deliver(channel: str, payload):
        """Carry one event into the box, if the box is still there."""
        box = getattr(service, "_sandbox_box", None)
        if box is None or not box.alive:
            return
        result = box.call("__event__", channel=channel, payload=payload)
        if not result.ok:
            # A subscriber that raises is the publisher's problem only if we
            # make it one, and the bus contract says we must not.
            logger.warning("%s failed handling %s: %s",
                           service.name, channel, result.error)

    service._listeners = subscribe_all(service, channels, deliver)


def _deafen(service) -> None:
    """Drop every subscription. Symmetrical with ``_unhook``."""
    from .events import unsubscribe_all

    unsubscribe_all(getattr(service, "_listeners", None))
    service._listeners = None


class ServiceCallFailed(RuntimeError):
    """An exported service method failed inside its box.

    Raised rather than returned because native callers reach a service by
    attribute access — ``services.get("embedder").embed(texts)`` — and expect
    a value or an exception, not a Result they have to unwrap.
    """


def _adapt_service(path, entry: str, base, declarations: dict, box_name: str):
    """Build a native-looking service backed by a resident box.

    The other families are *calls*: run once, translate the answer, tear down.
    A service is a *residency*, so the adapter maps the native lifecycle onto
    the box lifecycle instead:

        _load()   ->  open the box (start() runs inside it)
        unload()  ->  close the box (stop() runs inside it)

    and every method the plugin lists in ``exports`` becomes a real method on
    the adapter, because native callers reach services by attribute access
    rather than through ``service.call``.
    """
    source_path = str(path)
    name = declarations.get("name") or path.stem.split("_", 1)[-1]
    exports = list(declarations.get("exports") or [])

    if not exports:
        # Not fatal — a service may exist only for its side effects — but it
        # is nearly always a forgotten declaration, and the symptom (every
        # call failing as "not exported") points nowhere near the cause.
        logger.warning("sandboxed service %s declares no exports; nothing "
                       "will be able to call it", path.name)

    def _load(self) -> bool:
        """Open the resident box. Its start() runs inside."""
        try:
            self._sandbox_box = get_sandbox().open(
                source_path, entry, name=box_name)
        except Exception as exc:
            logger.error("service %s did not start: %s", name, exc)
            return False
        self.loaded = True
        # Binding and loading happen in either order depending on whether this
        # is boot or a live reload, so both ends call the same idempotent sync.
        _sync_hooks(self)
        _listen(self)
        return True

    def unload(self):
        """Close the box and step away from every doorway and channel."""
        self.loaded = False
        _unhook(self)
        _deafen(self)
        self._sandbox_box = None
        try:
            get_sandbox().close(box_name)
        except Exception:
            logger.exception("failed to close box %s", box_name)

    def bind_runtime(self, *, runtime=None, **_):
        """Receive the runtime. Idempotent, and may arrive before or after load."""
        if runtime is not None:
            self._runtime = runtime
        _sync_hooks(self)

    def _export(method: str):
        """One forwarding method, so callers see an ordinary service."""
        def call(self, **kwargs):
            """Invoke an exported method inside the box."""
            box = getattr(self, "_sandbox_box", None)
            if box is None or not box.alive:
                raise ServiceCallFailed(
                    f"service {name!r} is not loaded")
            result = box.call(method, **kwargs)
            if not result.ok:
                raise ServiceCallFailed(f"{name}.{method}: {result.error}")
            return result.data
        call.__name__ = method
        call.__qualname__ = f"{entry}.{method}"
        call.__doc__ = f"Call {name}.{method} inside its sandbox box."
        return call

    attributes = {
        "__doc__": f"Sandboxed service loaded from {path.name}.",
        "_source_path": source_path,
        "_sandboxed": True,
        "_sandbox_box": None,
        "_runtime": None,
        "_shims": None,
        "_listeners": None,
        "_hooks": dict(declarations.get("hooks") or {}),
        "_channels": list(declarations.get("subscribed_channels") or []),
        "_box": box_name,
        "_entry": entry,
        # Native services are named by model_name; the guest calls it name.
        "model_name": name,
        "name": name,
        # The box owns the start deadline (boxes.DEFAULT_START_TIMEOUT). Left
        # at its default, BaseService.load() would wrap _load in a *second*
        # timer racing the first, and whichever fired first would report a
        # failure the other could not see.
        "load_timeout": 0,
        "_load": _load,
        "unload": unload,
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value
    # exports rides along on purpose: handlers._service_call reads it to
    # refuse anything unexported, so the declaration keeps working when the
    # caller is other sandboxed code rather than the kernel.
    attributes["exports"] = exports
    attributes["bind_runtime"] = bind_runtime
    for method in exports:
        attributes[method] = _export(method)

    adapter = type(f"Sandboxed{entry}", (base,), attributes)

    module = types.ModuleType(f"sandboxed_{path.stem}")
    module.__file__ = source_path
    setattr(module, adapter.__name__, adapter)

    def build_services(config: dict) -> dict:
        """Services are discovered by calling this, not by scanning classes."""
        return {name: adapter()}

    module.build_services = build_services
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
