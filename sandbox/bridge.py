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
import threading
import types
from pathlib import Path

from .approval import describe_grant
from .facade import Sandbox
from .policy import Chain
from .validator import FAMILIES, validate_file

logger = logging.getLogger("Sandbox")

# Modules whose import means "this file is written against the SDK".
SANDBOX_MODULES = {"guest", "guest.bases", "guest.box", "guest.sdk",
                   "guest.forms",
                   "sandbox.guest", "sandbox.guest.bases",
                   "sandbox.guest.forms",
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

    # What one approval is allowed to buy. Read here, once, from the same
    # declaration the validator checks and the approval dialog names — so the
    # grant a user answers and the grant the policy honours cannot disagree.
    granted = frozenset(declarations.get("requests") or ())

    # A service is not a call, it is a residency: it opens a box and stays.
    # Different enough that it gets its own builder rather than a branch in
    # the per-call machinery below.
    if family == "service":
        return _adapt_service(path, entry, base, declarations, box_name)

    # A frontend is a residency too, but one the kernel drives rather than
    # calls: it owns a loop and nine render doorways. Its own builder for the
    # same reason a service has one.
    if family == "frontend":
        return _adapt_frontend(path, entry, base, declarations, box_name)

    def _forward(self, context, payload, method: str = "run"):
        """Run the migrated plugin and translate the answer back.

        The context is passed *per call* rather than held, because two calls
        can be in flight with different sessions and users behind them.
        """
        # Only the root is set here. ``Sandbox.start`` pushes the execution's
        # own name, and pushing it here too would put the plugin in the chain
        # twice — which the cycle detector correctly refuses.
        chain = Chain(
            root=_root_for(context),
            approved=(
                granted
                if getattr(context, "approved_by_state_machine", False)
                else None
            ),
        )
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
        # Unreachable while there are five families: services and frontends
        # returned above and the other three are here. It stands as the guard
        # for a sixth, which would otherwise load as a plugin that does nothing.
        logger.error("%s: %s plugins have no adapter", path.name, family)
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
        # The sentence the approval dialog asks, rendered from the same
        # declaration the grant is built from so the question a user answers
        # and the authority they hand over cannot drift apart.
        "approval_prompt": describe_grant(
            declarations.get("name") or path.stem.split("_", 1)[-1], granted),
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value
    attributes.setdefault("name", path.stem.split("_", 1)[-1])

    adapter, module = _build(entry, base, attributes, path, source_path)
    return module


def _build(entry: str, base, attributes: dict, path, source_path: str):
    """Make the adapter class and the synthetic module that carries it.

    One place because of ``__module__``. Discovery only accepts classes that
    belong to the module it just loaded, and a class built with ``type()``
    claims the module ``type()`` was *called* from — ``sandbox.bridge``. Every
    adapter therefore looked foreign to discovery and no migrated plugin could
    be found at all. Setting it here, once, is what makes a bridged plugin
    discoverable like any other.
    """
    module_name = f"sandboxed_{Path(path).stem}"
    adapter = type(f"Sandboxed{entry}", (base,),
                   {**attributes, "__module__": module_name})

    module = types.ModuleType(module_name)
    module.__file__ = source_path
    setattr(module, adapter.__name__, adapter)
    return adapter, module


def _make_response(answer: dict):
    """Build an ``LLMResponse`` for an escort that answered without dialing.

    Lives here rather than in ``sandbox.hooks`` because that type is a plugin
    type, and the bridge is the one part of the sandbox sanctioned to import
    across that line — the same reason it holds the native base classes.
    """
    from llm import LLMResponse

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

    adapter, module = _build(entry, base, attributes, path, source_path)

    def build_services(config: dict) -> dict:
        """Services are discovered by calling this, not by scanning classes."""
        return {name: adapter()}

    module.build_services = build_services
    return module


def _adapt_frontend(path, entry: str, base, declarations: dict, box_name: str):
    """Build a native-looking frontend backed by a resident box.

    Like a service this is a residency, but a frontend is the family the kernel
    calls *into*, and that changes two things.

    **The loop inverts.** A native frontend blocks in ``start()`` forever. A box
    serializes one call at a time, so a guest that never returned from ``start``
    would hold its box and no ``render`` could get in — the frontend would go
    deaf the moment it started listening. So the guest's ``start`` sets up and
    returns, and this adapter runs the loop on the daemon thread the frontend
    manager already gives it, calling ``poll`` over and over. Between polls is
    when a render lands.

    **Nine render methods become one call.** ``BaseFrontend`` hands subclasses
    nine typed methods; the box gets one ``render(kind, payload)``. Nine wire
    methods for one concept is surface with no payoff, and it lets a guest
    handle the kinds its transport can show and ignore the rest.
    """
    from .frontends import park, project_payload, unpark

    source_path = str(path)
    name = declarations.get("name") or path.stem.split("_", 1)[-1]
    interval = declarations.get("poll_interval")
    try:
        interval = max(0.0, min(float(interval), 5.0))
    except (TypeError, ValueError):
        interval = 0.05

    # A poll that keeps failing is a broken box, not a busy one. Spinning on it
    # would burn a core and fill the log; stopping makes the failure visible.
    # It is also how a console frontend ends at end-of-input: reading a closed
    # console fails, so a piped stdin runs out and the frontend stops itself.
    max_failures = 5

    wants_console = bool(declarations.get("uses_console"))
    restore_on_start = bool(declarations.get("restore_on_start"))

    def __init__(self, shutdown_event=None):
        """Take the host's shutdown Event, if the manager offers one.

        Named as a constructor parameter because that is how ``FrontendManager``
        supplies host resources — it matches parameters against what it has.
        """
        base.__init__(self)
        self._sandbox_box = None
        self._token = ""
        self._stopping = threading.Event()
        self._shutdown_event = shutdown_event

    def _done(self) -> bool:
        """Whether the loop should stop."""
        return self._stopping.is_set() or (
            self._shutdown_event is not None
            and self._shutdown_event.is_set())

    def start(self):
        """Open the box, hand it its authority, then drive it until stopped.

        Runs on the frontend manager's daemon thread, so blocking here is
        correct — this is the loop the guest is no longer allowed to write.
        """
        try:
            self._sandbox_box = get_sandbox().open(
                source_path, entry, name=box_name, manage_lifecycle=False)
        except Exception as exc:
            logger.error("frontend %s did not start: %s", name, exc)
            return

        # Persistent frontend Requests need the same host context native
        # commands receive. The execution object stays host-side; only Request
        # results cross into the box.
        try:
            context_for = getattr(self.commands, "context", None)
            context = (
                context_for(None) if callable(context_for)
                else types.SimpleNamespace(runtime=self.runtime,
                                           session_key=None)
            )
            self._sandbox_box.execution.context = context
        except Exception:
            logger.exception("frontend %s could not bind sandbox context", name)
            self.stop()
            return

        # The desk opens before the guest's start(), so a frontend can submit
        # from its very first line — restoring a session, say.
        self._token = park(self)
        box = self._sandbox_box

        # The console is exclusive. Losing the claim is not fatal — a frontend
        # that cannot read the keyboard may still have something to show — but
        # it must be loud, because the symptom is a frontend that ignores
        # everything typed at it.
        if wants_console:
            from .console import CONSOLE

            if not CONSOLE.claim(self._token):
                logger.error("frontend %s wants the console but %s already "
                             "has it; it will not receive input", name,
                             "another frontend")
        result = box.call("__bind__", token=self._token)
        if not result.ok:
            logger.error("frontend %s could not be bound: %s", name,
                         result.error)
            self.stop()
            return

        if not box.call("start").ok:
            logger.error("frontend %s refused to start", name)
            self.stop()
            return

        # Restoration may synchronously emit form/approval renders. It must
        # happen between guest calls, after start() has released the box.
        if restore_on_start:
            try:
                key = self.session_key(None)
                notice = self.runtime.restore_last_active(key)
                if notice:
                    self.render_messages(key, [notice])
            except Exception:
                logger.exception("frontend %s restore_last_active failed", name)

        failures = 0
        while not self._done() and box.alive:
            outcome = box.call("poll")
            if outcome.ok:
                failures = 0
                # Truthy means "I did work" — go straight back rather than
                # sleeping, so a busy transport is not rate-limited by us.
                if not outcome.data:
                    self._stopping.wait(interval)
                continue

            failures += 1
            logger.warning("frontend %s poll failed (%d/%d): %s", name,
                           failures, max_failures, outcome.error)
            if failures >= max_failures:
                logger.error("frontend %s stopped after repeated poll "
                             "failures", name)
                break
            self._stopping.wait(interval)

        self.stop()

    def stop(self):
        """Stop the loop, close the box, and take the frontend's authority.

        Idempotent: the loop calls it on the way out and the manager calls it
        on unregister, and either may be first.
        """
        self._stopping.set()
        box, self._sandbox_box = self._sandbox_box, None
        if box is not None and box.alive:
            try:
                box.call("stop")
            except Exception:
                logger.exception("frontend %s stop() failed", name)
        # Revoked before the box is closed, so nothing can submit during
        # teardown on a frontend that is already going away. Releasing the
        # console names the token, so a frontend that already lost the claim
        # cannot take it from whoever holds it now.
        if wants_console:
            from .console import CONSOLE

            CONSOLE.release(self._token)
        unpark(self._token)
        self._token = ""
        if box is not None:
            try:
                get_sandbox().close(box_name)
            except Exception:
                logger.exception("failed to close box %s", box_name)

    def _render(self, session_key: str, kind: str, payload=None):
        """One render, forwarded into the box.

        A frontend that cannot show something is not an error the kernel needs
        to hear about — the turn carries on either way — so failures are logged
        and swallowed, the same policy hooks have.
        """
        box = self._sandbox_box
        if box is None or not box.alive:
            return
        result = box.call("render", session_key=session_key, kind=kind,
                          payload=project_payload(kind, payload))
        if not result.ok:
            logger.warning("frontend %s could not render %s: %s", name, kind,
                           result.error)

    def session_key(self, ctx):
        """Ask the box to name a session.

        A transport context is the frontend's own object and cannot cross, so
        what goes in is whatever of it is plain data. Most frontends key off a
        string or a couple of ids, and one that cannot answer falls back to a
        single session rather than losing the message.
        """
        box = self._sandbox_box
        if box is None or not box.alive:
            return "default"
        result = box.call("session_key",
                          ctx=project_payload("session_key", ctx))
        return str(result.data) if result.ok and result.data else "default"

    def _live_session_keys(self):
        """Only the sessions this frontend actually owns.

        The native default is *every* session the runtime knows about, which
        works because each native frontend overrides it — the REPL answers
        ``["default"]``. A sandboxed frontend cannot override a native method,
        and inheriting that default would mean rendering another frontend's
        conversation to this one's transport.

        The tag is already there: ``_tag_session`` stamps ``frontend_name`` on
        every session a frontend submits for, so ownership can be read rather
        than declared. Untagged sessions are included, because a session that
        nobody has claimed is one this frontend may still be about to receive
        — dropping it would lose the first message of a conversation.
        """
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            return []
        return [key for key, session in (getattr(runtime, "sessions", None)
                                         or {}).items()
                if getattr(session, "frontend_name", None) in (None, self.name)]

    def _renderer(kind: str):
        """One native render method, funnelled into the single box call."""
        def render(self, session_key, payload=None):
            """Show something."""
            self._render(session_key, kind, payload)
        render.__name__ = f"render_{kind}"
        render.__doc__ = f"Forward a {kind} render into the box."
        return render

    attributes = {
        "__doc__": f"Sandboxed frontend loaded from {path.name}.",
        "_source_path": source_path,
        "_sandboxed": True,
        "_box": box_name,
        "_entry": entry,
        "name": name,
        "__init__": __init__,
        "_done": _done,
        "_render": _render,
        "_live_session_keys": _live_session_keys,
        "start": start,
        "stop": stop,
        "session_key": session_key,
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value

    # The native names differ from the wire kinds in three places, because the
    # kernel named them for what they are and the wire names them for what a
    # frontend does with them.
    native_names = {"messages": "render_messages",
                    "attachments": "render_attachments",
                    "form_field": "render_form_field",
                    "approval": "render_approval_request",
                    "buttons": "render_buttons",
                    "error": "render_error",
                    "typing": "render_typing",
                    "tool_status": "render_tool_status",
                    "stream_delta": "render_stream_delta"}
    for kind, method in native_names.items():
        attributes[method] = _renderer(kind)

    # ``capabilities`` is declared as a plain dict because a box cannot hold a
    # dataclass; the native side reads attributes off one. Same rebuild as
    # ``_form_step`` does for a command's form.
    attributes["capabilities"] = _capabilities(declarations.get("capabilities"))

    adapter, module = _build(entry, base, attributes, path, source_path)
    return module


def _capabilities(declared):
    """Rebuild a FrontendCapabilities from the literal dict a guest declares.

    Unknown keys are dropped rather than raising: a frontend claiming a
    capability this kernel has never heard of should lose that claim, not fail
    to load.
    """
    from plugins.BaseFrontend import FrontendCapabilities

    if not isinstance(declared, dict):
        return FrontendCapabilities()
    allowed = set(FrontendCapabilities.__dataclass_fields__)
    unknown = set(declared) - allowed
    if unknown:
        logger.warning("frontend declares unknown capabilities: %s",
                       ", ".join(sorted(unknown)))
    return FrontendCapabilities(**{k: v for k, v in declared.items()
                                   if k in allowed})


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
