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

from . import provenance
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
               "memory_mb", "requests", "exports", "hooks",
               # The per-class breakdown the bridge itself reads to decide how
               # many adapters to build. Copying it onto an adapter would put
               # a second, staler copy of every declaration on the object that
               # already carries them flattened.
               "classes", "entry"}

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
            from plugins.plugin_paths import PLUGIN_ROOTS
            sandbox.plugin_roots = [root.path for root in PLUGIN_ROOTS]
        except Exception:
            logger.debug("could not resolve plugin roots for dependencies")


def get_sandbox() -> Sandbox:
    """The sandbox migrated plugins run in."""
    global _SANDBOX
    if _SANDBOX is None:
        # Through ``configure`` rather than assigned directly: that is what
        # sets ``plugin_roots``, and a sandbox without them resolves
        # ``dependencies_files`` only inside the plugin's own tree.
        configure(Sandbox())
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
    # Where a raise happened, if one did. Each family carries it differently:
    # a tool has a field for it, a command returns markdown, and TaskResult
    # feeds the ingestion pipeline and is not worth growing for this.
    trace = result.traceback

    if family == "command":
        # Commands return markdown, or None.
        if result.ok:
            return result.data
        return (f"Error: {result.error}\n\n```\n{trace}```" if trace
                else f"Error: {result.error}")

    if family == "task":
        task_result = getattr(native_module, "TaskResult", None)
        if task_result is None:
            return result
        return task_result(success=result.ok,
                           error=f"{result.error}\n{trace}" if trace
                                 else result.error,
                           data=result.data or [],
                           also_contains=list(result.also_contains),
                           discovered_paths=list(result.discovered_paths))

    tool_result = getattr(native_module, "ToolResult", None)
    if tool_result is None:
        return result
    return tool_result(success=result.ok, error=result.error,
                       data=result.data, llm_summary=result.llm_summary,
                       attachment_paths=list(result.attachment_paths),
                       traceback=trace)


def _task_results(result, native_module, count: int) -> list:
    """One ``TaskResult`` per path, which is what the orchestrator zips.

    A guest task returns a single ``Result`` — one call, one answer — while
    ``Orchestrator._execute`` does ``zip(paths, results)`` and expects one
    outcome per path. Handing it a lone result made ``zip`` stop after the
    first: every other path in the batch was neither completed nor failed, so
    it stayed claimed and was never retried. Silent, and invisible until
    somebody wondered why a folder had stopped indexing.

    Two shapes come back from a guest, and the difference is real:

    - ``per_path`` given — the task judged each file, so each entry becomes its
      own result and ``data`` lands on the path that produced it.
    - ``per_path`` empty — the task succeeded or failed *as a batch*, so the
      one outcome applies to all of them. ``data`` rides on the first entry
      alone, because ``_handle_success`` writes the whole of ``result.data``
      for whichever path it is handling; repeating it would write every row
      once per path in the batch.
    """
    task_result = getattr(native_module, "TaskResult", None)
    if task_result is None:
        return [result] * max(count, 1)

    trace = result.traceback

    def _one(ok, error, data, also_contains, discovered):
        return task_result(
            success=ok,
            error=f"{error}\n{trace}" if trace and error else error,
            data=data or [],
            also_contains=list(also_contains or []),
            discovered_paths=list(discovered or []))

    if result.per_path:
        # Trusted as the task's own account of its batch. A length mismatch is
        # left alone rather than padded: zip truncates to the shorter side, and
        # inventing outcomes for paths the task did not mention would be worse
        # than the paths simply staying pending for the next sweep.
        return [_one(entry.get("ok", True), entry.get("error", ""),
                     entry.get("data"), entry.get("also_contains"),
                     entry.get("discovered_paths"))
                for entry in result.per_path
                if isinstance(entry, dict)]

    first = _one(result.ok, result.error, result.data,
                 result.also_contains, result.discovered_paths)
    rest = [_one(result.ok, result.error, None, None, None)
            for _ in range(max(count - 1, 0))]
    return [first, *rest]


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
    source_path = str(Path(path).resolve())
    box_name = declarations.get("box") or path.stem

    # What one approval is allowed to buy. Read here, once, from the same
    # declaration the validator checks and the approval dialog names — so the
    # grant a user answers and the grant the policy honours cannot disagree.
    granted = frozenset(declarations.get("requests") or ())

    # A service is not a call, it is a residency: it opens a box and stays.
    # Different enough that it gets its own builder rather than a branch in
    # the per-call machinery below.
    if family == "service":
        return _adapt_service(
            path,
            entry,
            base,
            declarations,
            box_name,
            report.source,
        )

    # A frontend is a residency too, but one the kernel drives rather than
    # calls: it owns a loop and nine render doorways. Its own builder for the
    # same reason a service has one.
    if family == "frontend":
        return _adapt_frontend(path, entry, base, declarations, box_name,
                               report.source)

    def _forward(self, context, payload, method: str = "run", paths=None):
        """Run the migrated plugin and translate the answer back.

        The context is passed *per call* rather than held, because two calls
        can be in flight with different sessions and users behind them.

        ``paths`` is passed only by a path task, and only so the translation
        knows how many outcomes the orchestrator is about to zip against. It
        is deliberately not read from ``payload``: a guest may return fewer
        entries than it was given, and the count that matters is what the
        kernel handed over.
        """
        # Only the root is set here. ``Sandbox.start`` pushes the execution's
        # own name, and pushing it here too would put the plugin in the chain
        # twice — which the cycle detector correctly refuses.
        #
        # When something already inside the sandbox reached us — a tool
        # calling a tool — we descend from *its* chain instead of starting a
        # fresh one beside it. Two things follow, and both are the documented
        # intent rather than a side effect. The cycle detector and MAX_DEPTH
        # start working, because there is finally a stack to measure. And the
        # callee spends the *caller's* grant: ``granted`` is deliberately not
        # consulted here, since re-deriving it from the callee's own
        # declarations is exactly the widening ``Chain.push`` exists to
        # prevent. A command's own manifest is read only when the command is
        # the root of the call.
        caller = provenance.current()
        chain = caller.chain if caller is not None else Chain(
            root=_root_for(context, family),
            approved=(
                granted
                if getattr(context, "approved_by_state_machine", False)
                else None
            ),
        )
        result = get_sandbox().run(
            source_path, entry, kwargs=payload, chain=chain, context=context,
            name=self.name or path.stem, method=method)
        # ``run_event`` is a second *entry point* rather than an exported
        # helper, so it wants the family's real return translation — the
        # orchestrator reads a ``TaskResult``, and handing it raw data means a
        # successful sweep is indistinguishable from a crashed one.
        if method not in ("run", "run_event"):
            return result.data if result.ok else None
        # A path task is the one entry point whose caller wants a *list*: the
        # orchestrator zips one outcome per path. ``run_event`` reads a single
        # result, so the two cannot share a translation.
        if family == "task" and method == "run" and paths is not None:
            return _task_results(result, native_module, len(paths))
        return _result_to_native(family, result, native_module)

    # The families disagree about argument order, and the adapter is the one
    # place that has to know. A generic ``run(self, context, *args)`` would
    # silently bind a task's ``paths`` to its ``context``.
    def run_tool(self, context, **kwargs):
        """Native tool contract."""
        return _forward(self, context, dict(kwargs))

    def run_task(self, paths, context):
        """Native task contract: a list of outcomes, one per path."""
        batch = list(paths or [])
        return _forward(self, context, {"paths": batch}, paths=batch)

    def run_task_event(self, run_id, payload, context):
        """Native event-task contract.

        A task has *two* entry points, not one, and only ``run`` was bridged —
        so a migrated task declaring ``trigger = "event"`` loaded, registered,
        subscribed to its channel, and then did nothing at all when the channel
        fired: the orchestrator called ``run_event``, got the native base
        class's do-nothing default, and recorded a successful run. Silence that
        looks like success is the worst shape a gap can take, which is why this
        is forwarded rather than left to the one-entry-point assumption.

        ``run_id`` is deliberately not passed through. It identifies a
        ``task_runs`` row the guest can neither read nor write, so it would be
        an opaque token with nothing to spend it on.
        """
        return _forward(self, context, {"payload": dict(payload or {})},
                        method="run_event")

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

    def agent_prompt(self, ctx):
        """Native prompt-contribution contract, answered inside the box.

        Only attached when the guest defines ``agent_prompt`` as a *method*.
        A guest that declares it as a plain string needs nothing here — the
        declaration is copied onto the adapter as an ordinary attribute and
        ``_collect`` reads it directly, which is why the static case costs no
        box call at all.

        Bridged for the same reason ``form`` and ``run_event`` are: a doorway
        the kernel calls and the adapter does not forward is answered by the
        native base's empty default, so every migrated plugin's point-of-use
        guidance vanished silently while the plugin went on working. The
        failure had no symptom beyond an agent that no longer knew things it
        used to.

        The ``PromptContext`` the kernel hands us is deliberately *not* what the
        box answers from. It is a light read-only bag for building a prompt, not
        a ``SecondBrainContext``, so passing it through would answer the guest's
        Requests out of a half-built world. ``None`` lets the interpreter build
        a kernel context instead, and roots the chain at ``kernel`` — the honest
        reading, since collecting prompt text is the kernel's own act and no
        session's. The chain therefore carries no grant, so an unsafe Request
        from here is refused; the guest contract already says not to make one.
        """
        return _cached_prompt(
            self, lambda: _forward(self, None, {}, method=_prompt_name))

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
        # Only when the file actually defines one, for the same reason
        # ``form`` is conditional: an adapter carrying a doorway the guest
        # never wrote would answer the orchestrator's call by forwarding into
        # nothing.
        **({"run_event": run_task_event} if family == "task"
           and _defines(report.source, entry, "run_event") else {}),
        # The sentence the approval dialog asks, rendered from the same
        # declaration the grant is built from so the question a user answers
        # and the authority they hand over cannot drift apart.
        "approval_prompt": describe_grant(
            declarations.get("name") or path.stem.split("_", 1)[-1], granted),
    }
    for key, value in declarations.items():
        if key not in NOT_CARRIED:
            attributes[key] = value
    # After the declaration copy, so a method always beats a literal of the
    # same name. Only attached when the guest wrote a method: a string
    # declaration already arrived above and needs no forwarding.
    _prompt_name = _prompt_method(report.source, entry)
    if _prompt_name:
        attributes["agent_prompt"] = agent_prompt
    attributes.setdefault("name", path.stem.split("_", 1)[-1])

    adapter, module = _build(entry, base, attributes, path, source_path)
    return module


def _build(entry: str, base, attributes: dict, path, source_path: str,
           module=None):
    """Make the adapter class and the synthetic module that carries it.

    One place because of ``__module__``. Discovery only accepts classes that
    belong to the module it just loaded, and a class built with ``type()``
    claims the module ``type()`` was *called* from — ``sandbox.bridge``. Every
    adapter therefore looked foreign to discovery and no migrated plugin could
    be found at all. Setting it here, once, is what makes a bridged plugin
    discoverable like any other.

    Pass ``module`` to add another adapter to a module already built — how a
    file declaring several services ends up as one module holding all of them,
    which is what discovery expects to find.
    """
    module_name = f"sandboxed_{Path(path).stem}"
    adapter = type(f"Sandboxed{entry}", (base,),
                   {**attributes, "__module__": module_name})

    if module is None:
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


MIN_POLL_INTERVAL = 0.01
MAX_POLL_INTERVAL = 3600.0
DEFAULT_POLL_FAILURES = 5


def _poll_settings(declarations: dict, default_interval: float) -> tuple:
    """Clamp a resident plugin's polling wishes to kernel-owned limits."""
    try:
        raw = float(declarations.get("poll_interval", default_interval))
    except (TypeError, ValueError):
        raw = default_interval
    interval = (
        0.0
        if raw <= 0
        else max(MIN_POLL_INTERVAL, min(raw, MAX_POLL_INTERVAL))
    )
    try:
        failures = int(
            declarations.get("max_poll_failures", DEFAULT_POLL_FAILURES)
        )
    except (TypeError, ValueError):
        failures = DEFAULT_POLL_FAILURES
    return interval, max(1, min(failures, 100))


def _drive_polls(
    *,
    family: str,
    name: str,
    box,
    stopping,
    interval: float,
    max_failures: int,
    done=None,
):
    """Drive one resident plugin's kernel-owned poll loop.

    Returns whether the loop ended because it was asked to. A box that dies
    under the loop used to end it in complete silence — the ``while``
    condition simply went false — which is indistinguishable from a clean
    stop and was how a starved REPL disappeared without a word.
    """
    failures = 0
    while not stopping.is_set() and not (callable(done) and done()):
        if not box.alive:
            logger.error("%s %s stopped: its box is no longer running",
                         family, name)
            return False
        outcome = box.call("poll")
        if outcome.ok:
            failures = 0
            # Truthy means work remains: drain it before sleeping.
            if not outcome.data:
                stopping.wait(interval)
            continue

        failures += 1
        logger.warning(
            "%s %s poll failed (%d/%d): %s",
            family,
            name,
            failures,
            max_failures,
            outcome.error,
        )
        if failures >= max_failures:
            logger.error(
                "%s %s stopped polling after repeated failures",
                family,
                name,
            )
            return False
        stopping.wait(interval)
    return True


class _Occupant:
    """One service's view of a box it may be sharing.

    Everything that calls into a resident service — the export forwarders, the
    poll loop, the bus deliverer, the hook shims, the prompt collector —
    reaches ``service._sandbox_box`` and calls ``.call(method, ...)``. Handing
    each adapter a handle that already knows its own target keeps every one of
    those sites written exactly as it was, instead of threading a ``target``
    argument through five call paths that have no other reason to know a box
    can hold more than one thing.
    """

    def __init__(self, box, target: str):
        self._box = box
        self._target = target

    @property
    def alive(self) -> bool:
        """Whether the underlying box can still take calls."""
        return self._box is not None and self._box.alive

    @property
    def execution(self):
        """The box's execution, for callers that adjust its context."""
        return self._box.execution

    def call(self, method: str, *args, **kwargs):
        """Invoke a method on *this* occupant."""
        return self._box.call(method, *args, target=self._target, **kwargs)


class _Residency:
    """The one box a file's services share, and who is still using it.

    A file holding two services registers as two services: the kernel loads
    and unloads each by name, with no idea they are neighbours. But there is
    one process behind them, and it exists precisely *because* they share
    something expensive — so the naive mapping, where each adapter's
    ``unload`` closes the box, means unloading the text embedder kills the
    image embedder's model too, with no symptom beyond calls suddenly failing.

    Hence a refcount. The first adapter to load opens the box; the last to
    unload closes it. Everything in between joins what is already there.
    """

    def __init__(self, source_path: str, box_name: str, entries: list):
        self.source_path = source_path
        self.box_name = box_name
        self.entries = list(entries)
        self.box = None
        self._users: set = set()
        self._lock = threading.Lock()

    def join(self, name: str, chain):
        """Open the box if nobody has yet, and record this user."""
        with self._lock:
            if self.box is None or not self.box.alive:
                self.box = get_sandbox().open(
                    self.source_path,
                    self.entries[0] if self.entries else "",
                    name=self.box_name,
                    chain=chain,
                    # Empty for a lone occupant, so a one-service file opens
                    # exactly the box it always did.
                    entries=self.entries if len(self.entries) > 1 else (),
                )
            self._users.add(name)
            return self.box

    def leave(self, name: str):
        """Drop this user, and close the box once the last one has gone."""
        with self._lock:
            self._users.discard(name)
            if self._users or self.box is None:
                return
            self.box = None
        try:
            get_sandbox().close(self.box_name)
        except Exception:
            logger.exception("failed to close box %s", self.box_name)


def _adapt_service(
    path,
    entry: str,
    base,
    declarations: dict,
    box_name: str,
    source: str,
):
    """Build native-looking services backed by one resident box.

    The other families are *calls*: run once, translate the answer, tear down.
    A service is a *residency*, so the adapter maps the native lifecycle onto
    the box lifecycle instead:

        _load()   ->  open the box (start() runs inside it)
        unload()  ->  close the box (stop() runs inside it)

    and every method the plugin lists in ``exports`` becomes a real method on
    the adapter, because native callers reach services by attribute access
    rather than through ``service.call``.

    A file may declare several service classes — the shape ``build_services``
    has always supported, and the reason it returns a dict. They share one box
    and are told apart by the call's ``target``, so a heavy library is
    imported once and an accelerator context is created once. Everything below
    is built per class; only :class:`_Residency` is shared.
    """
    source_path = str(path.resolve())
    classes = list(declarations.get("classes") or [])
    if not classes:
        # A file the validator did not describe per class — a hand-built
        # declarations dict in a test, most likely. Treat it as the one class
        # it names, which is what this function always assumed.
        classes = [{"entry": entry, **declarations}]

    residency = _Residency(source_path, box_name,
                           [c.get("entry") or entry for c in classes])

    adapters = {}
    module = None
    for spec in classes:
        adapter, module = _service_adapter(
            path, spec.get("entry") or entry, base, spec, residency, source,
            module)
        adapters[adapter.name] = adapter

    def build_services(config: dict) -> dict:
        """Services are discovered by calling this, not by scanning classes."""
        return {name: adapter() for name, adapter in adapters.items()}

    module.build_services = build_services
    return module


def _service_adapter(
    path,
    entry: str,
    base,
    declarations: dict,
    residency: "_Residency",
    source: str,
    module=None,
):
    """One service class's adapter. See :func:`_adapt_service`."""
    source_path = residency.source_path
    box_name = residency.box_name
    name = declarations.get("name") or path.stem.split("_", 1)[-1]
    exports = list(declarations.get("exports") or [])
    interval, max_failures = _poll_settings(declarations, 0.0)
    polls = interval > 0 and _defines(source, entry, "poll")

    if not exports:
        # Not fatal — a service may exist only for its side effects — but it
        # is nearly always a forgotten declaration, and the symptom (every
        # call failing as "not exported") points nowhere near the cause.
        logger.warning("sandboxed service %s declares no exports; nothing "
                       "will be able to call it", path.name)

    def __init__(self):
        """Initialize native adapter state without importing guest code."""
        base.__init__(self)
        self._poll_stop = threading.Event()
        self._poll_thread = None

    def _load(self) -> bool:
        """Open the resident box. Its start() runs inside."""
        self._poll_stop.clear()
        # A residency's prompt text is only knowable while it is resident, so
        # the cache is scoped to one lifetime rather than to the adapter. Asked
        # before load, the answer is "nothing" — and that must not be the answer
        # forever after.
        self._prompt_text = None
        try:
            box = residency.join(name, Chain(root=f"service:{name}"))
        except Exception as exc:
            logger.error("service %s did not start: %s", name, exc)
            return False
        # ``target`` is this service's own name, and empty when it is the box's
        # only occupant — which is what keeps a one-service file's wire
        # identical to what it always sent.
        self._sandbox_box = _Occupant(
            box, name if len(residency.entries) > 1 else "")
        self.loaded = True
        # Binding and loading happen in either order depending on whether this
        # is boot or a live reload, so both ends call the same idempotent sync.
        _sync_hooks(self)
        _listen(self)
        if polls:
            box = self._sandbox_box
            self._poll_thread = threading.Thread(
                target=_drive_polls,
                kwargs={
                    "family": "service",
                    "name": name,
                    "box": box,
                    "stopping": self._poll_stop,
                    "interval": interval,
                    "max_failures": max_failures,
                },
                daemon=True,
                name=f"{name}-poll",
            )
            self._poll_thread.start()
        return True

    def unload(self):
        """Close the box and step away from every doorway and channel."""
        self.loaded = False
        self._prompt_text = None
        self._poll_stop.set()
        thread, self._poll_thread = self._poll_thread, None
        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)
        _unhook(self)
        _deafen(self)
        self._sandbox_box = None
        # The box closes when the *last* service in this file unloads. See
        # ``_Residency``: unloading one of a pair must not take the other's
        # loaded model down with it.
        residency.leave(name)

    def bind_runtime(self, *, runtime=None, **_):
        """Receive the runtime. Idempotent, and may arrive before or after load."""
        if runtime is not None:
            self._runtime = runtime
        _sync_hooks(self)

    def agent_prompt(self, ctx):
        """Prompt contribution, answered in the resident box.

        No spawn to pay for here, but still cached: a resident box serializes
        its calls and prompt collection runs on the turn thread, so queuing
        behind a long export every turn would be a stall for no gain.
        """
        return _cached_prompt(self, lambda: _box_prompt(self, name, _prompt_name))

    def _export(method: str):
        """One forwarding method, so callers see an ordinary service."""
        def call(self, *args, **kwargs):
            """Invoke an exported method inside the box."""
            box = getattr(self, "_sandbox_box", None)
            if box is None or not box.alive:
                raise ServiceCallFailed(
                    f"service {name!r} is not loaded")
            result = box.call(method, *args, **kwargs)
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
        "__init__": __init__,
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
    _prompt_name = _prompt_method(source, entry)
    if _prompt_name:
        attributes["agent_prompt"] = agent_prompt
    for method in exports:
        attributes[method] = _export(method)

    return _build(entry, base, attributes, path, source_path, module)


def _adapt_frontend(path, entry: str, base, declarations: dict, box_name: str,
                    source: str = ""):
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

    source_path = str(Path(path).resolve())
    name = declarations.get("name") or path.stem.split("_", 1)[-1]
    interval, max_failures = _poll_settings(declarations, 0.05)

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
        self._prompt_text = None
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
                source_path,
                entry,
                name=box_name,
                chain=Chain(root=f"frontend:{name}"),
                manage_lifecycle=False,
            )
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

        asked = _drive_polls(
            family="frontend",
            name=name,
            box=box,
            stopping=self._stopping,
            interval=interval,
            max_failures=max_failures,
            done=self._done,
        )
        if not asked:
            # The guest cannot say this — its box is the thing that died — so
            # the host says it, on the console it already owns. Without this a
            # dead console frontend is a terminal that echoes nothing and
            # explains nothing, which is exactly how the deadlock presented.
            logger.error("frontend %s died; it is no longer accepting input",
                         name)
            if wants_console:
                from .console import CONSOLE
                CONSOLE.write(
                    f"\n[frontend {name} stopped: see app.log. Restart the "
                    f"app to recover.]")

        self.stop()

    def stop(self):
        """Stop the loop, close the box, and take the frontend's authority.

        Idempotent: the loop calls it on the way out and the manager calls it
        on unregister, and either may be first.
        """
        self._stopping.set()
        # Scoped to one residency, for the reason ``_load`` gives: asked while
        # the box is shut, the honest answer is "nothing", and it must not
        # outlive the shutdown that made it true.
        self._prompt_text = None
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

    _prompt_name = _prompt_method(source, entry)
    if _prompt_name:
        attributes["agent_prompt"] = (
            lambda self, ctx, _m=_prompt_name: _cached_prompt(
                self, lambda: _box_prompt(self, name, _m)))

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


def _root_for(context, family: str = "") -> str:
    """What caused this call, for the chain of provenance.

    A person's own action roots at ``user``. A slash command they typed roots at
    ``user:command`` specifically, because policy needs to recognise that case
    and cannot do it from ``user`` alone: ``Chain()``'s default root is
    ``user``, so a rule keyed on the bare string would also fire for any chain
    built without one — the opposite of a deliberate grant. Only the command
    family is qualified; naming every family would rewrite the root of every
    tool call for no decision's sake. The suffix is assigned here, from the
    dispatch path, so guest code cannot claim it.
    """
    if context is None:
        return "kernel"
    if getattr(context, "user_initiated", False):
        return "user:command" if family == "command" else "user"
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


def _box_prompt(plugin, name: str, method: str = "agent_prompt") -> str:
    """Ask a resident box for its prompt contribution.

    A box that is not open contributes nothing rather than raising: a service
    or frontend that failed to load must not also take the system prompt down
    with it.
    """
    box = getattr(plugin, "_sandbox_box", None)
    if box is None or not box.alive:
        return ""
    result = box.call(method)
    if not result.ok:
        logger.warning("agent_prompt failed for '%s': %s",
                       name, result.error)
        return ""
    return result.data or ""


def _cached_prompt(plugin, produce) -> str:
    """A migrated plugin's system-prompt contribution, computed once.

    ``_collect`` in ``agent/system_prompt.py`` runs per turn for every in-scope
    plugin, and for an ephemeral family every call into the guest is a *fresh
    box* — so forwarding this uncached would cost a subprocess spawn per
    migrated tool per turn. Caching is safe because the guest contract already
    demands it: ``agent_prompt`` is documented as cheap, stable and landing in
    a cacheable block.

    Nothing invalidates the cache, on purpose. A changed file goes through
    ``PluginWatcher``, which rebuilds the adapter — so a new instance is what a
    new answer arrives on, and an adapter that outlives the edit is one nothing
    reloaded.
    """
    cached = getattr(plugin, "_prompt_text", None)
    if cached is not None:
        return cached
    try:
        text = produce() or ""
    except Exception:
        logger.exception("agent_prompt failed for '%s'",
                         getattr(plugin, "name", "?"))
        text = ""
    plugin._prompt_text = text
    return text


def _prompt_method(source: str, class_name: str) -> str:
    """Which spelling of the prompt doorway this class defines, if any.

    ``agent_prompt_for`` was the old name for the dynamic half; the static
    half was a separate ``agent_prompt`` string. They are one name now, but an
    unmigrated store plugin still writes the old one, and an adapter that
    forwarded neither would drop its guidance in silence. Returns the method
    name to call inside the box, or "" when the file defines no method at all
    — in which case a literal declaration (if any) was already copied across
    and nothing needs forwarding.
    """
    for name in ("agent_prompt", "agent_prompt_for"):
        if _defines(source, class_name, name):
            return name
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
