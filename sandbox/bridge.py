"""The only doorway a plugin enters by.

Every plugin is sandboxed code, and sandboxed code cannot be registered
directly: its ``run`` wants an ``sdk``, and the registry will hand it a
context. So the bridge reads the file, validates it, and answers with a
*native* subclass whose ``run`` forwards into the sandbox and translates the
answer back. To the tool registry, the agent loop, and the frontends, it is an
ordinary plugin. Nothing downstream changes.

**Nothing is imported to find this out.** The same AST pass that reads a
plugin's declarations is what decides whether it can load, so asking costs
nothing and can never run the file being asked about.

**Refusal is reported, never raised.** Every discovery loop reads ``None`` as
"skip this file" with no ``try`` around it, so raising would let one bad
plugin abort the discovery of every other one.

This used to route between two loaders — a file importing ``guest.bases`` came
here, one importing ``plugins.BaseTool`` was imported natively — which let the
two contracts coexist for the length of the migration. Coexistence was the
point, and it stopped being a feature the moment it was only a way for
unmediated code to run in the kernel's own process. A plugin this module will
not carry is now a plugin that does not load.

Store plugins come along for free — discovery already scans the built-in,
workspace, and installed roots through the same code path, so an installed
package loads exactly like a kernel one.

**Three of the five families live here; two do not.** A tool, task or command
is adapted into a function that opens nothing and returns. A service or a
frontend is adapted into an object that *holds a process* for its whole life,
and everything that follows from that — refcounted shared boxes, declared
hooks, bus subscriptions, poll loops, the frontend's inverted start/render
loop — is :mod:`sandbox.residency`. It imports the construction machinery
from here; ``adapt`` reaches back for its two entry points with a
function-local import, which is where the cycle is broken and why.
"""

from __future__ import annotations

import ast
import logging
import types
from pathlib import Path

import prompt_cues

from . import provenance
from .approval import describe_grant
from .facade import Sandbox
from .policy import Chain
from .validator import FAMILIES, validate_file

logger = logging.getLogger("Sandbox")

# The native base class each family's adapter must subclass, so the kernel
# keeps seeing what it expects.
NATIVE_BASES = {
    "tool": ("plugins.native.tool", "BaseTool"),
    "task": ("plugins.native.task", "BaseTask"),
    "command": ("plugins.native.command", "BaseCommand"),
    "service": ("plugins.native.service", "BaseService"),
    "frontend": ("plugins.native.frontend", "BaseFrontend"),
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
    """Give the bridge the sandbox plugins should run in.

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
    """The sandbox plugins run in."""
    if _SANDBOX is None:
        # Through ``configure`` rather than assigned directly: that is what
        # sets ``plugin_roots``, and a sandbox without them resolves
        # ``dependencies_files`` only inside the plugin's own tree.
        configure(Sandbox())
    return _SANDBOX


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

    Returns None for a file this module will not carry — one whose name gives
    no family, one the validator rejected, one declaring no plugin class — and
    logs the specific reason. The caller reports the consequence; there is no
    other way to load a plugin, so None means the capability is absent.
    """
    path = Path(path)
    family = family or family_of(path)
    if not family or family not in NATIVE_BASES:
        return None

    report = validate_file(path)
    if not report.ok:
        logger.error("plugin %s will not load:\n%s",
                     path.name, report.render())
        return None

    declarations = report.declarations
    entry = entry or _entry_from(report.source)
    if not entry:
        logger.error("plugin %s declares no plugin class", path.name)
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

    # A service is not a call, it is a residency: it opens a box and stays. A
    # frontend is one too, but one the kernel *drives* rather than calls. Both
    # live in :mod:`sandbox.residency` — different enough from a per-call
    # plugin to be a different file, not a branch below.
    #
    # Imported here rather than at module scope because residency imports the
    # construction machinery from this module, and the cycle has to break
    # somewhere. It breaks on this side deliberately: a residency is reached
    # *through* the bridge and never instead of it, so the bridge is the half
    # that can afford to resolve late.
    if family in ("service", "frontend"):
        from .residency import _adapt_frontend, _adapt_service

        if family == "service":
            return _adapt_service(path, entry, base, declarations, box_name,
                                  report.source)
        return _adapt_frontend(path, entry, base, declarations, box_name,
                               report.source)

    def _forward(self, context, payload, method: str = "run", paths=None):
        """Run the plugin and translate the answer back.

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
            if not result.ok:
                # Said out loud, because every one of these doorways fails
                # *invisibly*. A failed ``agent_prompt`` is guidance that never
                # arrives, which looks exactly like a plugin with nothing to
                # say; a failed ``form`` is a command that asks nothing. The
                # resident half has warned here since it was written
                # (``_box_prompt``), and the ephemeral half returning a bare
                # ``None`` is why a prompt method broken by a missing runtime
                # went unnoticed — the kernel dropped it, logged nothing, and
                # the agent simply knew less.
                logger.warning("%s failed for '%s': %s", method,
                               self.name or path.stem, result.error)
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
        native base's empty default, so every plugin's point-of-use
        guidance vanished silently while the plugin went on working. The
        failure had no symptom beyond an agent that no longer knew things it
        used to.

        ``PromptContext`` never crosses into the box. Its session key selects
        the ordinary host-side ``SecondBrainContext`` that answers SDK
        Requests, so the guest can use ``sdk.session.get()`` just as it can in
        its normal entry point. Prompt collection remains rooted at ``kernel``
        and carries no approval grant.

        Which session — if any — is the plugin's own cue talking. Below
        ``session`` on the ladder there is nothing to lend, so the guest is
        answered from the kernel context and ``sdk.session.get()`` tells it
        nothing. That is the enforcement half of the tier: a contribution that
        cannot see the session cannot quietly depend on one while riding in a
        prefix shared by all of them.
        """
        cue = prompt_cues.of(self)
        context = get_sandbox().interpreter.context_for_session(
            prompt_cues.session_for(cue, ctx))
        return _cached_prompt(
            self,
            lambda: _forward(self, context, {}, method=_prompt_name),
            stamp=prompt_cues.stamp(cue, ctx),
        )

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
    adapter therefore looked foreign to discovery and no plugin could
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
    """The plugin class's name, read out of the source."""
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


def _box_prompt(plugin, name: str, method: str = "agent_prompt",
                *, for_session: str = "") -> str:
    """Ask a resident box for its prompt contribution.

    A box that is not open contributes nothing rather than raising: a service
    or frontend that failed to load must not also take the system prompt down
    with it.
    """
    box = getattr(plugin, "_sandbox_box", None)
    if box is None or not box.alive:
        return ""
    result = box.call(method, for_session=for_session)
    if not result.ok:
        logger.warning("agent_prompt failed for '%s': %s",
                       name, result.error)
        return ""
    return result.data or ""


def forget_prompt(plugin) -> None:
    """Drop a plugin's cached prompt contribution outright.

    For the *lifetime* resets in ``residency.py``, which are a different
    question from any cue's: a residency's prompt is only knowable while the
    box is open, so opening or closing one invalidates it however still the
    world has been.

    One function because the two attributes must move together — clearing only
    ``_prompt_text`` leaves the stamp matching, and ``_cached_prompt`` would
    answer "" from a residency that is now loaded.
    """
    plugin._prompt_text = None
    plugin._prompt_stamp = None


def _cached_prompt(plugin, produce, *, stamp) -> str:
    """A plugin's system-prompt contribution, recomputed when its cue fires.

    ``_collect`` in ``agent/system_prompt.py`` runs on every *LLM call*, not
    once per turn, and for an ephemeral family every call into the guest is a
    *fresh box* — so forwarding this uncached would cost a module import, or a
    subprocess spawn for anything foreign, per migrated plugin per iteration.

    The cache used to be permanent, which made the other half wrong: a method
    shape exists precisely because its text reads live state, so a tool that
    lists the scripts directory went on describing the directory as it stood
    when the adapter was built. ``prompt_cues`` resolves the two, and the
    plugin says which rung it is on: a ``session``-cued prompt sits still
    through a thousand writes, a ``write``-cued one costs exactly one recompute
    per effect, and neither has to guess about the other.

    One slot per adapter, and an adapter is shared by every session — so two
    live sessions alternately evict each other for anything at rank
    ``session`` or finer. That is unchanged from when the key was a variant
    tuple, and it is a second, quieter reason to declare the rarest cue that is
    still true: a contribution in the stable tier is not keyed on the session
    at all and stops thrashing entirely.

    The lifetime resets in ``residency.py`` stay on top of this: a residency's
    prompt is only knowable while it is resident, which is a question about the
    box rather than about the world.
    """
    if getattr(plugin, "_prompt_stamp", None) == stamp:
        return getattr(plugin, "_prompt_text", "") or ""
    try:
        text = produce() or ""
    except Exception:
        logger.exception("agent_prompt failed for '%s'",
                         getattr(plugin, "name", "?"))
        text = ""
    plugin._prompt_text = text
    # The stamp was taken *before* producing, by the caller: anything that
    # changed while we were asking has to be seen next time, and a prompt
    # method that writes would otherwise cache a stamp its own effect already
    # invalidated.
    plugin._prompt_stamp = stamp
    return text


def _prompt_method(source: str, class_name: str) -> str:
    """``"agent_prompt"`` if this class defines it as a method, else "".

    One name, two shapes: a plugin with nothing conditional to say declares a
    plain string, which the validator reads and the adapter carries for free.
    One whose text depends on live state defines a method, which costs a real
    call into the box — so the two have to be told apart before anything runs,
    and "" here means the literal (if any) was already copied across and
    nothing needs forwarding.
    """
    return "agent_prompt" if _defines(source, class_name, "agent_prompt") else ""


#: The one-shot administrative moments a plugin may declare. Named here rather
#: than spelled at each call site because three places have to agree on them:
#: the guest base that documents them, the package manager that fires them, and
#: the test that pins the pair.
LIFECYCLE_METHODS = ("on_install", "on_uninstall")


def lifecycle_entries(source: str, method: str) -> list:
    """``(entry, plugin_name)`` for every definition of ``method`` in a file.

    ``entry`` is what :func:`sandbox.facade.Sandbox.run` should be pointed at —
    a plugin class, or the function itself when a file with no class defines
    one at module scope. That second shape is free and is what lets a parser,
    an LLM backend or a script have a lifecycle hook; none of them has a base
    class, and none of them is reached through an adapter.

    Only a class that *defines* the method counts. Every plugin inherits a
    no-op from :class:`~guest.bases.BasePlugin`, so asking "does it have one"
    would answer yes for all of them and cost a box per installed file. Same
    reasoning, and the same ``_defines`` call, as :func:`_prompt_method`.

    ``plugin_name`` is the class's declared ``name``, and it matters: the run's
    chain link is what ``policy._owns_setting`` matches against the setting
    registry, and the registry knows ``memory_retrieve`` where the box knows
    ``service_memory_retrieve``. Without it a plugin cleaning up a setting it
    declared itself would be a stranger to its own bookkeeping — the same
    mismatch ``PersistentBox._identity`` exists to fix one layer over. Empty
    when the file declares none, which leaves the caller with the box's own
    name and is the honest answer rather than a guess.

    Never imports. A syntactically broken file answers with nothing, because a
    file that cannot be parsed cannot be run either.
    """
    from .validator import BASE_TO_FAMILY

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(isinstance(base, ast.Name) and base.id in BASE_TO_FAMILY
                   for base in node.bases):
            continue
        if any(isinstance(item, ast.FunctionDef) and item.name == method
               for item in node.body):
            found.append((node.name, _declared_name(node)))

    if not found:
        # Module scope only when no class claimed it — a plugin file that also
        # happens to hold a bare function of this name means the class.
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == method:
                found.append((method, ""))
                break
    return found


def _declared_name(node) -> str:
    """A class's literal ``name = "..."``, or "" when it declares none."""
    for item in node.body:
        if not isinstance(item, ast.Assign):
            continue
        for target in item.targets:
            if (isinstance(target, ast.Name) and target.id == "name"
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)):
                return item.value.value
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
