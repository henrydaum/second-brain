"""The doorways around the agent turn — one registry, six moments.

Every agent turn is the same short ritual: the turn starts, the model thinks,
the agent acts, think/act repeats, the turn ends. This module puts a labeled
doorway at every moment of that ritual, and the one rule of the whole system
is: **nothing influences a turn except by standing at a doorway.** If nobody
is registered at a doorway, the kernel walks straight through and behaves
exactly as it would with no hooks at all.

The six moments (in the order a turn meets them):

=================  ==========  =====================================================
Moment             Kind        What standing there means
=================  ==========  =====================================================
``turn_start``     adjuster    The agent is about to begin; slip a note into its
                               pocket (mutate the session: prompt extras,
                               staged attachments, queued actions).
``shape_scope``    adjuster    Here is the toolbox the agent will see; hand back
                               a changed one.
``vet_permission`` verdict     The agent wants to run something sensitive; say
                               yes, say no, or stay silent.
``model_call``     escort      Own the round trip to the model: rewrite the
                               request, place the call yourself, inspect the
                               answer, and go around again if you don't like it.
``end_turn``       verdict     The doorman at the exit: the agent says "I'm
                               done" — let it leave, send it back with a note,
                               or demand one last tool call first.
``turn_finish``    observer    The turn is over; look at what happened. Touch
                               nothing.
=================  ==========  =====================================================

Three kinds of doorway:

- **Observer** — watches, touches nothing.
- **Adjuster** — handed a thing, may return a changed thing, never sees what
  happens next. (A verdict is an adjuster whose answer is a decision object;
  the first non-``None`` answer wins.)
- **Escort** — signature ``fn(ctx, payload, proceed)``. The escort holds both
  the request and the phone: it decides when to dial (``proceed``), sees the
  response before anyone else, and may redial. Escorts nest like an onion —
  the first registered is the outermost wrapper.

The uniform contract, enforced here so every hook can rely on it:

- Every hook receives ``(ctx, payload)`` — ``ctx`` is a :class:`HookContext`
  (session, runtime, moment), ``payload`` is the moment-specific dataclass.
- Return ``None`` to abstain; return a value to speak.
- A raising hook is logged and skipped — a hook can never break a turn. For
  escorts, "skipped" means transparent: the call proceeds as if the escort
  were not there (and a response already obtained is never thrown away or
  fetched twice).

Registration: ``runtime.hooks.add(moment, fn)`` from a service's
``bind_runtime``/``_load``; ``runtime.hooks.remove(fn)`` at unload — always
with the *original* function, even for hooks registered through the legacy
``add_*`` aliases (the registry tracks the adapter for you, so plugin unload
never leaks a hook). See ``templates/hook_template.py`` for worked examples.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("Hooks")


# ──────────────────────────────────────────────────────────────────────
# The six moments.
# ──────────────────────────────────────────────────────────────────────

TURN_START = "turn_start"
SHAPE_SCOPE = "shape_scope"
VET_PERMISSION = "vet_permission"
MODEL_CALL = "model_call"
END_TURN = "end_turn"
TURN_FINISH = "turn_finish"

MOMENTS = (TURN_START, SHAPE_SCOPE, VET_PERMISSION, MODEL_CALL, END_TURN, TURN_FINISH)


# ──────────────────────────────────────────────────────────────────────
# The envelope and the moment payloads.
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HookContext:
    """The envelope every hook receives, at every doorway: whose turn this
    is (``session``), the runtime it lives in, and which doorway we are
    standing at (``moment``)."""

    session: Any
    runtime: Any
    moment: str


@dataclass
class ModelRequest:
    """One outgoing trip to the model, materialized so escorts can rewrite it.

    ``llm`` is the brain that will take the call; ``messages`` is exactly what
    it will be shown; ``tools`` is the toolbox offered (provider schemas);
    ``tool_choice`` forces tool use when the backend supports it (see
    ``BaseLLM.supports_tool_choice``); ``params`` are extra provider kwargs
    forwarded to ``chat_with_tools`` only when non-empty; ``attachments`` is
    the media bundle riding along on this call.
    """

    llm: Any
    messages: list
    tools: Optional[list] = None
    tool_choice: Any = None
    params: dict = field(default_factory=dict)
    attachments: Any = None


@dataclass
class TurnEnding:
    """What the doorman at the exit is shown: the text the agent wants to
    leave behind, why the turn is ending, and how many times a doorman has
    already sent it back this turn (the fire budget's odometer)."""

    final_text: Optional[str] = None
    # "model_finished" — the model produced final text and wants to stop.
    # "budget_exhausted" — the loop ran out of tool-call/iteration budget.
    reason: str = "model_finished"
    doorman_fires: int = 0


@dataclass
class TurnOutcome:
    """What the ``turn_finish`` observers see once the logical turn is over."""

    ok: bool = True
    cancelled: bool = False
    final_text: Optional[str] = None


@dataclass
class PermissionQuery:
    """The question a ``vet_permission`` verdict answers: may this command run?"""

    tool_name: Optional[str]
    command: str


# ──────────────────────────────────────────────────────────────────────
# Doorman verdicts (returned from ``end_turn`` hooks).
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Allow:
    """Let the agent leave. Equivalent to abstaining, but explicit — and it
    short-circuits later doormen, so a policy can positively wave someone
    through."""


@dataclass
class SendBack:
    """Send the agent back inside with a note; the loop clears the final text
    and asks the model again. ``ephemeral=True`` shows the note to the model
    without recording it in history; ``allow_tools=False`` makes the comeback
    call text-only (how the kernel's over-budget doorman gets a summary
    without more tool calls)."""

    note: str
    ephemeral: bool = False
    allow_tools: bool = True


@dataclass
class RequireTool:
    """Demand one specific tool before the agent may leave: the loop issues
    one more model call offering only that tool, forced via ``tool_choice``
    where the backend supports it, degrading to a :class:`SendBack`-style
    instruction where it doesn't."""

    name: str
    note: str = ""


@dataclass
class Redrive:
    """End this drive without ending the logical turn and have the runtime
    immediately re-drive it (the ``session.restart_turn`` semantics as a
    verdict). The re-driven loop finishes the turn; turn starters do not
    re-run."""


# Legacy type aliases, kept for reference and for plugin code written against
# the original registry.
PermissionGate = Callable[[Any, Optional[str], str], Optional["PermissionVerdict"]]
ScopeShaper = Callable[[Any, Any], Any]
TurnStarter = Callable[[Any], None]
TurnFinalizer = Callable[[Any], None]
LLMSelector = Callable[[Any, Any], Optional[str]]


class PermissionVerdict:
    """A gate's answer to "may this command run?".

    ``allow`` is the decision; ``reason`` is the model-facing explanation shown
    when a call is denied. A gate that has no opinion returns ``None`` instead
    of a verdict, letting the next gate (or the kernel default) decide.
    """

    __slots__ = ("allow", "reason")

    def __init__(self, allow: bool, reason: str = ""):
        """Initialize the verdict."""
        self.allow = bool(allow)
        self.reason = reason


class HookRegistry:
    """The switchboard: one labeled socket per moment, hung off the runtime
    as ``runtime.hooks``. Plugins plug callables into sockets; at each moment
    the kernel walks to the socket and calls whatever is plugged in. An empty
    socket costs nothing."""

    def __init__(self):
        """Initialize an empty registry."""
        self._hooks: dict[str, list[Callable]] = {m: [] for m in MOMENTS}
        # original fn -> adapter actually registered, so remove(original)
        # works for hooks added through the legacy aliases (plugin unload
        # must never leak a hook).
        self._adapters: dict[Callable, Callable] = {}
        # Legacy LLM selectors are kept addressable on their own so
        # ``select_llm`` (drive-time resolution in runtime_config.active_llm)
        # keeps working; the same selector is also registered as a
        # ``model_call`` escort so per-call resolution matches.
        self._legacy_llm_selectors: list[Callable] = []
        self._turn_attachments: dict[str, list[Any]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Registration (called by plugins at load).
    # ──────────────────────────────────────────────────────────────────

    def add(self, moment: str, fn: Callable) -> None:
        """Stand ``fn`` at the ``moment`` doorway. The one registration path."""
        if moment not in self._hooks:
            raise ValueError(f"Unknown hook moment: {moment!r}. Moments: {', '.join(MOMENTS)}")
        self._hooks[moment].append(fn)

    def remove(self, fn: Callable) -> None:
        """Walk ``fn`` away from every doorway (for plugin unload).

        Accepts the *original* function even when it was registered through a
        legacy alias — the adapter mapping is resolved here.
        """
        target = self._adapters.pop(fn, fn)
        for bucket in self._hooks.values():
            try:
                bucket.remove(target)
            except ValueError:
                pass
        try:
            self._legacy_llm_selectors.remove(fn)
        except ValueError:
            pass

    def _add_adapted(self, moment: str, original: Callable, adapter: Callable) -> None:
        """Register ``adapter`` at ``moment`` on behalf of ``original``."""
        self._adapters[original] = adapter
        self.add(moment, adapter)

    # --- legacy aliases: the original bespoke registration surface. Each is
    # now a thin coat over ``add()`` with a signature adapter, kept so store
    # plugins and tests written against the old API keep working unchanged.

    def add_permission_gate(self, gate: PermissionGate) -> None:
        """Legacy alias: a ``vet_permission`` verdict with the old
        ``(session, tool_name, command)`` signature."""
        def adapter(ctx: HookContext, query: PermissionQuery):
            return gate(ctx.session, query.tool_name, query.command)
        self._add_adapted(VET_PERMISSION, gate, adapter)

    def add_scope_shaper(self, shaper: ScopeShaper) -> None:
        """Legacy alias: a ``shape_scope`` adjuster with the old
        ``(session, registry)`` signature."""
        def adapter(ctx: HookContext, registry):
            return shaper(ctx.session, registry)
        self._add_adapted(SHAPE_SCOPE, shaper, adapter)

    def add_turn_starter(self, starter: TurnStarter) -> None:
        """Legacy alias: a ``turn_start`` adjuster with the old ``(session)``
        signature.

        The starter receives the session with the latest user text already in
        ``session.history``; it injects via ``session.system_prompt_extras``
        or ``stage_attachment``. Restart re-drives (``Redrive`` /
        ``session.restart_turn``) are the same logical turn and do NOT re-run
        starters. Starters run synchronously on the drive thread — keep them
        fast; a raising starter is logged and skipped.
        """
        def adapter(ctx: HookContext, _payload):
            starter(ctx.session)
        self._add_adapted(TURN_START, starter, adapter)

    def add_turn_finalizer(self, finalizer: TurnFinalizer) -> None:
        """Legacy alias: a ``turn_finish`` observer with the old ``(session)``
        signature. Finalizers run once per *logical* turn — a restart
        re-drive defers them to the drive that actually ends the turn."""
        def adapter(ctx: HookContext, _outcome):
            finalizer(ctx.session)
        self._add_adapted(TURN_FINISH, finalizer, adapter)

    def add_llm_selector(self, selector: LLMSelector) -> None:
        """Legacy alias: name a different LLM service for a session's turns.

        Registered two ways from the one function: kept on a side list for
        drive-time resolution (``select_llm``, consulted by
        ``runtime_config.active_llm``) and stood at the ``model_call`` doorway
        as an escort that rewrites ``request.llm`` per call — so a selector's
        choice holds even for calls issued mid-turn.
        """
        self._legacy_llm_selectors.append(selector)

        def adapter(ctx: HookContext, request: ModelRequest, proceed):
            try:
                name = selector(ctx.session, ctx.runtime)
            except Exception:
                logger.exception("LLM selector raised; treating as abstain")
                name = None
            if name and ctx.runtime is not None:
                svc = (getattr(ctx.runtime, "services", {}) or {}).get(name)
                if svc is not None:
                    request.llm = svc
            return proceed(request)
        self._add_adapted(MODEL_CALL, selector, adapter)

    def stage_attachment(self, session, attachment: Any) -> bool:
        """Queue one attachment for the next LLM call in this session."""
        key = getattr(session, "key", None)
        if not key:
            return False
        self._turn_attachments.setdefault(key, []).append(attachment)
        return True

    # ──────────────────────────────────────────────────────────────────
    # Consultation (called by the kernel at its doorways). The kernel-facing
    # signatures predate the uniform contract and are kept stable; each
    # method builds the HookContext envelope before knocking.
    # ──────────────────────────────────────────────────────────────────

    def _ctx(self, session, runtime, moment: str) -> HookContext:
        """Internal helper to build the envelope for one doorway visit."""
        return HookContext(session=session, runtime=runtime, moment=moment)

    def vet_permission(self, session, tool_name: str | None, command: str,
                       runtime=None) -> PermissionVerdict | None:
        """Return the first decisive verdict, or None if every gate abstains."""
        ctx = self._ctx(session, runtime, VET_PERMISSION)
        query = PermissionQuery(tool_name=tool_name, command=command)
        for gate in self._hooks[VET_PERMISSION]:
            try:
                verdict = gate(ctx, query)
            except Exception:
                logger.exception("Permission gate raised; treating as abstain")
                continue
            if verdict is not None:
                return verdict
        return None

    def select_llm(self, session, runtime) -> str | None:
        """Drive-time LLM resolution: first legacy selector's non-None name.

        Per-call resolution happens at the ``model_call`` doorway (the same
        selectors are registered there as escorts); this method exists so
        ``build_loop`` can still pick the drive's default brain up front.
        """
        for selector in self._legacy_llm_selectors:
            try:
                name = selector(session, runtime)
            except Exception:
                logger.exception("LLM selector raised; treating as abstain")
                continue
            if name:
                return name
        return None

    def shape_scope(self, session, registry, runtime=None):
        """Fold every shaper over the registry, in registration order."""
        ctx = self._ctx(session, runtime, SHAPE_SCOPE)
        for shaper in self._hooks[SHAPE_SCOPE]:
            try:
                shaped = shaper(ctx, registry)
            except Exception:
                logger.exception("Scope shaper raised; leaving registry unchanged")
                continue
            if shaped is not None:
                registry = shaped
        return registry

    def start_turn(self, session, runtime=None) -> None:
        """Run the ``turn_start`` adjusters (once per logical turn)."""
        ctx = self._ctx(session, runtime, TURN_START)
        for starter in self._hooks[TURN_START]:
            try:
                starter(ctx, None)
            except Exception:
                logger.exception("Turn starter raised; continuing")

    def finish_turn(self, session, outcome: TurnOutcome | None = None, runtime=None) -> None:
        """Run the ``turn_finish`` observers, then clear staged attachments."""
        ctx = self._ctx(session, runtime, TURN_FINISH)
        for finalizer in self._hooks[TURN_FINISH]:
            try:
                finalizer(ctx, outcome)
            except Exception:
                logger.exception("Turn finalizer raised; continuing")
        self._turn_attachments.pop(getattr(session, "key", None), None)

    def wrap_model_call(self, session, runtime, base: Callable[[ModelRequest], Any]) -> Callable[[ModelRequest], Any]:
        """Build the escort onion around one model call.

        ``base`` is the innermost step — the actual backend call. Each
        registered escort wraps the next; the first registered is outermost.
        A raising escort is skipped *transparently*: if it already fetched a
        response via ``proceed``, that response is used (never re-fetched);
        if it raised before dialing, the call proceeds as if the escort were
        not there.
        """
        ctx = self._ctx(session, runtime, MODEL_CALL)
        handler = base
        for fn in reversed(self._hooks[MODEL_CALL]):
            handler = self._escort_layer(ctx, fn, handler)
        return handler

    @staticmethod
    def _escort_layer(ctx: HookContext, fn: Callable, proceed: Callable) -> Callable:
        """Internal helper: one layer of the onion, with the skip-transparently
        error policy."""
        def layer(request: ModelRequest):
            last: dict[str, Any] = {"response": None, "called": False}

            def guarded_proceed(req: ModelRequest | None = None):
                last["response"] = proceed(req if req is not None else request)
                last["called"] = True
                return last["response"]

            try:
                out = fn(ctx, request, guarded_proceed)
            except Exception:
                logger.exception("Model-call escort raised; passing through")
                return last["response"] if last["called"] else proceed(request)
            if out is not None:
                return out
            # Escort abstained: use what it already fetched, or dial for it.
            return last["response"] if last["called"] else proceed(request)
        return layer

    def vet_end_turn(self, session, runtime, ending: TurnEnding):
        """Consult the doormen at the exit: first non-None verdict wins."""
        ctx = self._ctx(session, runtime, END_TURN)
        for doorman in self._hooks[END_TURN]:
            try:
                verdict = doorman(ctx, ending)
            except Exception:
                logger.exception("End-turn doorman raised; treating as abstain")
                continue
            if verdict is not None:
                return verdict
        return None

    def drain_attachments(self, session) -> list[Any]:
        """Return and clear attachments staged for the next model call."""
        return self._turn_attachments.pop(getattr(session, "key", None), [])
