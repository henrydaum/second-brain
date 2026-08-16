"""The doorways, as sandboxed code sees them.

Every agent turn is the same short ritual, and the kernel puts a labeled
doorway at every moment of it (``runtime/hooks.py`` is the kernel's side).
This is the guest's side: the same six moments, with payloads that are plain
data rather than live kernel objects.

**A hook is inbound.** Everything else in the SDK is your code asking the
kernel for something; a hook is the kernel calling you. That inversion is why
hooks are declared rather than registered — a service names them in a class
attribute:

    class Doorman(BaseService):
        name = "doorman"
        hooks = {"end_turn": "check_done"}

The kernel reads that without importing the file, stands a shim at the named
doorway, and takes it away when the service unloads. There is nothing to
register and therefore nothing to leak.

**What crosses, and what does not.** A payload is a projection, not the kernel
object. ``TurnEnding`` is the whole truth; ``PermissionQuery.request`` is a
Request rendered as data; and the tool registry is projected to *names*, so a
scope shaper can hide a tool but cannot synthesize one (use
``sdk.session.add_tool`` for that). It cannot reorder either — the names
arrive sorted and what it returns is kept as a set — so a shaper is a filter
and nothing else.

**Abstain by returning None.** It is the default and it composes: a hook that
speaks only when it must is one that works alongside every other plugin. A
hook that raises is logged and skipped, so it can never break a turn.

**Hooks run on the drive thread**, synchronously, inside every turn they touch.
Keep them fast — each one adds a box round trip to the latency of every reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

TURN_START = "turn_start"
SHAPE_SCOPE = "shape_scope"
VET_PERMISSION = "vet_permission"
LLM_CALL = "llm_call"
END_TURN = "end_turn"
TURN_FINISH = "turn_finish"

MOMENTS = (TURN_START, SHAPE_SCOPE, VET_PERMISSION, LLM_CALL, END_TURN,
           TURN_FINISH)


# ──────────────────────────────────────────────────────────────────────
# The envelope and the payloads.
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HookContext:
    """Whose turn this is, and which doorway you are standing at.

    The kernel's own ``HookContext`` carries the live session and runtime.
    Neither can cross, so this carries what identifies them instead —
    everything else is reachable through ``sdk``.
    """

    moment: str = ""
    session_key: str = ""
    user_id: int = 0
    conversation_id: int = 0
    attended: bool = True


@dataclass
class TurnEnding:
    """What the doorman at the exit is shown.

    ``reason`` is ``"model_finished"`` (the model produced final text) or
    ``"budget_exhausted"`` (the loop ran out of tool budget).
    ``doorman_fires`` is how many times a doorman has already intervened this
    turn — check it and abstain once satisfied, because the kernel stops
    consulting doormen past a fixed cap and the agent always gets to leave.
    """

    final_text: str = ""
    reason: str = "model_finished"
    doorman_fires: int = 0


@dataclass
class TurnOutcome:
    """What the ``turn_finish`` observers see once the turn is over.

    ``reason`` says how it ended, and matters because ``end_turn`` is only
    consulted on two of the nine ways a turn can end — a cancel, a priority
    handoff and a failed action all walk past the doorman. This doorway fires
    on every one of them, so it is where a doorman finds out what happened
    when it was not asked.

    Values: ``"model_finished"``, ``"budget_exhausted"`` (both shared with
    :class:`TurnEnding`), ``"cancelled"``, ``"priority_handoff"``,
    ``"action_failed"``, ``"no_action"``, ``"crashed"``, ``"redrive"``. Empty
    means the kernel did not say.
    """

    ok: bool = True
    cancelled: bool = False
    final_text: str = ""
    reason: str = ""


@dataclass
class PermissionQuery:
    """The question a ``vet_permission`` hook answers.

    ``stage`` is ``"approval"`` (something sensitive wants to happen;
    abstaining falls through to the user) or ``"unattended_call"`` (it was
    asked for with nobody present; abstaining means refuse).

    ``origin`` is ``"tool"`` for a tool call, or ``"request"`` for sandboxed
    code whose Request was classified unsafe — in which case ``request`` and
    ``chain`` describe what and who.
    """

    tool_name: str = ""
    command: str = ""
    stage: str = "approval"
    origin: str = "tool"
    request: dict = field(default_factory=dict)
    chain: dict = field(default_factory=dict)


@dataclass
class Scope:
    """The toolbox the agent is about to be shown.

    Return the names to keep — a subset. Order is not preserved and names you
    invent are ignored: a shaper narrows the toolbox and can do nothing else
    to it.

    **You are asked more than once per turn, and sometimes outside one.** The
    tool list is rebuilt for the state machine's specs, for the loop, and for
    every model call, so a long turn asks repeatedly; loading a conversation
    asks too, before any turn exists — and there ``ctx.attended`` is ``False``,
    because no session is active yet. Answer from what you are given rather
    than from a count of how often you have been asked.
    """

    tools: list = field(default_factory=list)


@dataclass
class ModelRequest:
    """One outgoing trip to the model, materialized so an escort can rewrite it.

    ``llm`` is the *name* of the brain that will take the call, not the
    backend object — swap it for another installed name to change brains for
    this call only. ``messages`` is exactly what it will be shown.
    """

    llm: str = ""
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    tool_choice: object = None
    params: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    """What came back. ``tool_calls`` is populated when the model wants to act."""

    content: str = ""
    tool_calls: list = field(default_factory=list)
    error: str = ""

    @property
    def has_tool_calls(self) -> bool:
        """Whether the model asked to call something."""
        return bool(self.tool_calls)


# ──────────────────────────────────────────────────────────────────────
# Verdicts — what a hook says back.
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PermissionVerdict:
    """A ``vet_permission`` answer. Abstain by returning None instead.

    **Deny beats allow.** Every gate is asked and any refusal wins, however
    late it is registered — so answering ``allow=True`` does not settle the
    question, it only declines to block it. This is the one doorway that does
    not work first-answer-wins, because under that rule a permissive gate
    loaded ahead of a restrictive one decided policy by filename order.

    So answer ``allow=False`` when you mean it and ``None`` when you have no
    opinion; do not use ``allow=True`` to try to overrule another plugin,
    because it cannot.
    """

    allow: bool = False
    reason: str = ""


@dataclass
class Allow:
    """Let the agent leave, and stop consulting later doormen."""


@dataclass
class SendBack:
    """Send the agent back inside with a note.

    ``ephemeral`` shows the note to the model without recording it in history;
    ``allow_tools=False`` makes the comeback call text-only.
    """

    note: str
    ephemeral: bool = False
    allow_tools: bool = True


@dataclass
class RequireTool:
    """Demand one specific tool before the agent may leave."""

    name: str
    note: str = ""


@dataclass
class Redrive:
    """End this drive without ending the turn; the runtime re-drives it."""


# ──────────────────────────────────────────────────────────────────────
# Crossing the boundary. Called by the base class, not by plugin authors.
# ──────────────────────────────────────────────────────────────────────

PAYLOADS = {
    TURN_START: None,
    SHAPE_SCOPE: Scope,
    VET_PERMISSION: PermissionQuery,
    LLM_CALL: ModelRequest,
    END_TURN: TurnEnding,
    TURN_FINISH: TurnOutcome,
}

_VERDICT_KINDS = {
    "allow": Allow, "send_back": SendBack, "require_tool": RequireTool,
    "redrive": Redrive, "permission": PermissionVerdict,
}
_VERDICT_NAMES = {cls: kind for kind, cls in _VERDICT_KINDS.items()}


def _fields(cls) -> set:
    """The field names a payload dataclass accepts."""
    return set(getattr(cls, "__dataclass_fields__", {}))


def wrap(moment: str, payload):
    """Rebuild a payload dataclass from the data the kernel sent.

    Unknown keys are dropped rather than raising: a kernel that grows a field
    should not break a hook written against the older shape.
    """
    cls = PAYLOADS.get(moment)
    if cls is None or payload is None:
        return None
    if isinstance(payload, cls):
        return payload
    allowed = _fields(cls)
    return cls(**{k: v for k, v in dict(payload).items() if k in allowed})


def unwrap(value):
    """Render a hook's answer as data the kernel can read.

    ``None`` stays ``None`` — abstaining is the one answer with no shape.
    """
    if value is None:
        return None
    kind = _VERDICT_NAMES.get(type(value))
    if kind is not None:
        data = {k: getattr(value, k) for k in _fields(type(value))}
        return {"verdict": kind, **data}
    if isinstance(value, (ModelRequest, ModelResponse)):
        return {k: getattr(value, k) for k in _fields(type(value))}
    # A scope shaper answers with a plain list of names, and an observer
    # answers with nothing worth carrying.
    return value


__all__ = ["MOMENTS", "TURN_START", "SHAPE_SCOPE", "VET_PERMISSION",
           "LLM_CALL", "END_TURN", "TURN_FINISH", "HookContext",
           "TurnEnding", "TurnOutcome", "PermissionQuery", "Scope",
           "ModelRequest", "ModelResponse", "PermissionVerdict", "Allow",
           "SendBack", "RequireTool", "Redrive", "wrap", "unwrap"]
