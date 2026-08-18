"""Standing a sandboxed service at a kernel doorway.

The kernel's hook contract is ``fn(ctx, payload)`` over live objects; the
guest's is the same shape over plain data. This module is the two-way
translation between them, and it is deliberately the only place that knows
both vocabularies.

**Projection, not serialization.** A payload is not "the kernel object,
encoded" — several of them could not be, and pretending otherwise would leak
kernel internals into a contract plugins depend on. What crosses is what a
hook can act on: a turn's ending, a permission question, a toolbox's *names*.
The narrowing is the design, not a limitation to be engineered away later.

**Failure is abstention.** A denied Request, a dead box, a raising method, a
verdict the kernel does not recognise — all of them come back as ``None``,
which every doorway already understands as "no opinion". This is why a
sandboxed hook cannot break a turn: the worst it can do is fall silent.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, NamedTuple

from .guest.hooks import MOMENTS

logger = logging.getLogger("Sandbox")


# ──────────────────────────────────────────────────────────────────────
# The escort's phone.
#
# Every other Request resolves to a handler from a static table. ``proceed``
# cannot: it means "place *the* call this escort was handed", which is a
# closure the kernel built moments ago and will discard as soon as the escort
# returns. So the shim parks it here under a one-shot token, hands the token
# to the box, and the handler trades the token back for the closure.
#
# The token is what makes this safe. It is unguessable, it lives only for the
# duration of one doorway visit, and code holding no token cannot reach any
# call — so ``llm.proceed`` outside an ``llm_call`` hook resolves to
# nothing and is refused, which is exactly right.
# ──────────────────────────────────────────────────────────────────────

_PHONES: dict = {}
_PHONE_LOCK = threading.Lock()


def _park(proceed) -> str:
    """Park one ``proceed`` closure and return the token that reaches it."""
    token = uuid.uuid4().hex
    with _PHONE_LOCK:
        _PHONES[token] = proceed
    return token


def _unpark(token: str) -> None:
    """Discard a parked closure. Always called, even when the escort raised."""
    with _PHONE_LOCK:
        _PHONES.pop(token, None)


def phone(token: str):
    """The closure a token reaches, or None if it reaches nothing."""
    with _PHONE_LOCK:
        return _PHONES.get(token or "")


# ──────────────────────────────────────────────────────────────────────
# Kernel object -> data.
# ──────────────────────────────────────────────────────────────────────

def project_context(ctx, moment: str) -> dict:
    """What identifies this turn, for a hook that cannot hold the session."""
    session = getattr(ctx, "session", None)
    runtime = getattr(ctx, "runtime", None)
    key = str(getattr(session, "key", "") or "")

    attended = True
    reader = getattr(runtime, "is_attended", None)
    if callable(reader) and key:
        try:
            attended = bool(reader(key))
        except Exception:
            attended = True

    return {
        "moment": moment,
        "session_key": key,
        "user_id": int(getattr(session, "user_id", None) or 0),
        "conversation_id": int(getattr(session, "conversation_id", None) or 0),
        "attended": attended,
    }


def _project_request(request) -> dict:
    """A sandbox Request as data, for a gate deciding about one."""
    if request is None:
        return {}
    return {"type": getattr(request, "type", ""),
            "args": dict(getattr(request, "args", None) or {})}


def _project_chain(chain) -> dict:
    """The provenance chain as data. A hook may read it but never set it."""
    if chain is None:
        return {}
    return {"root": str(getattr(chain, "root", "") or ""),
            "links": [str(link) for link in getattr(chain, "links", ()) or ()],
            "rendered": chain.render() if hasattr(chain, "render") else ""}


def project_payload(moment: str, payload) -> dict | None:
    """One moment's payload, narrowed to what can cross and be acted on."""
    if moment == "turn_start":
        # The kernel passes nothing here; the session identity in the envelope
        # is the whole input, and effects go back through sdk.session.*.
        return None

    if moment == "turn_finish":
        return {"ok": bool(getattr(payload, "ok", True)),
                "cancelled": bool(getattr(payload, "cancelled", False)),
                "final_text": str(getattr(payload, "final_text", "") or ""),
                "reason": str(getattr(payload, "reason", "") or "")}

    if moment == "end_turn":
        return {"final_text": str(getattr(payload, "final_text", "") or ""),
                "reason": str(getattr(payload, "reason", "") or ""),
                "doorman_fires": int(getattr(payload, "doorman_fires", 0) or 0)}

    if moment == "vet_permission":
        return {"tool_name": str(getattr(payload, "tool_name", "") or ""),
                "command": str(getattr(payload, "command", "") or ""),
                "stage": str(getattr(payload, "stage", "") or ""),
                "origin": str(getattr(payload, "origin", "") or ""),
                "request": _project_request(getattr(payload, "request", None)),
                "chain": _project_chain(getattr(payload, "chain", None))}

    if moment == "shape_scope":
        # A registry cannot cross, and a shaper does not need one: hiding is
        # expressible in names alone. Sorting here (and the set ``narrow_scope``
        # keeps) is what makes a shaper a filter and nothing more — it cannot
        # reorder, and adding a tool is a different act with its own Request
        # (sdk.session.add_tool).
        return {"tools": sorted(getattr(payload, "tools", None) or {})}

    if moment == "llm_call":
        return project_model_request(payload)

    return None


def project_model_request(request) -> dict:
    """One outgoing model call as data.

    ``llm`` is the profile's *name*. A live backend could not cross, and a name
    is the better contract anyway: it is the same handle-not-the-thing move as
    ``<secret:...>``, and swapping brains stays one assignment.

    It is read straight off the request because ``ModelRequest.llm`` already
    *is* a name — see ``runtime/hooks.py``. This used to reach for
    ``.model_name`` on it, which is an attribute a string does not have, so
    every escort ever built was shown an empty model name. That dated from
    before the LLM became kernel routing, when ``llm`` really did hold a
    service object.
    """
    return {
        "llm": str(getattr(request, "llm", "") or ""),
        "messages": list(getattr(request, "messages", None) or []),
        "tools": list(getattr(request, "tools", None) or []),
        "tool_choice": getattr(request, "tool_choice", None),
        "params": dict(getattr(request, "params", None) or {}),
    }


def project_model_response(response) -> dict:
    """What the model said, as data."""
    return {
        "content": str(getattr(response, "content", "") or ""),
        "tool_calls": list(getattr(response, "tool_calls", None) or []),
        "error": str(getattr(response, "error", "") or ""),
    }


def apply_model_request(request, changed: dict, runtime):
    """Fold an escort's rewrite back onto the live ModelRequest.

    Only the fields an escort is allowed to touch, and ``llm`` only when the
    name matches a configured profile — an escort naming a brain that does not
    exist should leave the call alone rather than silently retarget it.

    What is assigned is the **name**, not a brain. ``ModelRequest.llm`` is a
    handle the loop resolves when it places the call
    (``ConversationLoop._brain``), and putting an object there would work only
    by accident. This looked profiles up in ``runtime.services``, which is
    where backends lived before they became ``llm/`` routing — so it found
    nothing, warned, and no sandboxed escort could ever swap a model.
    """
    if not isinstance(changed, dict):
        return request

    for field in ("messages", "tools", "tool_choice", "params"):
        if field in changed and changed[field] is not None:
            setattr(request, field, changed[field])

    if "llm" not in changed:
        return request
    wanted = str(changed.get("llm") or "")
    current = str(getattr(request, "llm", "") or "")
    if wanted == current:
        return request
    if not wanted:
        # Clearing the name is meaningful: the loop reads empty as "this
        # session's default", so an escort can hand the call back that way.
        request.llm = ""
        return request
    try:
        import llm as llm_registry

        known = llm_registry.brain(wanted) is not None
    except Exception:
        logger.exception("could not check whether brain %r exists", wanted)
        return request
    if known:
        request.llm = wanted
    else:
        logger.warning("escort asked for brain %r, which is not a configured "
                       "profile; keeping %r", wanted, current or "the default")
    return request


# ──────────────────────────────────────────────────────────────────────
# Data -> kernel object.
# ──────────────────────────────────────────────────────────────────────

def _verdict_classes():
    """The kernel's verdict types, imported late to keep this module light."""
    from runtime.hooks import (Allow, PermissionVerdict, Redrive, RequireTool,
                               SendBack)
    return {"allow": Allow, "send_back": SendBack, "require_tool": RequireTool,
            "redrive": Redrive, "permission": PermissionVerdict}


def rebuild(moment: str, answer):
    """Turn a hook's data answer back into what the doorway expects.

    Returns ``None`` for anything unrecognised, because a doorway's contract
    already says that means "no opinion" — and inventing a verdict from a
    malformed answer would be worse than hearing nothing.
    """
    if answer is None:
        return None

    if moment == "shape_scope":
        # Handled by the caller, which owns the registry being narrowed.
        return answer

    if not isinstance(answer, dict):
        logger.warning("%s hook answered with %s, not a verdict",
                       moment, type(answer).__name__)
        return None

    kind = answer.get("verdict")
    classes = _verdict_classes()
    cls = classes.get(kind)
    if cls is None:
        logger.warning("%s hook answered with unknown verdict %r", moment, kind)
        return None

    if kind == "permission":
        return cls(bool(answer.get("allow")), str(answer.get("reason") or ""))
    if kind == "send_back":
        return cls(note=str(answer.get("note") or ""),
                   ephemeral=bool(answer.get("ephemeral")),
                   allow_tools=bool(answer.get("allow_tools", True)))
    if kind == "require_tool":
        return cls(name=str(answer.get("name") or ""),
                   note=str(answer.get("note") or ""))
    return cls()


def narrow_scope(registry, keep):
    """Apply a shaper's answer to the registry it was shown.

    A shaper returns names to keep. Anything it invents is ignored: narrowing
    is safe and widening is not, so the answer is intersected with what was
    already there rather than trusted.
    """
    if not isinstance(keep, (list, tuple, set)):
        return registry
    available = set(getattr(registry, "tools", None) or {})
    kept = available & {str(name) for name in keep}
    visible = getattr(registry, "visible_tool_names", None)
    registry.visible_tool_names = kept if visible is None else set(visible) & kept
    return registry


# ──────────────────────────────────────────────────────────────────────
# The shim.
#
# One walk from a kernel doorway into guest code, written once. It used to be
# written twice — ``build_shim`` for five moments and ``_build_escort`` for
# ``llm_call`` — and the two drifted, in the direction this whole module is
# built to prevent.
#
# What drifted: ``for_session`` was added to the generic path so a hook's own
# Requests would resolve against the session whose doorway it stands at, and
# the escort did not get it. A boxed escort touching ``sdk.session.*`` reached
# the kernel's default session instead. Nothing raised, the write returned
# True, and the text landed nowhere — the identical failure the ``for_session``
# argument had been introduced to fix one function earlier.
#
# The tell was there all along: the *guest* has one entry point
# (``BasePlugin.__hook__``, one signature, ``token`` defaulting to empty), so
# two host functions producing that one call was never describing a real
# difference. What actually differs between the six moments is three things —
# the fallback when the box is gone, the payload builder, and the token — and
# those are now arguments rather than a second copy of the walk.
# ──────────────────────────────────────────────────────────────────────

class _Visit(NamedTuple):
    """What came back from walking to a doorway.

    ``ok`` is False for *any* reason the guest did not answer — the service is
    registered but not loaded, its box has died, the method raised inside the
    box. Those are one case to the kernel and a different case to each caller:
    a generic hook abstains, while an escort still owes somebody a model call.
    So this reports the fact and lets the caller decide, rather than baking one
    of the two answers into the walk.
    """

    ok: bool
    data: Any = None


def _visit(service, moment: str, method: str, ctx, *, payload,
           extra: dict | None = None) -> _Visit:
    """Walk one doorway visit into a service's box.

    The single place that knows the route: find a live box, project the
    identity, lend the session, call ``__hook__``, and turn every failure into
    something the caller can read as abstention.

    ``extra`` is for what one moment needs and the others do not — today only
    the escort's ``token``. Passing it as a mapping rather than a named
    parameter is deliberate: a second copy of this function is exactly what
    went wrong before, and a moment that needs one more field should widen this
    dict rather than fork the walk.
    """
    box = _live_box(service)
    if box is None:
        # Registered but not loaded: abstain rather than fail. A service can be
        # unloaded and reloaded under a hook that stays standing.
        return _Visit(False)

    # ``handler``, not ``method``: PersistentBox.call names its own first
    # parameter ``method``, and passing one by keyword collides with it.
    #
    # ``for_session`` lends the box the session this doorway was opened for.
    # The projection below has always told the *guest* which session it was
    # standing in; nothing told the guest's *Requests*, so a hook could read
    # ``sdk`` fields naming a session and then have every session-scoped
    # Request answered from the kernel's default one — which is how
    # ``session.add_prompt_extra`` came to write into ``sessions.get(None)``
    # and return False, silently, for every turn. Context only: see
    # ``PersistentBox.call`` for why the chain deliberately does not move.
    projected = project_context(ctx, moment)
    result = box.call("__hook__", moment=moment, handler=method,
                      for_session=projected.get("session_key") or "",
                      ctx=projected, payload=payload, **(extra or {}))
    if not result.ok:
        logger.warning("%s hook %s.%s: %s", moment,
                       getattr(service, "name", "?"), method, result.error)
        return _Visit(False)
    return _Visit(True, result.data)


def build_shim(service, moment: str, method: str, make_response=None):
    """Build the callable that stands at ``moment`` on a service's behalf.

    The kernel sees an ordinary hook. Behind it, one call into the service's
    box — which is why a hook belongs to a *service*: something has to be
    resident for the kernel to call into.

    ``make_response`` builds an ``LLMResponse`` for an escort that answered
    without dialing. It is injected rather than imported because that type
    lives in ``plugins.*``, and the bridge is the only part of the sandbox
    allowed to reach across that line.
    """
    if moment not in MOMENTS:
        raise ValueError(f"unknown hook moment: {moment!r}")

    if moment == "llm_call":
        return _build_escort(service, method, make_response)

    def shim(ctx, payload, *rest):
        """One doorway visit, forwarded into the box.

        Five of the six moments answer with data and nothing else, so the
        whole of this is the walk plus one translation on the way back.
        """
        visit = _visit(service, moment, method, ctx,
                       payload=project_payload(moment, payload))
        if not visit.ok:
            return None
        if moment == "shape_scope":
            return narrow_scope(payload, visit.data)
        return rebuild(moment, visit.data)

    shim.__name__ = f"sandboxed_{moment}_{method}"
    shim.__doc__ = (f"{getattr(service, 'name', '?')}.{method} standing at "
                    f"the {moment} doorway.")
    return shim


def _live_box(service):
    """The service's box, if it is open and able to take a call."""
    box = getattr(service, "_sandbox_box", None)
    return box if box is not None and box.alive else None


def _build_escort(service, method: str, make_response=None):
    """Build the ``llm_call`` escort, which holds a phone as well as a request.

    The escort is the one doorway where the hook decides *when* the kernel
    acts, not just whether. That means a callback going the other way, and a
    callback is the one thing a Request cannot be — so the closure stays
    kernel-side and the box gets a token that reaches it.

    Abstention stays transparent, exactly as it is for a native escort: if the
    box already placed the call, that response is used rather than fetched
    again; if it never dialed, the kernel dials for it.
    """
    def escort(ctx, request, proceed):
        """One model call, escorted through the box."""
        runtime = getattr(ctx, "runtime", None)
        placed = {"response": None, "called": False}

        def dial(changed: dict | None = None):
            """What ``sdk.llm.proceed`` reaches. Rewrites, then calls."""
            apply_model_request(request, changed or {}, runtime)
            placed["response"] = proceed(request)
            placed["called"] = True
            return project_model_response(placed["response"])

        def settled():
            """The response this call has already produced, or place it now.

            Abstention stays transparent exactly as it is for a native escort:
            a call the box already placed is never placed twice, and a box
            that never dialed still owes the kernel its round trip. Every way
            of *not* answering — a dead box, a raise inside the guest, a
            literal ``None`` — lands here.
            """
            return placed["response"] if placed["called"] else proceed(request)

        token = _park(dial)
        try:
            visit = _visit(service, "llm_call", method, ctx,
                           payload=project_model_request(request),
                           extra={"token": token})
        finally:
            # The phone is disconnected the moment the escort steps away, so a
            # token that leaked cannot be used to place a call later.
            _unpark(token)

        if not visit.ok:
            return settled()

        answer = visit.data
        if isinstance(answer, dict) and answer.get("content") is not None:
            # The escort answered for itself. If it never dialed, the model was
            # never troubled — which is a legitimate thing for an escort to do
            # (a cache hit, a canned refusal).
            if placed["called"]:
                response = placed["response"]
                response.content = str(answer.get("content") or "")
                # ``tool_calls`` only when the answer carries the key. An
                # escort that round-trips what it was given writes back what
                # was already there, and one that edits them means it — while
                # an answer that never mentions them leaves the model's own
                # intent alone.
                #
                # This branch used to copy ``content`` and stop, which made the
                # two halves of one contract disagree: ``bridge._make_response``
                # (the never-dialed path, just below) has always carried
                # ``tool_calls`` and ``error`` too. So whether an escort could
                # shape what the model wanted to *do* depended on whether it
                # had placed the call — an asymmetry nobody chose.
                if "tool_calls" in answer:
                    response.tool_calls = list(answer.get("tool_calls") or [])
                return response
            if make_response is not None:
                return make_response(answer)
            logger.warning("escort %s answered without placing the call, but "
                           "there is no way to build a response", method)

        return settled()

    escort.__name__ = f"sandboxed_llm_call_{method}"
    escort.__doc__ = (f"{getattr(service, 'name', '?')}.{method} escorting the "
                      f"model call.")
    return escort


