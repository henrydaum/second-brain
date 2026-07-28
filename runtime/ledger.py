"""Helpers that feed the action ledger from the runtime's chokepoints.

The ledger is the kernel's flight recorder: every action that flows through
the two labeled ``cs.enact(...)`` sites is appended to the ``action_ledger``
table, alongside ``origin="system"`` rows for acts that happen outside the
state machine (package installs, config saves, conversation lifecycle ops).

Everything here is best-effort twice over: ``db.record_action`` already
swallows its own failures, and these helpers additionally tolerate a missing
or stubbed ``db`` (unit tests run the runtime with ``db=None`` or fakes), so
a chokepoint stays one readable call with no try/except at the call site.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Ledger")

#: Its own origin rather than ``system``, because the useful ledger question
#: is "what did plugin code do", and the guidance is to read the table
#: targeted rather than linearly. Folding these in with config saves and
#: package installs would make the one high-volume origin indistinguishable
#: from the ones you go looking for.
SANDBOX_ORIGIN = "sandbox"


def _args_of(content: Any) -> Any:
    """Ledger-facing view of an action's content. Private plumbing keys
    (``_tool_call_id``, ``_assistant_text``) ride call/tool content dicts but
    are recorded in their own columns or not at all."""
    if isinstance(content, dict):
        return {k: v for k, v in content.items() if not k.startswith("_")}
    return content


def _call_id_of(content: Any, result: Any) -> str | None:
    data = getattr(result, "data", None) or {}
    if data.get("call_id"):
        return data["call_id"]
    if isinstance(content, dict):
        return content.get("_tool_call_id")
    return None


def record_enact(db, *, origin: str, session_key: str | None,
                 conversation_id: int | None, user_id: int | None,
                 actor_id: str | None, action_type: str, content: Any,
                 result: Any = None, error_message: str | None = None,
                 duration_ms: int | None = None, data: Any = None) -> None:
    """Append one ``cs.enact()`` outcome to the ledger.

    Pass ``result`` for a completed enact (its ok/error are recorded), or
    ``error_message`` alone when the enact itself raised.
    """
    record = getattr(db, "record_action", None)
    if record is None:
        return
    try:
        if result is not None:
            ok = bool(getattr(result, "ok", False))
            err = getattr(result, "error", None)
            error_code = getattr(err, "code", None)
            error_message = getattr(err, "message", None)
        else:
            ok, error_code = False, "exception"
        record(
            origin=origin, action_type=action_type, ok=ok,
            session_key=session_key, conversation_id=conversation_id,
            user_id=user_id, actor_id=actor_id,
            name=content.get("name") if isinstance(content, dict) else None,
            args=_args_of(content),
            error_code=error_code, error_message=error_message,
            call_id=_call_id_of(content, result),
            duration_ms=duration_ms, data=data,
        )
    except Exception as e:
        logger.warning(f"Ledger enact record failed (ignored): {e}")


def sandbox_sink(db):
    """Build the ledger sink an :class:`~sandbox.interpreter.Interpreter` takes.

    Returns ``callable(chain, request, decision, result)``. Without one the
    sandbox records nothing at all, which left the flight recorder blind to
    every effect a plugin performed — the one part of the system where
    unattended operation most needs to be reconstructable after the fact.

    **Reads are not recorded, effects and refusals always are.** A console
    frontend issues a ``console.read`` Request every poll; at the 50 ms default
    that is twenty rows a second, forever, and it would bury the rows worth
    reading under about 1.7 million a day of nothing happening. The
    ``READ_ONLY`` set already draws exactly this line and exists for exactly
    this question. Anything the kernel had to *ask* about, and anything it
    refused, is kept regardless of type — a denied read is a real event even
    though the read itself would not have been.

    Built here rather than in the sandbox because it is the sandbox's *origin*
    on a kernel table; ``sandbox/`` stays ignorant of the database, and the
    composition root hands this in like it hands in the approver.
    """
    from sandbox.guest.requests import READ_ONLY

    def record(chain, request, decision, result) -> None:
        """Append one serviced Request. Never raises; never blocks a turn."""
        write = getattr(db, "record_action", None)
        if write is None:
            return
        try:
            ok = bool(getattr(result, "ok", False))
            refused = bool(getattr(result, "denied", False))
            if request.type in READ_ONLY and ok and decision.safe:
                return
            write(
                origin=SANDBOX_ORIGIN,
                action_type=request.type,
                ok=ok,
                name=chain.links[-1] if chain.links else chain.root,
                actor_id=chain.root,
                args=request.args,
                error_code=("denied" if refused
                            else None if ok else "failed"),
                error_message=getattr(result, "error", None) or None,
                data={"chain": chain.render(), "level": decision.level,
                      "reason": decision.reason},
            )
        except Exception as exc:
            logger.warning(f"Sandbox ledger record failed (ignored): {exc}")

    return record


def record_system(db, *, action_type: str, ok: bool, session_key: str | None = None,
                  conversation_id: int | None = None, user_id: int | None = None,
                  actor_id: str | None = None, name: str | None = None,
                  args: Any = None, data: Any = None, error_code: str | None = None,
                  error_message: str | None = None) -> None:
    """Append one ``origin="system"`` row for an act outside the state machine."""
    record = getattr(db, "record_action", None)
    if record is None:
        return
    try:
        record(
            origin="system", action_type=action_type, ok=ok,
            session_key=session_key, conversation_id=conversation_id,
            user_id=user_id, actor_id=actor_id, name=name, args=args,
            data=data, error_code=error_code, error_message=error_message,
        )
    except Exception as e:
        logger.warning(f"Ledger system record failed (ignored): {e}")
