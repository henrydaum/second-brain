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

#: Which arguments of a filesystem Request name a file, so the sink can lift
#: them somewhere nothing truncates.
#:
#: ``record_action`` caps ``args_json`` at ``LEDGER_JSON_CAP`` and past that
#: **replaces the object** with a ``head``/``tail`` wrapper — and the argument
#: that pushes a write over the cap is the file's own contents. So the rows
#: whose paths are hardest to recover are exactly the big edits somebody most
#: wants to see. Copying the paths into ``data`` costs a few bytes and makes
#: them unconditional; a reader never has to parse ``args_json`` at all.
FILE_ARGS = {
    "fs.write": ("path",),
    "fs.write_bytes": ("path",),
    "fs.delete": ("path",),
    "fs.move": ("src", "dst"),
}

#: Requests whose file effects are inside a command line rather than an
#: argument, so the paths have to be read out of it (``shell.files_touched``).
SHELL_REQUESTS = ("proc.run", "proc.start")


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


def identity_of(context) -> tuple:
    """Whose work an effect was, from the context that answered it.

    ``(session_key, conversation_id, user_id)``. Public because two callers ask
    it — the sandbox sink, about a Request the kernel serviced, and
    ``ledger.record``, about a note a plugin wrote itself. One spelling, so a
    plugin's own row lands in the same conversation as the rows written about
    it and answers to the same queries.

    Read from the **context** rather than from ``chain.root``, which is the
    other candidate and the wrong one. The root answers what *caused* the work,
    and ``policy.chain_session`` only recovers a session from it for
    agent-caused calls — a person's own action roots at ``user`` and names no
    session at all. The context is the kernel's own answer about whose call
    this is, and it is right in both directions that matter: ``frontend.act``
    moves chain and context together, while a resident service polling on its
    own initiative is handed ``kernel_context(None)`` and correctly records
    nothing, because a service poll belongs to no conversation.

    The conversation is resolved through the live session rather than stored on
    the context, because a session's conversation changes underneath it — a
    ``/new`` or a ``conv.load`` rebinds it — and the row should say which
    conversation the effect actually landed in.
    """
    key = getattr(context, "session_key", None)
    runtime = getattr(context, "runtime", None)
    sessions = getattr(runtime, "sessions", None) or {}
    session = sessions.get(key) if key else None
    return (key, getattr(session, "conversation_id", None),
            getattr(context, "user_id", None))


def sandbox_sink(db):
    """Build the ledger sink an :class:`~sandbox.interpreter.Interpreter` takes.

    Returns ``callable(chain, request, decision, result, context=None)``.
    Without one the sandbox records nothing at all, which left the flight
    recorder blind to every effect a plugin performed — the one part of the
    system where unattended operation most needs to be reconstructable after
    the fact.

    **A row says whose work it was**, via ``identity_of`` on the trailing
    context. It did not for a long time, and the omission was invisible because
    the columns existed and simply held NULL: ``action_ledger`` carries
    ``session_key``/``conversation_id``/``user_id`` and an index on
    ``(conversation_id, id)`` built for exactly this, and the sandbox origin —
    the highest-volume one, and the only per-effect record the system has — was
    the one origin that filled none of them. So ``get_ledger_rows(
    conversation_id=…)`` answered nothing for plugin effects and the index was
    dead weight, while ``my_action_ledger`` (which scopes on ``user_id``) hid
    every sandbox row from plugin code including its own. The enact sites had
    supplied all three since the beginning; this is the sandbox catching up
    rather than a new claim.

    **Reads are not recorded, effects and refusals always are.** A console
    frontend issues a ``console.read`` Request every poll; at the 50 ms default
    that is twenty rows a second, forever, and it would bury the rows worth
    reading under about 1.7 million a day of nothing happening. The
    ``READ_ONLY`` set already draws exactly this line and exists for exactly
    this question. Anything the kernel had to *ask* about, and anything it
    refused, is kept regardless of type — a denied read is a real event even
    though the read itself would not have been.

    ``llm.delta`` is the one *write* held to the same rule, and it is the
    worse case: a streaming backend sends one per token, so a single reply
    wrote thousands of rows, each an INSERT and a commit under the database's
    one lock — serializing the whole database against the model at fifty to a
    hundred round trips a second. It buys nothing either, since the call and
    the enact are already recorded and this is only the model's own output
    kept a character at a time. It is dropped here rather than added to
    ``READ_ONLY``, which says what it holds: things that read rather than
    change. What the sink drops is the sink's question.

    Built here rather than in the sandbox because it is the sandbox's *origin*
    on a kernel table; ``sandbox/`` stays ignorant of the database, and the
    composition root hands this in like it hands in the approver.
    """
    from sandbox import shell
    from sandbox.guest.requests import HTTP_PUSH, LLM_DELTA, READ_ONLY

    #: Reads, plus the writes too frequent to keep. ``http.push`` is the same
    #: token-by-token half of a stream one transport further out: an SSE
    #: frontend sends one per ``llm.delta``, so keeping it would restore the
    #: exact row-per-token flood dropping ``llm.delta`` was meant to end.
    #: ``respond`` and ``close`` are per-request rather than per-token and stay
    #: recorded — the sink's question is volume, not whether it is rendering.
    unrecorded = READ_ONLY | {LLM_DELTA, HTTP_PUSH}

    def record(chain, request, decision, result, context=None) -> None:
        """Append one serviced Request. Never raises; never blocks a turn."""
        write = getattr(db, "record_action", None)
        if write is None:
            return
        try:
            ok = bool(getattr(result, "ok", False))
            if request.type in unrecorded and ok and decision.safe:
                return
            session_key, conversation_id, user_id = identity_of(context)
            data = {"chain": chain.render(), "level": decision.level,
                    "reason": decision.reason}
            if (named := FILE_ARGS.get(request.type)) is not None:
                args = request.args or {}
                if paths := [args[k] for k in named if args.get(k)]:
                    data["paths"] = paths
                # The write's own answer, so a size does not cost a second
                # read. Absent on a refusal, which has no answer to give.
                written = getattr(result, "data", None)
                if ok and isinstance(written, dict) and "bytes" in written:
                    data["bytes"] = written["bytes"]
            elif request.type in SHELL_REQUESTS and ok:
                # Only a command that *succeeded*, and for ``proc.run`` only
                # one that exited cleanly: a failed ``rm`` deleted nothing, and
                # a row claiming otherwise is worse than a missing one.
                answer = getattr(result, "data", None) or {}
                if answer.get("code", 0) == 0:
                    paths, deleted = shell.files_touched(request.args or {})
                    if paths:
                        data["paths"] = paths
                        # Weaker than a path the kernel serviced: this was read
                        # out of a command line, so say where it came from
                        # rather than letting it pass as the same claim.
                        data["via"] = "shell"
                        data["command"] = shell.render_command(request.args or {})
                        if deleted:
                            data["deleted"] = deleted
            write(
                origin=SANDBOX_ORIGIN,
                action_type=request.type,
                ok=ok,
                session_key=session_key,
                conversation_id=conversation_id,
                user_id=user_id,
                name=chain.links[-1] if chain.links else chain.root,
                actor_id=chain.root,
                args=request.args,
                # The Result says why now. This used to ask ``result.denied``,
                # which asked whether the *message* began with "denied" — so
                # the ledger's two-value vocabulary was reverse-engineered
                # from prose. ``or "failed"`` keeps the old catch-all for an
                # uncoded failure, so existing rows stay comparable.
                error_code=None if ok else (
                    getattr(result, "code", "") or "failed"),
                error_message=getattr(result, "error", None) or None,
                data=data,
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
