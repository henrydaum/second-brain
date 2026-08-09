"""Kernel-facing handlers — the Requests that need ``SecondBrainContext``.

Everything here reaches into the context object the kernel supplies: the
database, the conversation runtime, config, services, the tool and command
registries. The guest never touches any of it.

Two conventions run through the file:

- **Ownership is checked, not assumed.** Conversation Requests go through
  ``runtime.assert_conversation_access`` wherever it exists, mirroring the
  kernel's own rule that listing filters are convenience and access checks
  are the real boundary.
- **A missing capability is an ordinary failure.** The kernel is a microkernel;
  the timekeeper, the parser, or a tool registry may simply not be installed.
  Sandboxed code gets a Result saying so, not an exception.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from ..guest.requests import (AGENT_COLLECT, AGENT_COMPLETE, AGENT_SCHEDULE,
                              AGENT_SPAWN, AGENT_STOP,
                              APP_STOP, COMMAND_CALL,
                              COMMAND_LIST,
                              LLM_DELTA, LLM_LIST, LLM_LOAD, LLM_PROCEED,
                              LLM_UNLOAD,
                              CONFIG_READ, CONFIG_WRITE, CONV_APPEND, CONV_CLEAR,
                              CONV_CREATE, CONV_DELETE, CONV_LIST, CONV_READ,
                              CONV_LOAD, CONV_SET_CATEGORY,
                              CONV_SET_NOTIFICATION_MODE, CONV_SET_TITLE,
                              CRON_CREATE,
                              CRON_ENABLE, CRON_GET, CRON_LIST, CRON_REMOVE,
                              CONSOLE_READ, CONSOLE_WRITE,
                              CRON_UPDATE, DB_DEFINE, DB_QUERY, DB_WRITE,
                              EVENT_EMIT, EVENT_REQUEST, FILE_LIST,
                              FILE_REGISTER, FRONTEND_ACT, FRONTEND_ATTEND,
                              FRONTEND_BIND, FRONTEND_CANCEL,
                              FRONTEND_COLLECT, FRONTEND_PENDING,
                              FRONTEND_RESOLVE,
                              FRONTEND_SUBMIT, HTTP_CLOSE, HTTP_DRAIN,
                              HTTP_PUSH, HTTP_RESPOND,
                              LEDGER_READ, LEDGER_RECORD,
                              PARSE_FILE, PARSE_MODALITY, PATH_GET, PLUGIN_DESCRIBE,
                              PLUGIN_INSTALL, PLUGIN_LIST, PLUGIN_UNINSTALL,
                              PLUGIN_REGISTER, PLUGIN_RELOAD,
                              PLUGIN_UNREGISTER, PLUGIN_UPDATE, PLUGIN_VALIDATE,
                              SERVICE_CALL, SERVICE_LIST,
                              SERVICE_LOAD, SERVICE_UNLOAD,
                              SESSION_ADD_ATTACHMENT,
                              SESSION_ADD_PROMPT, SESSION_ADD_TOOL,
                              SESSION_CANCEL, SESSION_GET, SESSION_LIST,
                              SESSION_PUSH, SESSION_REMOVE_PROMPT,
                              SESSION_REMOVE_TOOL, SESSION_SET_MODE,
                              SESSION_STATE_GET,
                              SESSION_STATE_SET,
                              SCRIPT_COLLECT, SCRIPT_RUN, SCRIPT_STOP,
                              SELF_BUDGET,
                              TASK_ENQUEUE, TASK_GRAPH,
                              TASK_LIST, TASK_OUTPUT, TASK_PAUSE, TASK_RESET,
                              TASK_STATUS, TASK_TRIGGER, TOOL_CALL, TOOL_LIST,
                              UI_APPROVE,
                              UI_ASK, UI_RENDER, USER_LIST, USER_READ,
                              USER_WRITE, ALL_TYPES, Request, Result)
from ..guest.codes import (ERROR_INVALID_ARGUMENT, ERROR_NOT_FOUND,
                          ERROR_NOT_PERMITTED, ERROR_UNAVAILABLE)
from .args import float_arg, int_arg
from ..credentials import lookup_from, redact, redact_nested, resolve
from ..events import publish as _publish_event
from ..protected import reason_for
from ..users import ScopeError, scope_sql, scope_write

# Two ``except`` paths here already logged and neither could: the name was
# never bound, so the fallback for "could not resolve the active LLM" and for
# a failed background submit raised NameError on top of whatever it was
# reporting. Same sink as the rest of the sandbox.
logger = logging.getLogger("Sandbox")

# Never returned by any Request, at any level.
HIDDEN_USER_COLUMNS = {"password_hash"}


def _need(value, what: str):
    """Return a Result explaining an absent capability, or None.

    Callers must compare against None. A failure Result is *falsy* by design
    — that is the whole point of the return contract — so ``if (bad := _need(
    ...)):`` silently does nothing, which is the opposite of a guard.
    """
    if value is None:
        return Result.failure(f"{what} is not available in this kernel",
                              code=ERROR_UNAVAILABLE)
    return None


def _db(ctx):
    """The database, or None."""
    return getattr(ctx, "db", None)


def _runtime(ctx):
    """The conversation runtime, or None."""
    return getattr(ctx, "runtime", None)


def _service(ctx, name: str):
    """A loaded service by name, or None."""
    return (getattr(ctx, "services", None) or {}).get(name)


def _rows(value):
    """Normalize sqlite rows into plain dicts, which is all that may cross."""
    if value is None:
        return []
    return [dict(row) for row in value]


# ──────────────────────────────────────────────────────────────────────
# Database.
# ──────────────────────────────────────────────────────────────────────

# The answer crosses a process boundary as JSON, so an unbounded SELECT is a
# hazard rather than a result. A caller that gets exactly this many rows back
# knows it was capped and can add its own LIMIT.
DB_MAX_ROWS = 500


def _db_query(ctx, args: dict) -> Result:
    """Read rows.

    Reads stay deliberately broad — a plugin that reads everything still
    cannot send anything anywhere, because egress is gated. What is narrowed
    is *whose* rows: user-scoped tables are rewritten to the current user.
    """
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    sql = args.get("sql")
    if not sql:
        return Result.failure("db.query requires sql")
    try:
        scoped, params = scope_sql(sql, args.get("params") or [],
                                   getattr(ctx, "user_id", None))
    except ScopeError as exc:
        # Policy, not breakage — so ``except sdk.Denied`` catches it, which is
        # the distinction the whole Result contract rests on.
        return Result.refusal(str(exc))
    # ``scope_sql`` answers *whose* rows, never whether the statement reads.
    # A mutation arriving here has skipped the kernel-table check ``db.write``
    # carries, so it is refused rather than run.
    try:
        limit = int(args.get("max_rows") or DB_MAX_ROWS)
    except (TypeError, ValueError):
        limit = DB_MAX_ROWS
    limit = max(1, min(limit, DB_MAX_ROWS))
    try:
        return Result(data=_rows(db.query_rows(scoped, params, max_rows=limit)))
    except ValueError as exc:
        return Result.refusal(str(exc))
    except sqlite3.Error as exc:
        # The guest wrote this SQL, so a malformed statement is its mistake
        # rather than the kernel's. Anything else reaches the interpreter's
        # net, where it belongs -- with a traceback.
        return Result.failure(f"query failed: {exc}",
                              code=ERROR_INVALID_ARGUMENT)


def _db_write(ctx, args: dict) -> Result:
    """Insert, update or delete in a plugin-owned table.

    Writes are narrower than reads, which is the reverse of how the two
    usually go, and it follows from what each can be walked around. A broad
    read is contained by egress being gated; a broad write is contained by
    nothing — it is the effect itself. So the kernel's own tables are refused
    here and reached through the Requests that carry their access checks.
    """
    sql = args.get("sql")
    if not sql:
        return Result.failure("db.write requires sql")
    # Before the database is even resolved: whether this may be asked is a
    # policy question, and the answer must not depend on whether a database
    # happens to be wired up in this execution.
    try:
        scope_write(sql)
    except ScopeError as exc:
        return Result.refusal(str(exc))
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        db.execute_write(sql, tuple(args.get("params") or ()))
        return Result(data=True)
    except sqlite3.Error as exc:
        return Result.failure(f"write failed: {exc}",
                              code=ERROR_INVALID_ARGUMENT)


def _db_define(ctx, args: dict) -> Result:
    """Create a plugin-owned table.

    Same table check as ``db.write``: redefining or dropping a kernel table is
    the same trespass as writing rows into one, and DDL is the obvious way to
    try it once the write path is closed.
    """
    ddl = args.get("ddl")
    if not ddl:
        return Result.failure("db.define requires ddl")
    try:
        scope_write(ddl)
    except ScopeError as exc:
        return Result.refusal(str(exc))
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        db.execute_write(ddl)
        return Result(data=True)
    except sqlite3.Error as exc:
        return Result.failure(f"define failed: {exc}",
                              code=ERROR_INVALID_ARGUMENT)


# ──────────────────────────────────────────────────────────────────────
# Conversations.
# ──────────────────────────────────────────────────────────────────────

def _check_access(ctx, conversation_id) -> Result | None:
    """Refuse a conversation belonging to somebody else."""
    runtime = _runtime(ctx)
    check = getattr(runtime, "assert_conversation_access", None)
    if check is None or conversation_id is None:
        return None
    try:
        allowed = check(getattr(ctx, "session_key", None), conversation_id)
    except Exception as exc:
        return Result.refusal(f"conversation {conversation_id}: {exc}")
    if not allowed:
        return Result.refusal(
            f"conversation {conversation_id} is not available to this user")
    return None


def _conv_create(ctx, args: dict) -> Result:
    """Start a conversation."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    title = args.get("title") or "New conversation"
    category = args.get("category")
    uid = getattr(ctx, "user_id", None)
    runtime = _runtime(ctx)
    creator = getattr(runtime, "create_conversation", None)
    if creator is not None:
        cid = creator(
            title, kind="user", category=category, user_id=uid)
    else:
        try:
            cid = db.create_conversation(
                title, kind="user", category=category, user_id=uid)
        except TypeError:
            # Compatibility with small database doubles predating
            # ownership. Activation still requires the real runtime.
            if args.get("activate"):
                return Result.failure(
                    "conversation activation is not available "
                    "in this context")
            cid = db.create_conversation(title)
    if cid is None:
        return Result.failure("failed to create conversation")
    if not args.get("activate"):
        return Result(data=cid)

    key = getattr(ctx, "session_key", None)
    loader = getattr(runtime, "load_conversation", None)
    if (bad := _need(loader, "conversation loading")) is not None:
        return bad
    existing = (getattr(runtime, "sessions", None) or {}).get(key)
    if (
        existing is not None
        and getattr(existing, "conversation_id", None) not in (None, cid)
    ):
        runtime.close_session(key)
        runtime.set_session_user(key, uid)
    session = loader(key, cid)
    profile = (
        getattr(session, "profile_override", None)
        or getattr(session, "active_agent_profile", None)
        or "default"
    )
    return Result(data={"id": cid, "profile": profile})


def _conv_read(ctx, args: dict) -> Result:
    """Messages and metadata for one conversation."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    messages = _rows(db.get_conversation_messages(cid))
    data = {
        "conversation": dict(db.get_conversation(cid) or {}),
        "messages": messages,
    }
    if args.get("details"):
        from runtime.notifications import notification_mode
        from state_machine.serialization import latest_state

        state = latest_state(messages) or {}
        data["state"] = state
        data["agent_profile"] = (
            state.get("profile_override")
            or state.get("active_agent_profile")
            or ""
        ).strip()
        data["notification_mode"] = notification_mode(
            state.get("notification_mode"))
    return Result(data=data)


def _conv_list(ctx, args: dict) -> Result:
    """Conversations belonging to the current user."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    limit, bad = int_arg(args, "limit", 50, lo=1, hi=200)
    if bad is not None:
        return bad
    user_id = getattr(ctx, "user_id", None)
    if args.get("details"):
        import time

        category = args.get("category")
        rows, has_more = db.list_conversations_page(
            offset=0,
            limit=limit,
            category=category,
            user_id=user_id,
        )
        items = _rows(rows)
        for row in items:
            row["updated_ago"] = _relative_time(
                row.get("updated_at"), time.time())
        return Result(data={
            "items": items,
            "has_more": bool(has_more),
            "categories": list(
                db.list_conversation_categories(user_id=user_id)),
        })
    if user_id is not None and hasattr(db, "list_user_conversations"):
        return Result(data=_rows(db.list_user_conversations(user_id)))
    return Result(data=_rows(db.list_conversations()))


def _relative_time(timestamp, now) -> str:
    """Coarse relative age matching the conversation picker."""
    try:
        value = max(0.0, float(now) - float(timestamp))
    except (TypeError, ValueError):
        return ""
    units = (
        (60, "second", "seconds"),
        (60, "minute", "minutes"),
        (24, "hour", "hours"),
        (7, "day", "days"),
        (4, "week", "weeks"),
        (12, "month", "months"),
        (None, "year", "years"),
    )
    for step, singular, plural in units:
        if step is None or value < step:
            number = int(value) if value >= 1 else 1
            if singular == "second" and number < 5:
                return "just now"
            return (
                f"{number} "
                f"{singular if number == 1 else plural} ago"
            )
        value /= step
    return ""


def _conv_append(ctx, args: dict) -> Result:
    """Add a message."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    db.save_message(cid, args.get("role") or "user", args.get("content") or "")
    return Result(data=True)


def _conv_set_title(ctx, args: dict) -> Result:
    """Retitle a conversation."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    db.update_conversation_title(cid, args.get("title") or "")
    return Result(data=True)


def _conv_set_category(ctx, args: dict) -> Result:
    """Categorize a conversation."""
    runtime = _runtime(ctx)
    setter = getattr(runtime, "set_conversation_category", None)
    if (bad := _need(setter, "conversation categories")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    changed = setter(
        getattr(ctx, "session_key", None),
        cid,
        args.get("category") or None,
    )
    return Result(data=bool(changed))


def _conv_set_notification_mode(ctx, args: dict) -> Result:
    """Set one conversation's normalized background notification mode."""
    runtime = _runtime(ctx)
    setter = getattr(runtime, "set_conversation_notification_mode", None)
    if (bad := _need(setter, "conversation notifications")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    mode = setter(
        getattr(ctx, "session_key", None), cid, args.get("mode"))
    if mode is None:
        return Result.failure("no such conversation", code=ERROR_NOT_FOUND)
    return Result(data=mode)


def _conv_load(ctx, args: dict) -> Result:
    """Load an owned conversation and restore its persisted runtime state."""
    runtime = _runtime(ctx)
    loader = getattr(runtime, "load_history", None)
    if (bad := _need(loader, "conversation loading")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    outcome = loader(getattr(ctx, "session_key", None), cid)
    return Result(data={
        "ok": bool(getattr(outcome, "ok", True)),
        "messages": list(getattr(outcome, "messages", None) or []),
        "error": getattr(outcome, "error", None),
        "data": dict(getattr(outcome, "data", None) or {}),
    })


def _conv_clear(ctx, args: dict) -> Result:
    """Clear a conversation and refresh any active session displaying it."""
    runtime = _runtime(ctx)
    db = _db(ctx)
    if (bad := _need(runtime, "the runtime")) is not None:
        return bad
    if (bad := _need(db, "the database")) is not None:
        return bad

    key = getattr(ctx, "session_key", None)
    session = (getattr(runtime, "sessions", None) or {}).get(key)
    cid = args.get("id")
    if cid is None:
        cid = getattr(session, "conversation_id", None)
    if cid is None:
        return Result.failure("no conversation loaded", code=ERROR_NOT_FOUND)
    if (refused := _check_access(ctx, cid)) is not None:
        return refused

    db.clear_conversation_messages(cid)
    conversation = db.get_conversation(cid) or {}
    title = (conversation.get("title") or "").strip()
    if title and not title.endswith(" (cleared)"):
        db.update_conversation_title(cid, f"{title} (cleared)")

    if session is not None and getattr(session, "conversation_id", None) == cid:
        uid = runtime.session_user_id(key)
        runtime.close_session(key)
        runtime.set_session_user(key, uid)
        runtime.load_conversation(key, cid)
    return Result(data=True)


def _conv_delete(ctx, args: dict) -> Result:
    """Delete a conversation and its messages."""
    runtime = _runtime(ctx)
    deleter = getattr(runtime, "delete_conversation", None) or getattr(
        _db(ctx), "delete_conversation", None)
    if (bad := _need(deleter, "conversation deletion")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    if runtime is not None and hasattr(runtime, "delete_conversation"):
        deleted = deleter(getattr(ctx, "session_key", None), cid)
    else:
        deleted = deleter(cid)
    return Result(data=bool(deleted))


# ──────────────────────────────────────────────────────────────────────
# Sessions.
# ──────────────────────────────────────────────────────────────────────

def _session_profile(runtime, session) -> str:
    """The agent profile this session resolves to, override and all.

    ``runtime_config.profile_for`` is the one place that knows the precedence
    (session override, then frontend pin, then user setting, then global), so
    this asks it rather than reimplementing three quarters of it.
    """
    try:
        from runtime.runtime_config import profile_for

        return profile_for(runtime, session) or ""
    except Exception:
        logger.exception("could not resolve the session's agent profile")
        return ""


def _session_get(ctx, args: dict) -> Result:
    """Describe one live session."""
    runtime = _runtime(ctx)
    if (bad := _need(runtime, "the runtime")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    session = (getattr(runtime, "sessions", None) or {}).get(key)
    if session is None:
        return Result(data=None)
    # The phase is what the state machine is doing right now. A frontend needs
    # it to know whether the machine is already collecting an answer — if it
    # is, the frontend must not also interpret the next line, or one keystroke
    # gets consumed twice.
    machine = getattr(session, "cs", None)
    data = {
        "key": key,
        "conversation_id": getattr(session, "conversation_id", None),
        "phase": getattr(machine, "phase", None),
        "busy": bool(getattr(session, "busy", False)),
        "attended": bool(runtime.is_attended(key))
        if hasattr(runtime, "is_attended") else None,
        # Which agent profile is actually driving *this* session, and which
        # frontend it belongs to. Both were reachable only through
        # ``list_sessions``, whose rows the guest receives stringified — so
        # anything wanting to report the live profile had to fall back to the
        # global default and be wrong for any session that overrode it.
        "agent_profile": _session_profile(runtime, session),
        "frontend": getattr(session, "frontend_name", None),
        "user_id": getattr(session, "user_id", None),
        # How this conversation answers approval dialogs. Answered from the
        # runtime's reader rather than the session field, because the field
        # alone does not say whether it still applies — it is scoped to the
        # conversation it was set against, and a turn-scoped mode outranks it.
        "mode": runtime.security_mode(key)
        if hasattr(runtime, "security_mode") else None,
    }
    if args.get("details"):
        if machine is None:
            data["debug"] = None
            return Result(data=data)
        # Whatever live services want to say about this session. The
        # state-machine dump and event log that used to sit beside this came
        # from state_machine/debug.py, which restated phase and cache contents
        # nobody read; ``phase``, ``busy`` and ``attended`` above are the part
        # that was actually useful.
        data["debug"] = {
            "service_flags": [
                flag
                for service in (getattr(ctx, "services", None) or {}).values()
                for flag in (
                    service.debug_flags(session)
                    if callable(getattr(service, "debug_flags", None)) else []
                )
            ],
        }
    return Result(data=data)


def _session_list(ctx, args: dict) -> Result:
    """Every live session key."""
    runtime = _runtime(ctx)
    if (bad := _need(runtime, "the runtime")) is not None:
        return bad
    lister = getattr(runtime, "list_sessions", None)
    if lister is not None:
        return Result(data=[str(s) for s in lister()])
    return Result(data=list(getattr(runtime, "sessions", None) or {}))


def _session_push(ctx, args: dict) -> Result:
    """Send a message to the user out of band."""
    runtime = _runtime(ctx)
    push = getattr(runtime, "push_message", None)
    if (bad := _need(push, "proactive messages")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    push(key, args.get("message") or "")
    return Result(data=True)


def _session_state_get(ctx, args: dict) -> Result:
    """Read this plugin's per-session scratch state."""
    runtime = _runtime(ctx)
    getter = getattr(runtime, "get_session_plugin_state", None)
    if (bad := _need(getter, "session state")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    return Result(data=getter(key, args.get("namespace") or "sandbox"))


def _session_state_set(ctx, args: dict) -> Result:
    """Write this plugin's per-session scratch state."""
    runtime = _runtime(ctx)
    setter = getattr(runtime, "update_session_plugin_state", None)
    if (bad := _need(setter, "session state")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    setter(key, args.get("namespace") or "sandbox", args.get("value"))
    return Result(data=True)


def _session_cancel(ctx, args: dict) -> Result:
    """Cancel the turn running on a session."""
    runtime = _runtime(ctx)
    canceller = getattr(runtime, "cancel_session", None)
    if (bad := _need(canceller, "session cancellation")) is not None:
        return bad
    outcome = canceller(args.get("key") or getattr(ctx, "session_key", None))
    if outcome is None:
        return Result(data=None)
    return Result(data={
        "ok": bool(getattr(outcome, "ok", True)),
        "messages": list(getattr(outcome, "messages", None) or []),
        "error": getattr(outcome, "error", None),
        "data": dict(getattr(outcome, "data", None) or {}),
    })


def _session_add_tool(ctx, args: dict) -> Result:
    """Widen the agent's scope for this session."""
    runtime = _runtime(ctx)
    adder = getattr(runtime, "add_session_tool", None)
    if (bad := _need(adder, "session scope")) is not None:
        return bad
    adder(args.get("key") or getattr(ctx, "session_key", None),
          args.get("tool"))
    return Result(data=True)


def _session_remove_tool(ctx, args: dict) -> Result:
    """Narrow the agent's scope for this session."""
    runtime = _runtime(ctx)
    remover = getattr(runtime, "remove_session_tool", None)
    if (bad := _need(remover, "session scope")) is not None:
        return bad
    remover(args.get("key") or getattr(ctx, "session_key", None),
            args.get("tool"))
    return Result(data=True)


def _session_set_mode(ctx, args: dict) -> Result:
    """Set how this conversation answers approval dialogs.

    The verdict about *whether* this may happen was already made by
    ``policy.classify`` and, for a widening, by the approver — so there is
    nothing to check here. Normalization is the runtime's, in one place, so an
    unknown mode degrades to ``ask`` rather than reaching the approver as
    something it has no answer for.
    """
    runtime = _runtime(ctx)
    setter = getattr(runtime, "set_security_mode", None)
    if (bad := _need(setter, "the security mode")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    mode = setter(key, args.get("mode"), scope=args.get("scope") or "conversation")
    if mode is None:
        return Result.failure(f"no live session for {key!r}",
                              code=ERROR_NOT_FOUND)
    return Result(data=mode)


def _prompt_slot(args: dict) -> str:
    """Which named overlay this write owns.

    Two keys are in play and conflating them is what broke this Request for
    its whole life: ``key`` is the *session*, ``slot`` is the entry within
    that session's ``system_prompt_extras``. The slot defaults to the calling
    plugin rather than to a constant, because overlays are a dict and two
    plugins sharing one name would silently overwrite each other — the leaf of
    the chain is the nearest thing to "who is writing this" that the guest
    cannot misstate.
    """
    if slot := str(args.get("slot") or "").strip():
        return slot
    from .. import provenance

    caller = provenance.current()
    chain = getattr(caller, "chain", None) if caller is not None else None
    links = list(getattr(chain, "links", ()) or ())
    return links[-1] if links else "sandbox"


def _session_add_prompt(ctx, args: dict) -> Result:
    """Inject system prompt text for this session."""
    runtime = _runtime(ctx)
    adder = getattr(runtime, "add_system_prompt_extra", None)
    if (bad := _need(adder, "prompt extras")) is not None:
        return bad
    slot = _prompt_slot(args)
    adder(args.get("key") or getattr(ctx, "session_key", None),
          slot, args.get("text") or "")
    # The slot is the handle: ``remove_prompt`` needs the name back, and
    # returning the runtime's bool told the caller nothing it could use.
    return Result(data=slot)


def _session_remove_prompt(ctx, args: dict) -> Result:
    """Withdraw injected prompt text."""
    runtime = _runtime(ctx)
    remover = getattr(runtime, "remove_system_prompt_extra", None)
    if (bad := _need(remover, "prompt extras")) is not None:
        return bad
    remover(args.get("key") or getattr(ctx, "session_key", None),
            str(args.get("handle") or "") or _prompt_slot(args))
    return Result(data=True)


def _session_add_attachment(ctx, args: dict) -> Result:
    """Stage a file for this session's next model call.

    The kernel opens the path rather than the guest, which is what lets an
    image reach the model at all: the bytes go straight into the prompt and
    never cross a wire. Read policy is therefore the read policy — the same
    ``reason_for`` guard ``fs.read`` applies, because a route to the model's
    context that skipped it would be a way to read ``config.json`` aloud.

    ``parse_attachment`` rather than a hand-built ``Attachment``: it is what
    fills ``parsed_text``, and that is exactly what the routing falls back to
    when the model cannot see the modality.
    """
    raw = args.get("path")
    if not raw:
        return Result.failure("session.add_attachment requires a path",
                              code=ERROR_INVALID_ARGUMENT)
    path = Path(str(raw))
    if (why := reason_for(path)):
        return Result.refusal(f"{raw} is not readable: {why}",
                              code=ERROR_NOT_PERMITTED)
    if not path.is_file():
        return Result.failure(f"{raw} is not a file", code=ERROR_NOT_FOUND)

    runtime = _runtime(ctx)
    stage = getattr(runtime, "add_turn_attachment", None)
    if (bad := _need(stage, "attachment staging")) is not None:
        return bad
    from attachments.parse import parse_attachment

    attachment = parse_attachment(str(path), file_name=path.name,
                                  config=getattr(ctx, "config", None))
    key = args.get("key") or getattr(ctx, "session_key", None)
    if not stage(key, attachment):
        # Answering ``data=False`` would let a caller that ignores the return
        # value drop the file silently, which reads to the agent as a model
        # that looked and saw nothing.
        return Result.failure("no live session to attach to",
                              code=ERROR_NOT_FOUND)
    return Result(data=True)


# ──────────────────────────────────────────────────────────────────────
# Talking to the user.
# ──────────────────────────────────────────────────────────────────────

def _ui_ask(ctx, args: dict) -> Result:
    """Ask a question and wait for the answer.

    The guest says ``choices``; the state machine says ``enum``. Translating
    between the two is this handler's job, and for a while it did not do it —
    ``choices=`` went straight through to ``request_input``, which has no such
    parameter, so every question with options died as a ``TypeError`` reported
    back as "could not ask". Nothing caught it because nothing sandboxed had
    asked a multiple-choice question yet.

    The prompt is assembled through ``form_step_display`` so a sandboxed
    question reads exactly like a native form step — "Select an option.",
    "Reply with a whole number." and the rest. That assembly used to live in
    the asking plugin, which is no longer possible: the guest cannot import
    ``state_machine``.
    """
    asker = getattr(ctx, "request_user_input", None)
    runtime = _runtime(ctx)
    key = getattr(ctx, "session_key", None)
    if asker is None and runtime is not None and key:
        def asker(title, prompt, **kw):
            """Fall back to the runtime's own prompt."""
            return runtime.request_input(key, title, prompt, **kw)
    if (bad := _need(asker, "asking the user")) is not None:
        return bad

    # "text" was the old default and is not a FormStep type, so it matched
    # none of the display branches and rendered a question with no assistance.
    answer_type = args.get("type") or "string"
    choices = args.get("choices") or None
    default = args.get("default")
    required = args.get("required")
    required = True if required is None else bool(required)
    timeout, bad = float_arg(args, "timeout", 300.0, lo=0.0)
    if bad is not None:
        return bad

    try:
        prompt = _ask_prompt(args.get("prompt") or "", answer_type,
                             choices, default, required)
        request = asker(args.get("title") or "Question", prompt,
                        type=answer_type, enum=choices,
                        default=default, required=required)
        if not request.wait(timeout=timeout):
            return Result.failure("the user did not answer", retryable=True)
        if request.metadata.get("cancelled"):
            return Result.refusal("the user cancelled")
        return Result(data=getattr(request, "value", None)
                      if hasattr(request, "value") else request.approved)
    except Exception as exc:
        logger.exception("ui_ask failed")
        return Result.failure(f"could not ask: {exc}")


def _ask_prompt(prompt: str, answer_type, choices, default, required) -> str:
    """The question plus whatever assistance its answer type calls for."""
    try:
        from state_machine.conversation import FormStep
        from state_machine.form_display import form_step_display

        display = form_step_display(FormStep(
            "answer", prompt, required, answer_type, choices, default=default))
    except Exception:
        # A missing or changed state machine must not make the question
        # unaskable — the prompt on its own is still a usable question.
        return prompt
    return "\n\n".join(part for part in
                       [display.get("prompt"), display.get("assist")] if part)


def _ui_approve(ctx, args: dict) -> Result:
    """Report that the user approved — which reaching here already means.

    The Request *is* the question. ``ui.approve`` is classified unconditionally
    unsafe (``policy.classify``), so the gate has already run the whole
    approval pipeline — hooks, attendance, the dialog — and a refusal never
    arrives here at all. There is no second question to ask and nothing left
    to decide.

    This used to call ``context.approve_command``, a parallel doorway with its
    own hook call, its own reading of the trusted list, and no attendance
    check. It is gone; this is what replaced it.
    """
    return Result(data=True)


def _ui_render(ctx, args: dict) -> Result:
    """Show files to the user in chat.

    The paths ride on the push rather than being counted into its text. This
    once said ``f"{len(paths)} file(s)"`` and dropped them, so the one Request
    named for showing a file was the one thing that could not.
    """
    runtime = _runtime(ctx)
    push = getattr(runtime, "push_message", None)
    if (bad := _need(push, "rendering to the user")) is not None:
        return bad
    paths = [str(p) for p in (args.get("paths") or [])]
    caption = args.get("caption") or ""
    try:
        push(getattr(ctx, "session_key", None), caption, attachments=paths)
        return Result(data={"rendered": len(paths)})
    except Exception as exc:
        logger.exception("ui_render failed")
        return Result.failure(f"render failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Config, users.
# ──────────────────────────────────────────────────────────────────────

def _config_read(ctx, args: dict) -> Result:
    """Read a setting, redacting credentials into handles."""
    config = getattr(ctx, "config", None) or {}
    key = args.get("key")
    if args.get("details"):
        from config.config_data import SETTINGS_DATA
        from plugins.plugin_discovery import (
            get_plugin_setting_scope,
            get_plugin_settings,
            get_setting_plugin_names,
        )

        plugin_entries = list(get_plugin_settings())
        plugin_keys = {entry[1] for entry in plugin_entries}
        items = []
        for entry in [*SETTINGS_DATA, *plugin_entries]:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 5
            ):
                continue
            title, name, description, default, raw_info = entry
            info = raw_info if isinstance(raw_info, dict) else {}
            if info.get("hidden") is True or (key and name != key):
                continue
            scope = (
                "user"
                if info.get("scope") == "user"
                or (
                    name in plugin_keys
                    and get_plugin_setting_scope(name) == "user"
                )
                else "global"
            )
            owners = list(get_setting_plugin_names(name) or [])
            category = (
                "user" if scope == "user"
                else "plugin" if name in plugin_keys
                else "kernel"
            )
            items.append({
                "title": title,
                "key": name,
                "description": description,
                "default": default,
                "info": info,
                "scope": scope,
                "category": category,
                "storage": {
                    "kernel": "config.json",
                    "plugin": "plugin_config.json",
                    "user": "per-user",
                }[category],
                "owners": owners,
                "current": redact(
                    name, _config_value(ctx, name, entry), guess=True),
                "restart_required": False,
            })
        # Plugin discovery already tracks the declaring family. Use its
        # public query rather than exposing plugin objects to the guest.
        from plugins.plugin_discovery import get_plugin_setting_type
        for item in items:
            item["restart_required"] = (
                get_plugin_setting_type(item["key"]) == "frontend")
        return Result(data=sorted(items, key=lambda item: item["key"]))
    if args.get("present"):
        return Result(data=bool(config.get(key))) if key else Result(
            data=bool(config))
    if args.get("keys"):
        value = config.get(key) if key else config
        if value is None:
            return Result(data=[])
        if not isinstance(value, dict):
            return Result.failure(
                f"config setting {key!r} is not a mapping")
        return Result(data=sorted(str(item) for item in value))
    if key is None:
        return Result(data={
            k: redact_nested(k, v) for k, v in config.items()})
    if key not in config:
        return Result(data=None)
    return Result(data=redact_nested(key, config[key]))


def _config_write(ctx, args: dict) -> Result:
    """Change a setting."""
    key = args.get("key")
    if not key:
        return Result.failure("config.write requires a key")
    config = getattr(ctx, "config", None)
    if (bad := _need(config, "config")) is not None:
        return bad
    from config import config_manager

    value = config_manager.migrate_secret_keys(
        key, resolve(args.get("value"), lookup_from(ctx)))
    if args.get("merge"):
        current = config.get(key)
        if current is not None and not isinstance(current, dict):
            return Result.failure(
                f"config setting {key!r} is not a mapping")
        if not isinstance(value, dict):
            return Result.failure(
                "config.write merge requires a mapping value")
        value = {**(current or {}), **value}
    old = config.get(key)
    # Only what can actually fail stays guarded: the settings
    # catalogue, the database, the config file, and the watcher.
    try:
        if config_manager.is_user_scoped(key):
            db = _db(ctx)
            getter = getattr(db, "get_user_config", None)
            setter = getattr(db, "set_user_config", None)
            if getter is None or setter is None:
                return Result.failure(
                    "user settings are not available in this context")
            uid = getattr(ctx, "user_id", None)
            user_config = getter(uid) or {}
            user_config[key] = value
            setter(uid, user_config)
            config[key] = value
            runtime = _runtime(ctx)
            if (
                key == "active_agent_profile"
                and runtime is not None
                and getattr(ctx, "session_key", None)
                and hasattr(runtime, "set_agent_profile")
            ):
                runtime.set_agent_profile(
                    getattr(ctx, "session_key"), value)
            if (
                runtime is not None
                and hasattr(runtime, "refresh_session_specs")
            ):
                runtime.refresh_session_specs()
            return Result(data=True)
        config[key] = value
        renames = config_manager.detect_profile_renames(old, value) \
            if key == "agent_profiles" else {}
        if renames:
            runtime = _runtime(ctx)
            for session in (
                getattr(runtime, "sessions", {}) or {}
            ).values():
                active = getattr(
                    session, "active_agent_profile", None)
                override = getattr(session, "profile_override", None)
                if active in renames:
                    session.active_agent_profile = renames[active]
                if override in renames:
                    session.profile_override = renames[override]
        config_manager.save(config)
        # ``is_kernel_setting`` first, and it overrides both the declaration
        # and the caller's ``scope``: a key the kernel declares has one home,
        # and taking this branch as well wrote it to a second file and
        # announced it a second time.
        if not config_manager.is_kernel_setting(key) \
                and (key in config_manager.plugin_setting_keys()
                     or args.get("scope") == "plugin"):
            plugin_config = config_manager.load_plugin_config()
            plugin_config[key] = value
            config_manager.save_plugin_config(plugin_config)
        runtime = _runtime(ctx)
        if runtime is not None and getattr(runtime, "config", None) is not None:
            runtime.config[key] = value
        if key in {"llm_profiles", "default_llm_profile"}:
            try:
                import llm

                llm.refresh(config)
            except Exception:
                # The persisted profile is authoritative and discovery can
                # reconcile it on restart when live refresh is unavailable.
                pass
        if runtime is not None and hasattr(runtime, "refresh_session_specs"):
            runtime.refresh_session_specs()
        if (
            value != old
            and key in {
                "sync_directories", "ignored_extensions",
                "ignored_folders", "skip_hidden_folders",
            }
        ):
            watcher = getattr(
                getattr(ctx, "orchestrator", None), "watcher", None)
            rescan = getattr(watcher, "rescan", None)
            if rescan is not None:
                rescan()
        return Result(data=True)
    except Exception as exc:
        logger.exception("config_write failed")
        return Result.failure(f"config write failed: {exc}")


def _path_get(ctx, args: dict) -> Result:
    """Resolve one of the application locations exposed to plugins.

    Two of these are not directories, and belong here anyway. The validator
    refuses ``sys``, which is correct — it is a door to the interpreter, not a
    fact about it — but two of the facts behind that door are things a plugin
    legitimately needs and cannot otherwise learn: which Python is hosting the
    app (so ``pip install`` targets *this* environment rather than whatever is
    first on PATH), and which platform it is on (so a command line is built
    for the right shell). Both are constants the kernel already knows, so
    answering them costs nothing and closes the only honest reason to want
    ``sys``.
    """
    import sys

    import trees
    from paths import DATA_DIR, ROOT_DIR

    locations = {
        "project": getattr(ctx, "root_dir", None) or ROOT_DIR,
        "data": DATA_DIR,
        "bundled": trees.tree("bundled").path,
        "installed": trees.tree("installed").path,
        "workspace": trees.tree("workspace").path,
        # Named rather than left to be built out of ``workspace``: where a
        # script goes is what decides whether it can be run without a dialog,
        # so it is a fact the kernel states rather than a path convention a
        # plugin is expected to remember.
        "scripts": trees.tree("workspace").path / "scripts",
        "python": sys.executable,
        "platform": sys.platform,
    }
    name = args.get("name")
    if name not in locations:
        return Result.failure(
            f"unknown application path {name!r}; expected one of "
            f"{sorted(locations)}")
    return Result(data=str(locations[name]))


def _visible_user(row) -> dict:
    """A user row with its secret columns removed."""
    return {k: v for k, v in dict(row or {}).items()
            if k not in HIDDEN_USER_COLUMNS}


def _user_read(ctx, args: dict) -> Result:
    """One user, minus anything never returned."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    uid = args.get("id", getattr(ctx, "user_id", None))
    return Result(data=_visible_user(db.get_user(uid)))


def _user_list(ctx, args: dict) -> Result:
    """Every user, minus anything never returned."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    return Result(data=[_visible_user(r) for r in db.list_users() or []])


def _user_write(ctx, args: dict) -> Result:
    """Update a user's config blob or type."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    uid = args.get("id", getattr(ctx, "user_id", None))
    if "config" in args:
        db.set_user_config(uid, args["config"])
    if "user_type" in args:
        db.set_user_type(uid, args["user_type"])
    return Result(data=True)


# ──────────────────────────────────────────────────────────────────────
# Plugins, services, tools, commands.
# ──────────────────────────────────────────────────────────────────────

def _frontend_description(name: str, adapters: dict) -> str:
    """One frontend's declared description, running or not.

    A running frontend has an adapter carrying the declaration; a merely
    installed one has only its class, which discovery can still hand over. The
    listing shows both, so the description has to come from both.
    """
    adapter = adapters.get(name)
    if adapter is not None:
        return getattr(adapter, "description", "") or ""
    try:
        from plugins.plugin_discovery import discover_frontends

        found = discover_frontends().get(name)
        return getattr(found, "description", "") or ""
    except Exception:
        logger.exception("could not read %s's description", name)
        return ""


def _plugin_list(ctx, args: dict) -> Result:
    """Everything currently registered, by family."""
    source = args.get("source") or "registered"
    if source == "families":
        # Which categories the store can hold, derived from the layout rather
        # than restated. ``/packages`` hardcoded six and zipped its labels
        # against them, which silently discarded the ``llm`` and ``parsers``
        # counts it had already computed — so two whole categories of package
        # were invisible in a menu built from the correct data.
        import trees

        from bundled.commands.helpers.package_manager import EXTRA_FAMILIES

        return Result(data=[root.name for root in trees.ROOTS]
                      + list(EXTRA_FAMILIES))
    if source != "registered":
        try:
            from paths import ROOT_DIR
            from bundled.commands.helpers import package_manager

            root = getattr(ctx, "root_dir", None) or ROOT_DIR
            if source == "available":
                installed = {
                    item["path"]
                    for item in package_manager.installed_packages()
                }
                items = [
                    item for item in package_manager.search_packages(root)
                    if item["path"] not in installed
                ]
            elif source == "installed":
                items = package_manager.installed_packages()
            elif source == "removable":
                items = (
                    package_manager.removable_packages()
                    + package_manager.search_bundles(root)
                )
            else:
                return Result.failure(
                    f"unknown plugin list source {source!r}")
            category = args.get("category")
            if category:
                items = [
                    item for item in items
                    if item.get("family") == category
                ]
            return Result(data=items)
        except Exception as exc:
            logger.exception("plugin_list failed")
            return Result.failure(str(exc))

    category = args.get("category")
    if args.get("details") and category == "frontends":
        runtime = _runtime(ctx)
        manager = getattr(runtime, "frontend_manager", None)
        available = set(
            getattr(manager, "available_frontends", None) or [])
        adapters = dict(getattr(manager, "adapters", None) or {})
        config = getattr(ctx, "config", None) or {}
        enabled = set(config.get("enabled_frontends") or [])
        profiles = set((config.get("frontend_profiles") or {}).keys())
        names = sorted(available | set(adapters) | enabled | profiles)
        return Result(data=[
            {
                "name": name,
                # A disabled frontend has no adapter, so its description comes
                # from the class discovery found rather than from a live
                # object. Both are the same declaration.
                "description": _frontend_description(name, adapters),
                "available": name in available,
                "loaded": name in adapters,
                "config_settings": [
                    {
                        "title": entry[0],
                        "key": entry[1],
                        "description": entry[2],
                        "default": entry[3],
                        "info": (
                            entry[4]
                            if isinstance(entry[4], dict)
                            else {}
                        ),
                        "current": redact(
                            entry[1],
                            _config_value(ctx, entry[1], entry),
                            guess=True,
                        ),
                    }
                    for entry in (
                        getattr(
                            adapters.get(name),
                            "config_settings",
                            None,
                        )
                        or []
                    )
                    if isinstance(entry, (list, tuple))
                    and len(entry) == 5
                    and not (
                        isinstance(entry[4], dict)
                        and entry[4].get("hidden") is True
                    )
                ],
            }
            for name in names
        ])

    role = args.get("role")
    if role:
        if role != "llm_backend":
            return Result.failure(f"unknown plugin role {role!r}",
                              code=ERROR_INVALID_ARGUMENT)
        if args.get("category") not in (None, "", "services", "llm"):
            return Result(data=[])
        import llm

        # Names, not display names: this answers "what may a profile's
        # ``llm_service_class`` be set to", and the stored value has to be the
        # class name. What a *person* reads is ``llm.list``'s ``backends``,
        # which carries both.
        return Result(data=llm.backend_names())

    registry = getattr(ctx, "tool_registry", None)
    orchestrator = getattr(ctx, "orchestrator", None)
    commands = getattr(ctx, "command_registry", None)
    return Result(data={
        "tools": sorted(getattr(registry, "tools", None) or {}),
        "tasks": sorted(getattr(orchestrator, "tasks", None) or {}),
        "services": sorted(getattr(ctx, "services", None) or {}),
        "commands": sorted(getattr(commands, "commands", None) or {}),
    })


def _package_progress(ctx):
    """Send long-running package progress to the issuing frontend."""
    runtime = _runtime(ctx)
    key = getattr(ctx, "session_key", None)
    push = getattr(runtime, "push_message", None)
    if push is None or not key:
        return None
    return lambda message: push(key, message, source="packages")


def _plugin_install(ctx, args: dict) -> Result:
    """Install one store package through the kernel package manager."""
    try:
        from paths import ROOT_DIR
        from bundled.commands.helpers import package_manager

        root = getattr(ctx, "root_dir", None) or ROOT_DIR
        outcome = package_manager.install_package(
            root, args.get("package_id") or "", ctx,
            progress=_package_progress(ctx))
        return Result(data=outcome.text())
    except Exception as exc:
        logger.exception("plugin_install failed")
        return Result.failure(str(exc))


def _plugin_uninstall(ctx, args: dict) -> Result:
    """Uninstall one package through the kernel package manager."""
    try:
        from paths import ROOT_DIR
        from bundled.commands.helpers import package_manager

        root = getattr(ctx, "root_dir", None) or ROOT_DIR
        outcome = package_manager.uninstall_package(
            args.get("package_id") or "", ctx,
            progress=_package_progress(ctx), root_dir=root)
        return Result(data=outcome.text())
    except Exception as exc:
        logger.exception("plugin_uninstall failed")
        return Result.failure(str(exc))


def _plugin_update(ctx, args: dict) -> Result:
    """Update all installed packages through the kernel package manager."""
    try:
        from paths import ROOT_DIR
        from bundled.commands.helpers import package_manager

        root = getattr(ctx, "root_dir", None) or ROOT_DIR
        outcome = package_manager.update_packages(
            root, ctx, progress=_package_progress(ctx))
        return Result(data=outcome.text())
    except Exception as exc:
        logger.exception("plugin_update failed")
        return Result.failure(str(exc))


def _plugin_describe(ctx, args: dict) -> Result:
    """Metadata for one registered plugin."""
    name = args.get("name")
    registry = getattr(ctx, "tool_registry", None)
    getter = getattr(registry, "get_schema", None)
    if getter is not None:
        schema = getter(name)
        if schema is not None:
            return Result(data=schema)
    return Result.failure(f"no plugin named {name!r}")


def _known_names(ctx, path) -> list:
    """Every registered plugin name, minus the ones this file already owns.

    The validator's duplicate-name check exists to stop a new plugin
    shadowing an existing one. Re-validating a file that is *already*
    registered would otherwise report it as a duplicate of itself — the
    single most common case, since the point of the check is to run it after
    every edit. So entries whose ``_source_path`` is this file are dropped.
    """
    target = str(path)
    names = []
    registries = (
        getattr(getattr(ctx, "tool_registry", None), "tools", None),
        getattr(getattr(ctx, "orchestrator", None), "tasks", None),
        getattr(ctx, "services", None),
        getattr(getattr(ctx, "command_registry", None), "commands", None),
    )
    for registry in registries:
        for name, obj in dict(registry or {}).items():
            if str(getattr(obj, "_source_path", "") or "") != target:
                names.append(name)
    return names


def _plugin_validate(ctx, args: dict) -> Result:
    """Lint one source file against the sandbox contract.

    The same validator the loader runs, answering to the code being authored
    rather than only to the kernel refusing it — which is the difference
    between "your plugin did not load" and a line number with a fix on it.
    Nothing is imported or executed: this is a pure AST walk.
    """
    from plugins.plugin_paths import resolve_plugin_path

    from ..validator import validate_file

    path, error = resolve_plugin_path((args.get("path") or "").strip())
    if error:
        return Result.failure(error)
    if not path.is_file():
        return Result.failure(f"no such file: {path}", code=ERROR_NOT_FOUND)
    if path.suffix != ".py":
        return Result.failure(f"not a Python file: {path.name}")
    try:
        report = validate_file(path, known_names=_known_names(ctx, path))
    except OSError as exc:
        return Result.failure(f"could not read {path}: {exc}", retryable=True)

    return Result(data={
        "path": str(path),
        "filename": report.filename,
        "ok": report.ok,
        "disclaimed": report.disclaimed,
        "digest": report.digest,
        # A set does not cross the wire, and the order matters to nobody but
        # the reader — so sorted, which also makes the answer deterministic.
        "unmediated": sorted(report.unmediated),
        "declarations": _plain(report.declarations),
        "findings": [{"level": f.level, "line": f.line,
                      "message": f.message, "fix": f.fix}
                     for f in report.findings],
    })


def _plain(value):
    """Coerce declarations to JSON-safe plain data.

    ``declarations`` comes off an AST walk, so a value is whatever
    ``ast.literal_eval`` produced — including tuples and sets, which the wire
    does not carry.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _plugin_watcher(ctx):
    """The kernel coordinator shared by filesystem and SDK changes."""
    runtime = _runtime(ctx)
    return getattr(runtime, "plugin_watcher", None)


def _plugin_target(ctx, args: dict):
    """Resolve and validate one requested plugin source path."""
    watcher = _plugin_watcher(ctx)
    if watcher is None:
        return None, Result.failure("the plugin watcher is not available")
    raw_path = args.get("path") or ""
    if raw_path:
        from plugins.plugin_paths import plugin_info, resolve_plugin_path

        path, error = resolve_plugin_path(raw_path)
        if error:
            return None, Result.failure(error)
        _info, error = plugin_info(path)
        if error:
            return None, Result.failure(error)
        return path, None
    name = args.get("name") or ""
    if not name:
        return None, Result.failure("path or name is required")
    path, error = watcher.resolve_registered(
        name,
        args.get("family") or "",
    )
    if error:
        return None, Result.failure(error)
    return path, None


def _plugin_register(ctx, args: dict) -> Result:
    """Load one path through the kernel plugin coordinator."""
    watcher = _plugin_watcher(ctx)
    if watcher is None:
        return Result.failure("the plugin watcher is not available")
    path, failure = _plugin_target(ctx, args)
    if failure is not None:
        return failure
    outcome = watcher.register(path)
    if not outcome.get("ok"):
        return Result.failure(outcome.get("error") or "plugin registration failed")
    return Result(data=outcome)


def _plugin_unregister(ctx, args: dict) -> Result:
    """Unload one path through the kernel plugin coordinator."""
    watcher = _plugin_watcher(ctx)
    path, failure = _plugin_target(ctx, args)
    if failure is not None:
        return failure
    outcome = watcher.unregister(path)
    if not outcome.get("ok"):
        return Result.failure(
            outcome.get("error") or "plugin unregistration failed"
        )
    return Result(data=outcome)


def _plugin_reload(ctx, args: dict) -> Result:
    """Reload one path through the kernel plugin coordinator."""
    watcher = _plugin_watcher(ctx)
    path, failure = _plugin_target(ctx, args)
    if failure is not None:
        return failure
    outcome = watcher.reload(path)
    if not outcome.get("ok"):
        return Result.failure(outcome.get("error") or "plugin reload failed")
    return Result(data=outcome)


def _service_list(ctx, args: dict) -> Result:
    """Loaded services and whether each is ready."""
    services = getattr(ctx, "services", None) or {}
    if args.get("details"):
        from plugins.native.service import service_lifecycle

        return Result(data=[{
            "name": name,
            "description": getattr(service, "description", "") or "",
            "loaded": bool(getattr(service, "loaded", False)),
            "lifecycle": service_lifecycle(service),
            "config_settings": [
                {
                    "title": entry[0],
                    "key": entry[1],
                    "description": entry[2],
                    "default": entry[3],
                    "info": entry[4] if isinstance(entry[4], dict) else {},
                    # Redacted like every other ``details=True`` listing —
                    # frontends, tools and tasks all do this, and services
                    # were the one that did not, so ``/services`` printed a
                    # provider's API key in plaintext into the chat. Masking
                    # here rather than in the command is the point: a
                    # command-side mask still puts the key on the wire.
                    "current": redact(entry[1],
                                      _config_value(ctx, entry[1], entry),
                                      guess=True),
                }
                for entry in (getattr(service, "config_settings", None) or [])
                if isinstance(entry, (list, tuple))
                and len(entry) == 5
                and not (
                    isinstance(entry[4], dict)
                    and entry[4].get("hidden") is True
                )
            ],
        } for name, service in sorted(services.items())])
    return Result(data={
        name: bool(getattr(service, "loaded", False))
        for name, service in services.items()
    })


def _config_value(ctx, key, entry):
    """Resolve a setting value with the same user/global scope semantics."""
    info = entry[4] if isinstance(entry[4], dict) else {}
    scope = "user" if info.get("scope") == "user" else "global"
    if scope == "user":
        db = _db(ctx)
        getter = getattr(db, "get_user_config", None)
        if getter is not None:
            values = getter(getattr(ctx, "user_id", None)) or {}
            return values.get(
                key, (getattr(ctx, "config", None) or {}).get(key, entry[3]))
    return (getattr(ctx, "config", None) or {}).get(key)


def _clear_task_skip_cache(ctx):
    orchestrator = getattr(ctx, "orchestrator", None)
    clear = getattr(orchestrator, "clear_skip_cache", None)
    if clear is not None:
        clear()


def _service_load(ctx, args: dict) -> Result:
    """Load one user-managed service."""
    name = args.get("name")
    service = _service(ctx, name)
    if service is None:
        return Result.failure(f"service {name!r} is not registered",
                              code=ERROR_NOT_FOUND)
    from plugins.native.service import is_user_managed_service

    if not is_user_managed_service(service):
        return Result.refusal(
            f"{name} is an installed extension and is loaded automatically")
    try:
        loaded = service.load()
        if loaded is not False:
            _clear_task_skip_cache(ctx)
        return Result(data=loaded is not False)
    except Exception as exc:
        logger.exception("service_load failed")
        return Result.failure(f"service load failed: {exc}")


def _service_unload(ctx, args: dict) -> Result:
    """Unload one user-managed service."""
    name = args.get("name")
    service = _service(ctx, name)
    if service is None:
        return Result.failure(f"service {name!r} is not registered",
                              code=ERROR_NOT_FOUND)
    from plugins.native.service import is_user_managed_service

    if not is_user_managed_service(service):
        return Result.refusal(
            f"{name} is an installed extension and is loaded automatically")
    try:
        service.unload()
        _clear_task_skip_cache(ctx)
        return Result(data=True)
    except Exception as exc:
        logger.exception("service_unload failed")
        return Result.failure(f"service unload failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# The model authority. Profiles are not services and have not been since
# ``service_llm.py`` was deleted — but ``/llm`` went on asking the service
# registry about them, so it answered "not installed" and "Unloaded" for every
# profile while conversations resolved those same profiles perfectly well
# through ``llm.registry``. Two registries, one question, and the command was
# reading the wrong one. These are the doorway to the right one; ``cron.*``
# fronting the timekeeper is the same shape.
# ──────────────────────────────────────────────────────────────────────

def _llm_list(ctx, args: dict) -> Result:
    """Every configured profile, and every backend one could name.

    ``describe()`` has existed since the migration with exactly this row shape
    and no callers at all — it was written for this command and never wired.

    Backends carry their *display* name. A person picking one should read
    "LiteLLM (any provider)", which is what the file declares; the raw class
    name is an implementation detail that leaked all the way into the profile
    card. ``aliases`` comes along because a profile stores whatever backend
    name it was written with, and a migrated backend claims its predecessor's
    name — so a config saying ``LiteLLMService`` has to be resolvable to the
    class that replaced it before it can be displayed.
    """
    import llm

    return Result(data={
        "profiles": llm.describe(),
        "backends": [
            {"name": name, "display_name": display}
            for name, display in sorted(llm.backend_display_names().items())
        ],
        "aliases": llm.backend_aliases(),
        "default": llm.default_name(getattr(ctx, "config", None) or {}),
    })


def _llm_load(ctx, args: dict) -> Result:
    """Open one profile's box pool."""
    import llm

    name = args.get("name") or ""
    target = llm.brain(name)
    if target is None:
        return Result.failure(f"no LLM profile named {name!r}",
                              code=ERROR_NOT_FOUND)
    if not target.available:
        # The honest version of the message ``/llm`` used to invent. It said
        # "No backend is installed for <profile>" whenever the *service*
        # registry had no such key, which it never did and never will.
        return Result.failure(
            f"no backend installed for {name!r} (it names "
            f"{target.backend_name!r})", code=ERROR_NOT_FOUND)
    return Result(data=bool(target.load()))


def _llm_unload(ctx, args: dict) -> Result:
    """Close one profile's box pool."""
    import llm

    name = args.get("name") or ""
    target = llm.brain(name)
    if target is None:
        return Result.failure(f"no LLM profile named {name!r}",
                              code=ERROR_NOT_FOUND)
    target.unload()
    return Result(data=True)


def _service_call(ctx, args: dict) -> Result:
    """Invoke a method on a loaded service.

    Safe *because of* provenance, not despite it: the callee's own Requests
    are classified with the caller in the chain, so routing through a service
    launders nothing. Only methods the service lists in ``exports`` are
    reachable — anything else is internal.
    """
    name, method = args.get("name"), args.get("method")
    service = _service(ctx, name)
    if service is None:
        return Result.failure(f"service {name!r} is not loaded",
                              code=ERROR_NOT_FOUND)

    exports = getattr(service, "exports", None)
    if exports is not None and method not in exports:
        return Result.refusal(
            f"{name}.{method} is not exported; {sorted(exports)} are")

    fn = getattr(service, method or "", None)
    if not callable(fn):
        return Result.failure(f"{name} has no method {method!r}")
    try:
        return Result(data=fn(*(args.get("args") or ()),
                              **(args.get("kwargs") or {})))
    except Exception as exc:
        logger.exception("service_call failed")
        return Result.failure(f"{name}.{method} failed: {exc}")


def _tool_list(ctx, args: dict) -> Result:
    """Tools the current scope exposes."""
    registry = getattr(ctx, "tool_registry", None)
    if (bad := _need(registry, "the tool registry")) is not None:
        return bad
    if args.get("details"):
        tools = getattr(registry, "tools", None) or {}
        return Result(data=[
            _tool_details(ctx, tool)
            for _, tool in sorted(tools.items())
        ])
    return Result(data=sorted(registry.list_tools()))


def _tool_details(ctx, tool) -> dict:
    """Serializable schema, requirements, and editable settings for a tool."""
    schema = (tool.to_schema() or {}).get("function", {})
    return {
        "name": schema.get("name") or getattr(tool, "name", ""),
        "description": schema.get("description") or "",
        "parameters": schema.get("parameters") or {},
        "requires_services": list(
            getattr(tool, "requires_services", None) or []),
        "config_settings": [
            {
                "title": entry[0],
                "key": entry[1],
                "description": entry[2],
                "default": entry[3],
                "info": entry[4] if isinstance(entry[4], dict) else {},
                "current": redact(
                    entry[1], _config_value(ctx, entry[1], entry),
                    guess=True),
            }
            for entry in (getattr(tool, "config_settings", None) or [])
            if isinstance(entry, (list, tuple))
            and len(entry) == 5
            and not (
                isinstance(entry[4], dict)
                and entry[4].get("hidden") is True
            )
        ],
    }


def _tool_call(ctx, args: dict) -> Result:
    """Call another tool. The Request that makes a chain two links deep."""
    call = getattr(ctx, "call_tool", None)
    if (bad := _need(call, "tool-to-tool calls")) is not None:
        return bad
    name = args.get("name")
    try:
        kwargs = dict(args.get("kwargs") or {})
        command_origin = (
            getattr(ctx, "command_registry", None) is not None
            and not getattr(ctx, "current_tool_name", None)
        )
        if args.get("user_initiated") and command_origin:
            kwargs["_user_initiated"] = True
        outcome = call(name, **kwargs)
        if args.get("result"):
            return Result(data={
                "success": bool(getattr(outcome, "success", True)),
                "data": getattr(outcome, "data", outcome),
                "error": str(getattr(outcome, "error", "") or ""),
                "llm_summary": str(
                    getattr(outcome, "llm_summary", "") or ""),
                "attachment_paths": list(
                    getattr(outcome, "attachment_paths", None) or []),
            })
        return Result(ok=bool(getattr(outcome, "success", True)),
                      data=getattr(outcome, "data", outcome),
                      error=str(getattr(outcome, "error", "")))
    except Exception as exc:
        logger.exception("tool_call failed")
        return Result.failure(f"tool {name!r} failed: {exc}")


def _command_list(ctx, args: dict) -> Result:
    """Registered slash commands."""
    registry = getattr(ctx, "command_registry", None)
    if (bad := _need(registry, "the command registry")) is not None:
        return bad
    registered = (
        getattr(registry, "commands", None)
        or getattr(registry, "_commands", None)
        or {}
    )
    if not args.get("details"):
        return Result(data=sorted(registered))

    predicate = None
    if args.get("visible"):
        from plugins.command_registry import (
            frontend_command_filter,
        )

        runtime = _runtime(ctx)
        session = (getattr(runtime, "sessions", None) or {}).get(
            getattr(ctx, "session_key", None)
        )
        frontend = getattr(session, "frontend_name", None)
        predicate = frontend_command_filter(
            getattr(ctx, "config", None), frontend
        )

    commands = registry.visible_commands(predicate)
    form_context = registry.context(None)
    return Result(data=[{
        "name": command.name,
        "description": command.description,
        "category": command.category or "Other",
        "form": [{
            "name": step.name,
            "required": bool(step.required),
        } for step in command.form({}, form_context)],
    } for command in commands])


def _command_call(ctx, args: dict) -> Result:
    """Run a slash command in one shot.

    Two things this must not do, both of which it did before.

    It looked for ``registry.run`` or ``registry.execute``, and
    ``CommandRegistry`` has neither — only ``dispatch_dict`` — so the Request
    was unreachable however it was classified.

    ``command.call`` is itself unsafe, so a denied or unattended request never
    reaches this handler. Reaching it therefore carries the user's answer for
    this exact command and argument payload. When the command declares that
    those completed arguments require approval, pass that answer into dispatch
    so its declared nested-Request grant applies. Ordinary commands are *not*
    marked approved: authorizing use of the command surface is not a skeleton
    key for a command that declared no gated action.
    """
    registry = getattr(ctx, "command_registry", None)
    runner = getattr(registry, "dispatch_dict", None)
    if (bad := _need(runner, "running commands")) is not None:
        return bad

    name = args.get("name")
    command = (getattr(registry, "_commands", None) or {}).get(name)
    if command is None:
        return Result.failure(f"unknown command: /{name}",
                              code=ERROR_NOT_FOUND)
    command_args = args.get("args") or {}
    requires = getattr(command, "requires_approval", None)
    approved = bool(requires(command_args)) if callable(requires) else bool(
        getattr(command, "require_approval", False))

    try:
        return Result(data=runner(
            name, command_args,
            session_key=getattr(ctx, "session_key", None),
            _approved=approved,
        ))
    except Exception as exc:
        logger.exception("command_call failed")
        return Result.failure(f"command failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Agent, scheduling, events, pipeline, parsing, ledger.
# ──────────────────────────────────────────────────────────────────────

def _model_proceed(ctx, args: dict) -> Result:
    """Place the model call an escort is holding.

    Unlike every other handler this one resolves through a token rather than
    a static table, because what it invokes is a closure the kernel built for
    one particular call and will discard the moment the escort returns. Code
    that is not standing at the ``llm_call`` doorway holds no token, reaches
    no closure, and is refused — which is the correct answer, not an omission.
    """
    from ..hooks import phone

    dial = phone(args.get("token") or "")
    if dial is None:
        return Result.refusal(
            "llm.proceed is only available inside an llm_call hook")
    try:
        return Result(data=dial(args.get("request")))
    except Exception as exc:
        logger.exception("model_proceed failed")
        return Result.failure(f"model call failed: {exc}")


def _model_delta(ctx, args: dict) -> Result:
    """Carry one fragment of streamed assistant text out of a backend's box.

    Token-scoped exactly like ``llm.proceed``, and one-way: the answer says
    only whether it landed, never anything about the conversation. A backend
    that is not inside a call the kernel asked for holds no token and is
    refused.
    """
    from ..streams import deliver

    text = args.get("text")
    if not isinstance(text, str) or not text:
        return Result(data=False)
    if not deliver(args.get("token") or "", text):
        return Result.refusal(
            "llm.delta is only available inside an LLM backend's chat call")
    return Result(data=True)


def _agent_complete(ctx, args: dict) -> Result:
    """A model call.

    Its own Request, never a generic ``service.call``: keys, sockets and
    provider details stay kernel-side and the sandbox sees a prompt.

    Three ways to name the model, narrowest first. An explicit ``profile`` is
    a *model name* the kernel resolves — the same handle-not-the-thing move
    ``ModelRequest.llm`` makes, and what lets a background task select a
    cheap model without holding one. A ``session_key`` means "whatever is
    driving that session". Neither means the default profile.
    """
    from llm import default_brain
    from llm.registry import usable_brain

    config = getattr(ctx, "config", None) or {}
    brain = None
    profile = (args.get("profile") or "").strip()
    session_key = args.get("session_key")
    runtime = _runtime(ctx)
    if profile:
        brain = usable_brain(profile)
        if brain is None:
            return Result.failure(
                f"no usable LLM profile named {profile!r}")
    elif runtime is not None and session_key:
        try:
            from runtime.runtime_config import active_llm

            session = (getattr(runtime, "sessions", None) or {}).get(
                session_key
            )
            brain = active_llm(runtime, session)
        except Exception:
            logger.exception(
                "could not resolve the active LLM for %s", session_key
            )
    brain = brain or default_brain(config)
    if (bad := _need(brain, "an LLM")) is not None:
        return bad
    messages = args.get("messages")
    if not messages:
        prompt = args.get("prompt") or ""
        messages = [{"role": "user", "content": prompt}]
    try:
        from sandbox.guest.llm import LLMRequest

        response = brain.chat(LLMRequest(messages=list(messages)))
        return Result(ok=not response.is_error,
                      data={"content": response.content or "",
                            "tool_calls": list(response.tool_calls or []),
                            "llm": getattr(brain, "name", "") or ""},
                      error=str(response.error or ""))
    except Exception as exc:
        logger.exception("agent_complete failed")
        return Result.failure(f"model call failed: {exc}")


_SPAWN_POLL = 0.25


def _subagents(ctx):
    """The kernel's subagent registry, or None."""
    return getattr(_runtime(ctx), "subagents", None)


def _spawn_owner(ctx):
    """Who is owed this child's report, and in which conversation.

    A session key is the owner. Code with no session — a scheduled task, a
    script the kernel started — owns nothing, and its children are collected
    explicitly or not at all.
    """
    key = getattr(ctx, "session_key", None) or None
    session = (getattr(_runtime(ctx), "sessions", None) or {}).get(key)
    return key, getattr(session, "conversation_id", None)


def _agent_spawn(ctx, args: dict) -> Result:
    """Run an agent on a prompt, in its own conversation.

    ``wait=False`` answers with a handle the caller collects later, which is
    what makes a fan-out expressible from a script. ``wait=True`` answers with
    the report and does the waiting here.

    ``profile`` is an agent profile *name*, resolved kernel-side — the same
    handle-not-the-thing move ``ModelRequest.llm`` makes. It is how a caller
    spawns a child that may do less than it can, and it cannot widen: the
    profiles are the user's own config.
    """
    from .. import provenance

    registry = _subagents(ctx)
    if (bad := _need(registry, "subagents")) is not None:
        return bad
    owner, owner_cid = _spawn_owner(ctx)
    try:
        handle = registry.spawn(
            args.get("prompt") or "",
            title=args.get("title") or "Subagent",
            attachments=args.get("attachments"),
            timeout_seconds=args.get("timeout_seconds"),
            owner=owner,
            owner_conversation_id=owner_cid,
            user_id=int(getattr(ctx, "user_id", 1) or 1),
            # A profile the user configured, naming tools the user installed.
            # Choosing among them can only ever narrow what the child may do,
            # so this needs no classification of its own — and naming an
            # unknown one raises rather than silently running unrestricted.
            profile=args.get("profile") or None,
        )
    except (PermissionError, ValueError, FileNotFoundError) as exc:
        return Result.failure(str(exc))
    except Exception as exc:
        logger.exception("agent_spawn failed")
        return Result.failure(f"could not start the subagent: {exc}")

    if not args.get("wait", True):
        return Result(data=handle.report())

    # Waiting in slices for the reason ``_script_run`` does: this handler makes
    # no Requests while it waits, so cancellation cannot reach it any other
    # way, and a cancelled caller would otherwise hold a pool worker until the
    # child's own deadline for an answer nobody will read.
    caller = provenance.current()
    while True:
        reports = registry.collect([handle.id], timeout=_SPAWN_POLL)
        report = reports[0] if reports else handle.report()
        if report["state"] != "running":
            return Result(ok=report["ok"], data=report,
                          error=report["error"] or "")
        if caller is not None and caller.abandoned:
            registry.cancel(handle.id)
            return Result.failure(
                f"subagent '{handle.title}' was cancelled "
                f"(partial transcript in conversation #{handle.conversation_id})")


def _agent_collect(ctx, args: dict) -> Result:
    """Take the reports of subagents this session started."""
    registry = _subagents(ctx)
    if (bad := _need(registry, "subagents")) is not None:
        return bad
    owner, _ = _spawn_owner(ctx)
    timeout = args.get("timeout")
    try:
        return Result(data=registry.collect(
            args.get("ids"), owner=owner,
            timeout=None if timeout is None else float(timeout)))
    except Exception as exc:
        logger.exception("agent_collect failed")
        return Result.failure(f"could not collect subagents: {exc}")


def _agent_stop(ctx, args: dict) -> Result:
    """Cancel a running subagent."""
    registry = _subagents(ctx)
    if (bad := _need(registry, "subagents")) is not None:
        return bad
    return Result(data=bool(registry.cancel(args.get("id") or "")))


def _agent_schedule(ctx, args: dict) -> Result:
    """Run a subagent later, on a schedule.

    A Timekeeper job firing the kernel's own spawn channel — which is why this
    is a Request of its own rather than ``cron.create`` with a hand-built job
    definition: the channel and payload shape are the kernel's, and a caller
    that had to spell them itself could spell them wrong.
    """
    from events.event_channels import SUBAGENT_SPAWN

    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return Result.failure("a scheduled subagent needs a prompt")
    cron = (args.get("cron") or "").strip()
    if not cron:
        return Result.failure("a scheduled subagent needs a cron expression",
                              code=ERROR_INVALID_ARGUMENT)
    title = (args.get("title") or "Scheduled subagent").strip()
    name = (args.get("name") or "").strip() or _job_name(title)

    try:
        schedule = _schedule_def(cron, bool(args.get("one_time")))
    except Exception as exc:
        logger.exception("agent_schedule failed")
        return Result.failure(f"bad cron expression: {exc}",
                              code=ERROR_INVALID_ARGUMENT)
    job = {**schedule, "channel": SUBAGENT_SPAWN, "enabled": True,
           "payload": {"title": title, "prompt": prompt,
                       "attachments": list(args.get("attachments") or [])}}
    try:
        keeper.create_job(name, job)
    except Exception as exc:
        logger.exception("agent_schedule failed")
        return Result.failure(f"could not schedule the subagent: {exc}")
    return Result(data={"name": name, "title": title, **schedule})


def _schedule_def(cron: str, one_time: bool) -> dict:
    """A Timekeeper schedule from a cron expression.

    A one-time job wants an absolute ``run_at`` rather than a cron, so the
    next match is resolved here and the cron is discarded — a person saying
    "at 9am tomorrow" spells it as a cron and means one firing.
    """
    from datetime import datetime

    from croniter import croniter

    if one_time:
        run_at = croniter(cron, datetime.now().astimezone()).get_next(datetime)
        return {"run_at": run_at.isoformat(), "cron": None, "one_time": True}
    croniter(cron)  # raises on a malformed expression
    return {"cron": cron, "run_at": None, "one_time": False}


def _job_name(title: str) -> str:
    """A Timekeeper job name from a human title."""
    import re

    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "subagent"


def _timekeeper(ctx):
    """The scheduling service, or None."""
    return _service(ctx, "timekeeper")


def _cron_list(ctx, args: dict) -> Result:
    """Every scheduled job."""
    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    return Result(data=keeper.list_jobs())


def _cron_get(ctx, args: dict) -> Result:
    """One scheduled job."""
    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    return Result(data=keeper.get_job(args.get("name")))


def _cron_create(ctx, args: dict) -> Result:
    """Add a job."""
    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    try:
        return Result(data=keeper.create_job(args.get("name"),
                                             args.get("job") or {}))
    except Exception as exc:
        logger.exception("cron_create failed")
        return Result.failure(f"could not create job: {exc}")


def _cron_update(ctx, args: dict) -> Result:
    """Change a job."""
    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    try:
        return Result(data=keeper.update_job(args.get("name"),
                                             args.get("patch") or {}))
    except Exception as exc:
        logger.exception("cron_update failed")
        return Result.failure(f"could not update job: {exc}")


def _cron_remove(ctx, args: dict) -> Result:
    """Delete a job."""
    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    return Result(data=bool(keeper.remove_job(args.get("name"))))


def _cron_enable(ctx, args: dict) -> Result:
    """Enable or disable a job."""
    keeper = _timekeeper(ctx)
    if (bad := _need(keeper, "the timekeeper")) is not None:
        return bad
    return Result(data=keeper.enable_job(args.get("name"),
                                         bool(args.get("enabled", True))))


def _event_emit(ctx, args: dict) -> Result:
    """Publish on a bus channel, without lending the bus this guest's thread.

    ``EventBus.emit`` runs handlers on the caller's thread, and the caller here
    is a guest that may be holding its box's single call lock — a service
    emitting from ``poll`` is exactly that. Any subscriber calling back into a
    service then blocks on a lock the publisher will not release until this
    Request answers. ``sandbox/events.publish`` is the one place that breaks
    the cycle; see its module docstring for the outage it was written for.
    """
    return Result(data=_publish_event(args.get("channel"),
                                      args.get("payload")))


def _event_request(ctx, args: dict) -> Result:
    """Publish and wait for one answer."""
    timeout, bad = float_arg(args, "timeout", 120.0, lo=0.0)
    if bad is not None:
        return bad
    try:
        from events.event_bus import bus
        return Result(data=bus.request(args.get("channel"),
                                       args.get("payload") or {},
                                       timeout=timeout))
    except Exception as exc:
        logger.exception("event_request failed")
        return Result.failure(f"request failed: {exc}", retryable=True)


# ──────────────────────────────────────────────────────────────────────
# Frontends: carrying what a person did into the state machine.
#
# Every one of these resolves through a token to the calling frontend's own
# adapter, so the authority is the frontend's identity rather than anything
# the Request says about itself. A caller that is not a loaded frontend holds
# no token, reaches no adapter, and is refused — which is the correct answer
# rather than an omission, exactly as it is for ``llm.proceed``.
# ──────────────────────────────────────────────────────────────────────

def _at_desk(args: dict):
    """The adapter behind a frontend Request, or a refusal explaining why not."""
    from ..frontends import desk

    adapter = desk(args.get("token") or "")
    if adapter is None:
        return None, Result.refusal(
            "sdk.frontend is only available inside a loaded frontend")
    return adapter, None


def _the_sandbox(what: str):
    """The process's sandbox, or a refusal saying it could not be reached.

    Five handlers need this and each had its own copy of the same four lines.
    Guarded rather than left to raise because ``get_sandbox`` *builds* one on
    demand and the build can fail — an import error in the plugin roots, say —
    and "the sandbox is not available" is a far more useful thing for a plugin
    author to read than a traceback from the interpreter's generic net.
    """
    from ..bridge import get_sandbox

    try:
        return get_sandbox(), None
    except Exception as exc:
        logger.exception("%s could not reach the sandbox", what)
        return None, Result.failure(f"the sandbox is not available: {exc}")


def _not_yours(adapter, session_key: str) -> Result | None:
    """Refuse a session belonging to a *different* frontend, else None.

    The token says which frontend is asking; this says which sessions it may
    say things about. Both halves are needed, and only the first was there:
    ``mark_attended`` takes any string, so one frontend could declare another's
    session attended — and attendance is what decides whether an unsafe Request
    gets a dialog rather than a refusal.

    A session that does not exist yet is *allowed*, deliberately, and for the
    reason ``_live_session_keys`` includes untagged ones: ``_tag_session``
    stamps ownership when a frontend first submits, so a brand-new thread has
    no owner and refusing it would mean a frontend could never act on the first
    request of a conversation.
    """
    if not session_key:
        return Result.refusal("name the session to act as",
                              code=ERROR_INVALID_ARGUMENT)
    runtime = getattr(adapter, "runtime", None)
    session = (getattr(runtime, "sessions", None) or {}).get(session_key)
    owner = (None if session is None
             else getattr(session, "frontend_name", None))
    if owner is not None and owner != getattr(adapter, "name", ""):
        return Result.refusal(
            f"session {session_key} belongs to the {owner} frontend")
    return None


def _prepare_attachment(ctx, args: dict):
    """Build the richer send_attachment payload, or None for the plain path.

    Two jobs, both of which exist because a frontend's files arrive over a
    *transport* rather than off the user's disk.

    **Ingestion.** ``attachments.cache.save`` is the front door for those files:
    it names them stably, keeps the folder under its size cap, and — the part
    that matters — puts them in a watched directory, so the pipeline extracts,
    chunks and indexes them like anything else. A sandboxed frontend cannot
    call it (kernel module) and cannot be handed the folder either, since
    writing straight in would skip the eviction the cap depends on. So the
    guest downloads into scratch it got from ``sdk.fs.temp`` and names that
    path here; the bytes never cross the boundary, which is the only way a
    50 MB file gets in at all — one wire message holds ~11 MB.

    **Metadata.** ``caption``, ``file_name`` and ``is_photo`` are what the
    native frontends have always put on the action and the plain
    ``submit_attachment`` entry point has no room for.

    Returns None when the guest asked for neither, so the original path stays
    byte-for-byte what it was.
    """
    extras = {key: args.get(key) for key in
              ("file_name", "caption", "is_photo")}
    if not args.get("ingest") and not any(extras.values()):
        return None

    path = Path(str(args.get("path") or ""))
    file_name = str(extras.get("file_name") or "") or path.name
    if args.get("ingest"):
        from attachments.cache import save as save_attachment

        try:
            data = path.read_bytes()
        except OSError as exc:
            return Result.failure(f"could not read {path.name}: {exc}")
        try:
            cap = float((getattr(ctx, "config", None) or {}).get(
                "attachment_cache_size_gb", 2.0))
        except (TypeError, ValueError):
            cap = 2.0
        try:
            path = save_attachment(file_name, data, cap)
        except OSError as exc:
            return Result.failure(f"could not cache {file_name}: {exc}")
        # Scratch, and the kernel is what allocated it. Failing to clean up is
        # not worth failing the message a person just sent.
        try:
            Path(str(args.get("path"))).unlink()
        except OSError:
            logger.debug("could not remove the ingested temp file %s",
                         args.get("path"))

    return {
        "path": str(path),
        "extension": (str(args.get("extension") or "")
                      or path.suffix.lstrip(".")),
        "caption": str(extras.get("caption") or ""),
        "file_name": file_name,
        "is_photo": bool(extras.get("is_photo")),
    }


def _drive(adapter, work, what: str) -> Result:
    """Run something that drives the state machine, off the box's thread.

    This is the one shape every re-entrant frontend Request has, and it is a
    deadlock unless it is handled in exactly one place.

    A resident frontend calls in from ``poll``, which holds its box's single
    call lock for the duration of the Request. ``handle_action`` runs the
    turn *synchronously* — and a turn renders: tool status, messages, a form.
    Every one of those is a ``box.call("render", ...)`` back into the box that
    is still sitting inside ``poll`` waiting for this answer. The render
    blocks on the lock, the turn never finishes, the Request never returns,
    and the frontend is frozen for good.

    ``submit`` was given a thread for this reason. ``resolve`` and ``cancel``
    were not, and both reach ``handle_action`` by exactly the same path —
    ``resolve_approval`` and ``cancel`` are ``submit`` with a different action
    type. Answering an approval from an inline button therefore froze the
    transport every time. Hoisted here so a sixth entry point cannot be added
    without inheriting the answer.

    The REPL escaped it by luck rather than design: in ``approving_request``
    it answers through ``submit_text``, which was already detached. It is the
    *bus-driven* approval path — the one a rich frontend with buttons uses —
    that had no protection.

    Returns a Result when the work was detached, or None to run it inline.
    Detaching costs the caller a real answer, so it reports acceptance: see
    ``_frontend_resolve`` for what a frontend should ask instead.
    """
    if not getattr(adapter, "background_submit", False):
        return None

    import threading

    def run():
        """Drive it, and never let the failure escape onto a bare thread."""
        try:
            work()
        except Exception:
            logger.exception("detached frontend %s failed", what)

    threading.Thread(
        target=run, daemon=True,
        name=f"{getattr(adapter, 'name', 'frontend')}-{what}",
    ).start()
    return Result(data=True)


def _frontend_submit(ctx, args: dict) -> Result:
    """Hand a person's input to the state machine.

    The three kinds go to three different native entry points because they
    coerce differently — text may be a slash command, an attachment has to be
    parsed and staged — and collapsing them here would lose that.
    """
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    kind = args.get("input_kind") or "text"

    if kind == "attachment":
        prepared = _prepare_attachment(ctx, args)
        if isinstance(prepared, Result):
            return prepared

    def submit():
        if kind == "text":
            return adapter.submit_text(session_key, args.get("text") or "")
        elif kind == "attachment":
            if prepared is not None:
                from state_machine.action_map import ACTION_SEND_ATTACHMENT

                return adapter.submit(
                    session_key, ACTION_SEND_ATTACHMENT, prepared)
            return adapter.submit_attachment(
                session_key, args.get("path") or "",
                args.get("extension") or None)
        elif kind == "action":
            return adapter.submit(session_key,
                                  args.get("action_type") or "",
                                  args.get("payload"))
        raise ValueError(f"unknown submit kind {kind!r}")

    if (detached := _drive(adapter, submit, "submit")) is not None:
        return detached

    try:
        result = submit()
    except Exception as exc:
        logger.exception("frontend_submit failed")
        return Result.failure(f"submit failed: {exc}")

    # A RuntimeResult is a live object. What a frontend needs back is whether
    # it landed, and the rest reaches it as a render call like everything else.
    return Result(data=bool(getattr(result, "ok", result is not None)))


def _frontend_cancel(ctx, args: dict) -> Result:
    """Stop whatever a session is doing.

    ``cancel`` is ``submit`` with a different action type, so it drives the
    state machine and must be detached for the same reason.
    """
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")

    def cancel():
        return adapter.cancel(session_key)

    if (detached := _drive(adapter, cancel, "cancel")) is not None:
        return detached
    try:
        cancel()
        return Result(data=True)
    except Exception as exc:
        logger.exception("frontend_cancel failed")
        return Result.failure(f"cancel failed: {exc}")


def _frontend_bind(ctx, args: dict) -> Result:
    """Say whose data a session is. Returns the user id.

    Which of the two native paths runs is decided by whether an external
    identity was named, not by the plugin choosing — so a frontend cannot
    upgrade a session to an arbitrary user by picking the wrong call.
    """
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    external_id = args.get("external_id")
    if external_id is None:
        return Result(data=adapter.bind_session(session_key))
    return Result(data=adapter.identify(
        session_key, external_id, args.get("config") or None,
        user_type=str(args.get("user_type") or "user")))


def _frontend_attend(ctx, args: dict) -> Result:
    """Say whether a person is watching a session."""
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    if (refusal := _not_yours(adapter, session_key)) is not None:
        return refusal
    if args.get("present"):
        adapter.mark_attended(session_key)
    else:
        adapter.mark_unattended(session_key)
    return Result(data=True)


def _frontend_resolve(ctx, args: dict) -> Result:
    """Answer a pending approval by id, or the session's next one.

    Answering drives the state machine — the approved action runs, and the
    turn it belongs to carries on — so this detaches like ``submit``. That
    froze every button-answered approval on a rich frontend until it did:
    the turn's first render blocked on the box lock the caller was holding.

    The answer stays truthful across the detach. Whether the approval is
    still there is settled *here*, synchronously, because that is a lookup
    rather than a turn; only the driving is handed to a thread. So ``False``
    still means "there was nothing to answer" — which is what a frontend
    branches on to decide whether the person's text was a yes/no or an
    ordinary message.
    """
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    request_id = str(args.get("request_id") or "")
    value = args.get("value")

    if not adapter.is_approval_pending(session_key, request_id or None):
        return Result(data=False)

    def resolve():
        if request_id:
            return adapter.resolve_approval(session_key, request_id, value)
        return adapter.resolve_next_approval(session_key, value)

    if (detached := _drive(adapter, resolve, "resolve")) is not None:
        return detached
    # Inline only when the frontend did not ask to be detached — and inline
    # means the whole turn runs here, tools and all. That is foreign code, so
    # it keeps a guard where the lookup above does not.
    try:
        return Result(data=bool(resolve()))
    except Exception as exc:
        logger.exception("frontend_resolve failed")
        return Result.failure(f"resolve failed: {exc}")


def _frontend_pending(ctx, args: dict) -> Result:
    """The id of the approval a session is waiting on, or None.

    Asked rather than remembered. A frontend knows an approval exists — it was
    handed one to render — but not when it stops existing: another frontend can
    answer it, or it can time out. A frontend acting on a stale record would
    swallow the next thing a person typed as a yes/no.

    With ``details``, the *question* rather than its id, tagged with the render
    kind that would have carried it::

        {"kind": "approval",   "payload": {id, title, body, type, enum, …}}
        {"kind": "form_field", "payload": {name, field, collected, display}}
        None

    **A render is an event, and events are not re-sent on demand.** A frontend
    that was not connected when the question was asked — a browser that
    reloaded, a transport that dropped — cannot get back to it any other way,
    and an id alone only buys the ability to answer a question nobody can read.
    So this hands back the same projections the two renders made, and a
    reconnecting client shows the real dialog instead of a reconstruction of one.

    Both kinds are here because they are one thing: ``runtime.request_input``
    and a suspended callable's form are both "this session is blocked until a
    person answers", and a client that restores one and not the other still
    strands people. Approvals are asked for first because they nest — a form
    step can raise one, and the inner question is the one to answer.
    """
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    detailed = bool(args.get("details"))

    order = getattr(adapter, "_pending_approval_order", None) or {}
    waiting = list(order.get(session_key) or [])

    if adapter.has_pending_approval(session_key):
        if not detailed:
            # The id is enough to answer and only enough to answer — the same
            # projection the ``approval`` render makes.
            return Result(data=waiting[0] if waiting else True)
        request = _pending_approval(adapter, session_key, waiting)
        if request is None:
            return Result(data=None)
        from ..frontends import project_approval
        return Result(data={"kind": "approval",
                            "payload": project_approval(request)})

    # Nothing registered. That is not the same as nothing waiting: the table is
    # process memory and the phase stack is persisted, so ask the session
    # itself before concluding a client should take its dialog down.
    if (request := _approval_from_phase(adapter, session_key)) is not None:
        if not detailed:
            return Result(data=getattr(request, "id", "") or True)
        from ..frontends import project_approval
        return Result(data={"kind": "approval",
                            "payload": project_approval(request)})

    # Without ``details`` this Request has only ever spoken about approvals,
    # and answering None for a session sitting on a form is what its callers
    # already expect. Widening that silently would change what an existing
    # frontend believes it is holding.
    return Result(data=_pending_form(adapter, session_key) if detailed else None)


def _pending_approval(adapter, session_key: str, waiting: list):
    """The question this session is blocked on, or None.

    The *registered* object first: it is the one ``resolve`` will answer, so
    projecting anything else risks handing back a question that does not match
    the id travelling with it. Order before the unordered fallback, since that
    is the queue ``resolve_next_approval`` works down.

    Then the phase stack, which is the authority the registration is only a
    cache of — see :func:`_approval_from_phase`.
    """
    registered = (getattr(adapter, "_pending_approvals", None) or {}).get(session_key) or {}
    for request_id in waiting:
        request = registered.get(request_id)
        if request is not None and not getattr(request, "is_resolved", False):
            return request
    for request in registered.values():
        if not getattr(request, "is_resolved", False):
            return request
    return _approval_from_phase(adapter, session_key)


def _approval_from_phase(adapter, session_key: str):
    """Rebuild the question from the session's own phase frame, or None.

    **The registration table is process memory; the phase stack is persisted.**
    A kernel restart, or a frontend loaded after its session was restored,
    leaves the table empty while the stack still says the session is blocked —
    and the bus announcement that would have filled it fired once, before there
    was anything live to catch it. Answering ``None`` there tells a client to
    take down a dialog for a question that is still waiting, which is the exact
    failure this Request exists to prevent.

    **Registering what it rebuilds, which is why a read writes.** An id nobody
    has registered is an id ``frontend.resolve`` refuses: it settles existence
    against this same table before it drives anything, so handing back a
    projection without one would answer "here is your question" and then "no
    such question" to the very next call.

    A live object beats a rebuild when the process still holds one. A tool
    blocked inside ``request_input`` is waiting on *that* request, and the
    rebuild carries ``render_result_on_resolve``, whose render path waits on the
    guest lock the blocked call is holding — answering it would deadlock against
    the call it was answering.
    """
    runtime = getattr(adapter, "runtime", None)
    build = getattr(adapter, "_current_approval_request", None)
    if runtime is None or build is None:
        return None
    try:
        from state_machine.conversation_phases import PHASE_APPROVING_REQUEST

        frame = runtime.get_session(session_key).cs.frame
        if frame is None or getattr(frame, "phase", "") != PHASE_APPROVING_REQUEST:
            return None
        held = (getattr(runtime, "_approval_requests", None) or {}).get(
            (getattr(frame, "data", None) or {}).get("request_id") or "")
        request = (held if held is not None and not held.is_resolved
                   else build(session_key))
        if request is None:
            return None
        adapter._register_pending_approval(session_key, request)
        return request
    except Exception:
        logger.exception("frontend_pending could not rebuild the pending approval")
        return None


def _pending_form(adapter, session_key: str):
    """The form step a session is sitting on, drawn as ``render_form_field`` drew it.

    Built through ``decorate_form`` rather than read off the phase frame, so
    the ``display`` block a client renders is the same one the live path
    produces — a second spelling of it would drift, and the drift would only
    show up after a reload, which is the one moment nobody is watching for it.
    """
    runtime = getattr(adapter, "runtime", None)
    if runtime is None:
        return None
    try:
        from runtime.dispatch import decorate_form
        from runtime.session import RuntimeResult

        out = RuntimeResult()
        decorate_form(runtime.get_session(session_key), out)
    except Exception:
        logger.exception("frontend_pending could not read the pending form")
        return None
    return {"kind": "form_field", "payload": dict(out.form)} if out.form else None


#: Requests ``frontend.act`` will not carry. ``frontend.act``/``collect`` would
#: recurse; the ``http.*`` family belongs to the *transport*, which is the
#: frontend's own possession rather than anything a session may reach — a
#: client closing the socket it is talking over is the shape to avoid.
_ACT_REFUSED = frozenset({FRONTEND_ACT, FRONTEND_COLLECT,
                          HTTP_DRAIN, HTTP_RESPOND, HTTP_PUSH, HTTP_CLOSE})


def _frontend_act(ctx, args: dict) -> Result:
    """Run one Request as a session this frontend owns. Returns a handle.

    A frontend box is rooted ``frontend:<name>``, which names no session, so
    ``attended_now`` answers False for it forever and every unsafe Request it
    makes is refused rather than asked — correctly, since there is nobody a
    dialog could be drawn for. But a frontend serving an authenticated request
    is not acting on its own initiative: somebody clicked something. This says
    so, by rooting the Request at the session instead.

    That is the whole of the widening, and it is self-limiting. Rooting at a
    session makes ``attended_now`` ask ``runtime.is_attended``, which reads
    what this same frontend declared through ``frontend.attend``. Say nobody
    is watching and the authority goes with it. Rooting at ``user`` would have
    been unconditionally attended and would have taken the decision away from
    the mechanism built to hold it.

    Chain *and* context both move, unlike ``PersistentBox.call(for_session=)``,
    which deliberately moves only the context. Its argument is that a service
    standing at a hook doorway is still acting on its own initiative; a
    frontend serving a person is not, so the answer differs. The context has to
    move regardless — ``conv.load`` and its neighbours read ``ctx.session_key``
    and would otherwise act on nothing and report success.

    Detached, and that is correctness rather than speed: see ``Sandbox.act``.
    """
    from ..policy import Chain, attended_now

    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    if (refusal := _not_yours(adapter, session_key)) is not None:
        return refusal

    request_type = str(args.get("request_type") or "")
    if request_type not in ALL_TYPES:
        return Result.failure(f"no such Request type: {request_type!r}",
                              code=ERROR_INVALID_ARGUMENT)
    if request_type in _ACT_REFUSED:
        return Result.refusal(f"{request_type} cannot be run through act")

    inner = dict(args.get("args") or {})
    if request_type.startswith("frontend."):
        # The *kernel* supplies the identity, never the caller: these resolve
        # an adapter by token, and one arriving in the args would be somebody
        # else's claim about who they are. We already know — it is the token
        # that got us here.
        inner["token"] = args.get("token")

    name = getattr(adapter, "name", "") or "frontend"
    sandbox, refusal = _the_sandbox("frontend_act")
    if refusal is not None:
        return refusal

    chain = Chain(root=session_key).push(f"frontend:{name}")

    # A dedicated permission selector has to be able to leave Lockdown. The
    # typed `/mode ask` command already has exactly that standing, but an HTTP
    # control reaches the same Request through ``frontend.act`` and otherwise
    # arrives as an ordinary unsafe action — which Lockdown refuses before the
    # user can escape it. Give *only* an explicit, attended switch back to Ask
    # the same narrow provenance as the typed command. Lockdown is already
    # safe because it tightens; YOLO deliberately keeps the ordinary frontend
    # chain so it still raises a real approval dialog.
    if (request_type == SESSION_SET_MODE
            and str(inner.get("mode") or "").strip().lower() == "ask"
            and attended_now(chain, runtime=getattr(adapter, "runtime", None))):
        chain = Chain(root="user:command").push("mode")

    return Result(data=sandbox.act(
        Request(request_type, inner),
        chain,
        sandbox.interpreter.context_for_session(session_key),
        owner=name))


def _frontend_collect(ctx, args: dict) -> Result:
    """Take the answer to a ``frontend.act``, or None while it is still going.

    The Result comes back as a plain dict rather than being unwrapped, because
    a refusal is an ordinary answer here: the frontend's job is to forward it
    to whoever asked, not to treat it as its own failure.
    """

    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    sandbox, refusal = _the_sandbox("frontend_collect")
    if refusal is not None:
        return refusal

    # Owned by the *frontend*, not by the session it was run as: one plugin,
    # one box, one memory — two of its threads sharing a namespace is not a
    # boundary worth drawing. What matters is that another frontend's handle
    # answers None rather than a result.
    outcome = sandbox.collect_act(str(args.get("handle") or ""),
                                  getattr(adapter, "name", "") or "frontend")
    return Result(data=None if outcome is None else outcome.to_dict())


def _console_read(ctx, args: dict) -> Result:
    """Take the next line a person typed, if one has arrived.

    Non-blocking on purpose. The kernel's reader thread is what waits; if this
    blocked, it would hold the calling box for the duration and the frontend
    could not render until the user pressed return.
    """
    from ..console import CONSOLE

    token = args.get("token") or ""
    if not token or CONSOLE.owner != token:
        return Result.refusal(
            "the console belongs to another frontend, or to none")
    try:
        return Result(data=CONSOLE.read_line())
    except EOFError as exc:
        # Not a refusal: nothing was denied, the input simply ended. A frontend
        # that lets this propagate out of poll() stops itself, which is what
        # end-of-input on a pipe should do.
        return Result.failure(str(exc))


def _console_write(ctx, args: dict) -> Result:
    """Put a line on the console."""
    from ..console import CONSOLE

    token = args.get("token") or ""
    if not token or CONSOLE.owner != token:
        return Result.refusal(
            "the console belongs to another frontend, or to none")
    CONSOLE.write(str(args.get("text") or ""),
                  end=str(args.get("end", "\n")))
    return Result(data=True)


def _server(args: dict):
    """The server, if this caller is the frontend holding the port.

    One check for four handlers, because the ownership question is identical
    and writing it out four times is how the fourth copy comes to differ. The
    refusal is a ``Result`` rather than an exception so a frontend whose claim
    was lost mid-poll learns it the ordinary way.
    """
    from ..http_server import SERVER

    token = args.get("token") or ""
    if not token or SERVER.owner != token:
        return None, Result.refusal(
            "this port belongs to another frontend, or to none")
    return SERVER, None


def _http_drain(ctx, args: dict) -> Result:
    """Take the requests that have arrived, if any.

    Non-blocking on purpose, the same as ``console.read``: the kernel's
    listener thread is what waits. If this blocked it would hold the calling
    box for the duration and the frontend could not render — which for an SSE
    transport means it could not answer the very request it is blocked on.
    """
    server, refusal = _server(args)
    if refusal is not None:
        return refusal
    try:
        limit = int(args.get("limit") or 0)
    except (TypeError, ValueError):
        return Result.failure("limit must be a number",
                              code=ERROR_INVALID_ARGUMENT)
    return Result(data=server.drain(limit))


def _http_respond(ctx, args: dict) -> Result:
    """Answer a request, or open it as an event stream."""
    server, refusal = _server(args)
    if refusal is not None:
        return refusal
    headers = args.get("headers") or {}
    if not isinstance(headers, dict):
        return Result.failure("headers must be a mapping",
                              code=ERROR_INVALID_ARGUMENT)
    try:
        status = int(args.get("status") or 200)
    except (TypeError, ValueError):
        return Result.failure("status must be a number",
                              code=ERROR_INVALID_ARGUMENT)
    # The body is passed through rather than coerced: bytes are how a frontend
    # serves an image or a font, and ``protocol.pack`` already carries them
    # across the wire. Coercing here turned every binary asset into mojibake.
    body = args.get("body")
    if not isinstance(body, (str, bytes, bytearray)):
        body = "" if body is None else str(body)
    ok = server.respond(str(args.get("request_id") or ""), status=status,
                        headers={str(k): str(v) for k, v in headers.items()},
                        body=body, stream=bool(args.get("stream")))
    if not ok:
        return Result.failure("no request is open under that id",
                              code=ERROR_NOT_FOUND)
    return Result(data=True)


def _http_push(ctx, args: dict) -> Result:
    """Write one frame to an open stream."""
    server, refusal = _server(args)
    if refusal is not None:
        return refusal
    ok = server.push(str(args.get("request_id") or ""),
                     str(args.get("data") or ""),
                     event=str(args.get("event") or ""),
                     ident=str(args.get("ident") or ""))
    if not ok:
        # Deliberately a failure rather than a silent success: a client that
        # went away is the ordinary end of a stream, and a frontend that never
        # hears about it goes on rendering a turn into a closed socket.
        return Result.failure("no stream is open under that id",
                              code=ERROR_NOT_FOUND)
    return Result(data=True)


def _http_close(ctx, args: dict) -> Result:
    """End a reply."""
    server, refusal = _server(args)
    if refusal is not None:
        return refusal
    if not server.close(str(args.get("request_id") or "")):
        return Result.failure("no request is open under that id",
                              code=ERROR_NOT_FOUND)
    return Result(data=True)


def _task_enqueue(ctx, args: dict) -> Result:
    """Queue pipeline work."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    for path in args.get("paths") or []:
        db.enqueue_task(args.get("name"), path)
    return Result(data=True)


def _task_status(ctx, args: dict) -> Result:
    """Where one task stands for one path."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    return Result(data=db.get_task_status(args.get("name"),
                                          args.get("path")))


def _task_output(ctx, args: dict) -> Result:
    """Read a task's output table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    return Result(data=_rows(db.get_task_output(args.get("name"),
                                                args.get("path"))))


def _task_list(ctx, args: dict) -> Result:
    """Registered task names or structured management metadata."""
    orchestrator = getattr(ctx, "orchestrator", None)
    tasks = getattr(orchestrator, "tasks", None)
    if (bad := _need(tasks, "the task orchestrator")) is not None:
        return bad
    if not args.get("details"):
        return Result(data=sorted(tasks))

    db = _db(ctx)
    counts = {}
    if db is not None:
        counts.update((db.get_system_stats() or {}).get("tasks", {}) or {})
        getter = getattr(db, "get_run_stats", None)
        if getter is not None:
            counts.update(getter() or {})
    timekeeper = _service(ctx, "timekeeper")
    jobs = (
        timekeeper.list_jobs()
        if timekeeper is not None
        and getattr(timekeeper, "loaded", False)
        and callable(getattr(timekeeper, "list_jobs", None))
        else {}
    )
    paused = set(getattr(orchestrator, "paused", None) or [])
    return Result(data=[
        {
            "name": name,
            "description": getattr(task, "description", "") or "",
            "trigger": getattr(task, "trigger", "path"),
            "counts": dict(counts.get(name) or {}),
            "paused": name in paused,
            "requires_services": list(
                getattr(task, "requires_services", None) or []),
            "trigger_channels": list(
                getattr(task, "trigger_channels", None) or []),
            "event_payload_schema": dict(
                getattr(task, "event_payload_schema", None) or {}),
            "schedule_count": sum(
                1 for job in jobs.values()
                if (job.get("channel") or "") in set(
                    getattr(task, "trigger_channels", None) or [])
            ),
            "config_settings": [
                {
                    "title": entry[0],
                    "key": entry[1],
                    "description": entry[2],
                    "default": entry[3],
                    "info": entry[4] if isinstance(entry[4], dict) else {},
                    "current": redact(
                        entry[1], _config_value(ctx, entry[1], entry),
                        guess=True),
                }
                for entry in (getattr(task, "config_settings", None) or [])
                if isinstance(entry, (list, tuple))
                and len(entry) == 5
                and not (
                    isinstance(entry[4], dict)
                    and entry[4].get("hidden") is True
                )
            ],
        }
        for name, task in sorted(tasks.items())
    ])


def _task_graph(ctx, args: dict) -> Result:
    orchestrator = getattr(ctx, "orchestrator", None)
    graph = getattr(orchestrator, "dependency_pipeline_graph", None)
    if (bad := _need(graph, "the dependency pipeline")) is not None:
        return bad
    return Result(data=graph())


def _task_pause(ctx, args: dict) -> Result:
    orchestrator = getattr(ctx, "orchestrator", None)
    tasks = getattr(orchestrator, "tasks", None)
    if (bad := _need(tasks, "the task orchestrator")) is not None:
        return bad
    name = args.get("name")
    if name not in tasks:
        return Result.failure(f"unknown task {name!r}", code=ERROR_NOT_FOUND)
    paused = getattr(orchestrator, "paused", None)
    if (bad := _need(paused, "task pause state")) is not None:
        return bad
    if args.get("paused", True):
        paused.add(name)
    else:
        paused.discard(name)
        clear = getattr(orchestrator, "clear_skip_cache", None)
        if clear is not None:
            clear(name)
    return Result(data=True)


def _task_reset(ctx, args: dict) -> Result:
    orchestrator = getattr(ctx, "orchestrator", None)
    tasks = getattr(orchestrator, "tasks", None)
    if (bad := _need(tasks, "the task orchestrator")) is not None:
        return bad
    name = args.get("name")
    task = tasks.get(name)
    if task is None:
        return Result.failure(f"unknown task {name!r}", code=ERROR_NOT_FOUND)
    if getattr(task, "trigger", "path") == "event":
        return Result.failure("only path-driven tasks can be reset")
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    if args.get("failed_only"):
        db.reset_failed_tasks(name)
    else:
        db.reset_task(name)
    clear = getattr(orchestrator, "clear_skip_cache", None)
    if clear is not None:
        clear(name)
    return Result(data=True)


def _task_trigger(ctx, args: dict) -> Result:
    import json
    from uuid import uuid4

    orchestrator = getattr(ctx, "orchestrator", None)
    tasks = getattr(orchestrator, "tasks", None)
    if (bad := _need(tasks, "the task orchestrator")) is not None:
        return bad
    name = args.get("name")
    task = tasks.get(name)
    if task is None:
        return Result.failure(f"unknown task {name!r}", code=ERROR_NOT_FOUND)
    if getattr(task, "trigger", "path") != "event":
        return Result.failure(
            "only event-driven tasks can be triggered manually")
    db = _db(ctx)
    creator = getattr(db, "create_run", None)
    if (bad := _need(creator, "task runs")) is not None:
        return bad
    schema = getattr(task, "event_payload_schema", None) or {}
    properties = (schema.get("properties") or {}).keys()
    payload = {
        key: value for key, value in (args.get("payload") or {}).items()
        if key in properties
    }
    run_id = f"{name}:{uuid4().hex[:12]}"
    # The payload's *values* came from the guest, so this is its mistake to
    # hear about rather than a kernel failure.
    try:
        payload_json = json.dumps(payload)
    except (TypeError, ValueError) as exc:
        return Result.failure(f"payload is not JSON-serializable: {exc}",
                              code=ERROR_INVALID_ARGUMENT)
    creator(run_id, name, triggered_by="manual", payload_json=payload_json)
    notify = getattr(orchestrator, "on_run_enqueued", None)
    if notify is not None:
        notify(run_id, name)
    return Result(data=run_id)


def _file_register(ctx, args: dict) -> Result:
    """Add a path to the watched-file table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    db.upsert_file(args.get("path"), **(args.get("meta") or {}))
    return Result(data=True)


def _file_list(ctx, args: dict) -> Result:
    """Query the watched-file table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    modality = args.get("modality")
    rows = (db.get_files_by_modality(modality) if modality
            else db.get_all_files())
    return Result(data=_rows(rows))


# What a parse can hand back across the boundary. The other modalities
# (image, audio, video, tabular) resolve to live objects from foreign
# libraries — PIL images, numpy arrays, an open ``av.Container`` — which are
# the whole point of asking for them and cannot be sent anywhere. Code that
# needs one imports the parser into its own box and consumes it there; what
# leaves the box is the text or the paths it produced.
CROSSABLE_MODALITIES = {"text", "container"}


def _parse_file(ctx, args: dict) -> Result:
    """Parse a file and return its text, or the paths it contained."""
    import parsing

    modality = args.get("modality") or "text"
    if modality not in CROSSABLE_MODALITIES:
        # Reaching here means the caller did *not* declare this modality — a
        # declared one is provisioned into its box and never becomes a
        # Request. So the answer is the declaration, not a lecture about
        # object lifetimes: the result cannot cross, but it does not have to,
        # because the parser can come to the caller instead.
        return Result.failure(
            f"{modality!r} parsing produces live objects that cannot cross the "
            f"sandbox boundary. Declare parse_modalities = [{modality!r}] on "
            f"your plugin and the kernel loads that parser into your own box, "
            f"where the result is usable. Modalities that cross as they are: "
            f"{sorted(CROSSABLE_MODALITIES)}")

    try:
        parsed = parsing.parse(args.get("path"), modality)
    except Exception as exc:
        logger.exception("parse_file failed")
        return Result.failure(f"parse failed: {exc}")

    if not getattr(parsed, "success", True):
        return Result.failure(str(getattr(parsed, "error", "") or "parse failed"))

    # ``output`` is the payload — there has never been a ``.text`` attribute,
    # so the old getattr fell through to the ParseResult itself and handed
    # back an object that only looked right in-process.
    return Result(data=getattr(parsed, "output", None),
                  also_contains=list(getattr(parsed, "also_contains", None) or []))


def _parse_modality(ctx, args: dict) -> Result:
    """Resolve a file extension's modality.

    Always answerable: the kernel's native defaults cover image/audio/video
    with no parser installed at all, which is what attachment routing needs.
    """
    import parsing

    return Result(data=parsing.get_modality(args.get("extension") or ""))


def _ledger_record(ctx, args: dict) -> Result:
    """Write an audit row for something that is not itself a Request."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        db.record_action(origin="sandbox",
                         action_type=args.get("action") or "note",
                         ok=bool(args.get("ok", True)),
                         session_key=getattr(ctx, "session_key", None),
                         data_json=args.get("data"))
        return Result(data=True)
    except Exception:
        # Ledger writes are best-effort at every layer: the ledger observes
        # the system and must never break it.
        return Result(data=False)


def _ledger_read(ctx, args: dict) -> Result:
    """Query the ledger, targeted rather than linearly."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    limit, bad = int_arg(args, "limit", 50, lo=1)
    if bad is not None:
        return bad
    return Result(data=_rows(db.get_ledger_rows(limit=limit)))


# How often the wait below looks up to see whether the caller still wants an
# answer. Short enough that a cancelled turn dies promptly, long enough that a
# script running for minutes costs a negligible number of wakeups.
_SCRIPT_POLL = 0.2


def _script_owner(chain) -> str:
    """Who may collect a script this chain detached.

    The chain *root* — what caused the work — rather than the innermost link.
    Two scripts started by the same turn should be collectable together, which
    is what ``collect(ids=None)`` means, and the root is the only part of a
    chain that is the same for both. It is also the part a guest cannot state
    about itself, so a box cannot claim somebody else's runs.
    """
    return getattr(chain, "root", None) or "user"


def _script_run(ctx, args: dict) -> Result:
    """Run a file of SDK code that is not a plugin.

    The whole of this handler is plumbing: validation, isolation and
    classification all happened before it was reached, and every effect the
    script performs comes back through the gate on its own. What is left is
    resolving a path, descending the chain, and — the part that is not
    obvious — noticing if the caller goes away.
    """
    from plugins.plugin_paths import resolve_plugin_path

    from .. import provenance
    from ..isolation import is_script, resolve_script

    raw = (args.get("path") or "").strip()
    # The script trees answer first, and only for the two shapes they recognise
    # (see ``resolve_script``). Anything else falls through to the general
    # resolver, which knows about the plugin families this does not.
    path = resolve_script(raw)
    if path is None:
        path, error = resolve_plugin_path(raw)
        if error:
            return Result.failure(error)
    if not path.is_file():
        return Result.failure(f"no such script: {path}", code=ERROR_NOT_FOUND)
    # Re-checked here as well as in the policy function. The two answer
    # different questions — the policy decides whether to *ask*, this decides
    # whether to *run* — and a handler that trusted the classifier to have
    # covered it would silently start running scripts from anywhere the day
    # somebody added a branch above it.
    if not is_script(path):
        # Naming the real directory rather than ``<tree>/scripts/``: this
        # message is what the agent has to learn from when it guessed, and an
        # abstraction is not somewhere a file can be put.
        import trees
        return Result.failure(
            f"{path.name} is not in a scripts/ directory; put it in "
            f"{trees.tree("workspace").path / 'scripts'} and run it from there, so "
            f"that it is contained before it runs")

    caller = provenance.current()
    chain = caller.chain if caller is not None else None
    entry = (args.get("entry") or "main").strip()
    kwargs = dict(args.get("args") or {})

    sandbox, refusal = _the_sandbox("script_run")
    if refusal is not None:
        return refusal

    wait = args.get("wait", True)
    try:
        run = sandbox.start(str(path), entry, kwargs=kwargs, chain=chain,
                            context=getattr(caller, "context", None),
                            # Only a detached run is kept for collection. A
                            # waited one hands its Result back here and there is
                            # nothing left to come back for.
                            collect_owner=None if wait else _script_owner(chain))
    except Exception as exc:
        # A BoxError here is the useful case: a script declaring a persistent
        # lifetime is told to be opened rather than run, which is a real
        # message and not a crash.
        logger.exception("script_run failed")
        return Result.failure(f"{path.name} could not start: {exc}")

    if not wait:
        # ``started`` and ``script`` are what this answered before there was an
        # id; kept so nothing reading them has to change.
        return Result(data={"script": path.name, "id": run.id,
                            "started": True})

    # Waiting in slices rather than one blocking call. Cancellation reaches
    # code that is making Requests, and this handler is making none while it
    # waits — so a cancelled caller would otherwise sit here until the child
    # hit its own ceiling, holding a pool worker and finishing work nobody is
    # going to read.
    while True:
        outcome = run.wait(timeout=_SCRIPT_POLL)
        if run.done:
            return outcome
        if caller is not None and caller.abandoned:
            run.cancel()
            return Result.failure(f"{path.name} was cancelled")


def _script_collect(ctx, args: dict) -> Result:
    """Take the results of scripts this caller started with ``wait=False``.

    The counterpart of ``agent.collect``, and it waits the same way ``_script_
    run`` does — in slices, watching for the caller going away — because this
    handler makes no Requests of its own while it waits, so cancellation has no
    other route in.
    """
    from .. import provenance

    caller = provenance.current()
    owner = _script_owner(getattr(caller, "chain", None))
    sandbox, refusal = _the_sandbox("script_collect")
    if refusal is not None:
        return refusal

    runs = sandbox.collectable(owner, args.get("ids"))
    if not runs:
        return Result(data=[])

    timeout = args.get("timeout")
    timeout = None if timeout is None else float(timeout)
    started = time.monotonic()
    while True:
        pending = [run for run in runs if not run.done]
        # Nothing left to wait for, or the caller asked not to wait, or its own
        # limit has passed. ``timeout=0`` is the poll: one pass, no sleeping.
        if (not pending
                or timeout == 0
                or (timeout is not None
                    and time.monotonic() - started >= timeout)):
            break
        if caller is not None and caller.abandoned:
            # The caller is gone. Leave the runs alone — cancelling them here
            # would end work a *different* collector may still be owed, and
            # ``interrupt_session`` already reaches anything this turn started.
            return Result.failure("collection was cancelled")
        time.sleep(_SCRIPT_POLL)

    reports = []
    for run in runs:
        report = run.report()
        reports.append(report)
        # Delivered once, and only what is actually finished: a run still going
        # stays in the registry so a later collect still gets it.
        if report["state"] != "running":
            sandbox.take(run)
    return Result(data=reports)


def _script_stop(ctx, args: dict) -> Result:
    """Cancel a detached script. Answers whether there was one to cancel."""
    from .. import provenance

    caller = provenance.current()
    owner = _script_owner(getattr(caller, "chain", None))
    sandbox, refusal = _the_sandbox("script_stop")
    if refusal is not None:
        return refusal

    run = sandbox.find_run(args.get("id") or "", owner)
    if run is None:
        return Result(data=False)
    run.cancel()
    return Result(data=True)


def _self_budget(ctx, args: dict) -> Result:
    """How much of its deadline the calling execution has left.

    The kernel is the only party that can answer this. The guest can read a
    clock, but the deadline it is judged against is *running* time — elapsed
    minus whatever the kernel spent owing it an answer — and it can see neither
    that discount nor the ceiling its declared timeout was clamped to.

    Without it the watchdog is the only thing that ends an over-long run, and
    it ends it by killing the box: a loop three-quarters through a corpus
    returns nothing at all. Answering lets it stop itself and hand back what it
    has.
    """
    from .. import provenance
    from ..watchdog import HARD_CEILING

    caller = provenance.current()
    execution = getattr(caller, "execution", None)
    if execution is None:
        # No execution to speak about — an in-process caller the provenance
        # stack never marked. Say so with nulls rather than inventing a
        # deadline: a fabricated number would be believed and acted on.
        return Result(data={"running": None, "wall": None,
                            "deadline": None, "ceiling": HARD_CEILING})
    return Result(data=execution.remaining())


def _app_stop(ctx, args: dict) -> Result:
    """End the process, optionally starting it again.

    The kernel owns the two callables (only the composition root has them) and
    the delay: answering first and stopping a moment later is what lets the
    frontend print why it is going away. Without that the process would be gone
    before the Result reached the box that asked.
    """
    control = getattr(ctx, "app_control", None)
    if (bad := _need(control, "stopping the application")) is not None:
        return bad
    restart = bool(args.get("restart"))
    action = getattr(control, "restart" if restart else "stop", None)
    if action is None:
        return Result.failure(
            "restart is not supported in this frontend" if restart
            else "stopping the application is not available")
    message = action()
    # No message means nothing was scheduled — a frontend that cannot restart.
    # Answering ok with no data would print as silence and read as success.
    if not message:
        return Result.failure(
            "restart is not supported in this frontend" if restart
            else "the application could not be stopped")
    return Result(data=message)


HANDLERS = {
    DB_QUERY: _db_query, DB_WRITE: _db_write, DB_DEFINE: _db_define,
    CONV_CREATE: _conv_create, CONV_READ: _conv_read, CONV_LIST: _conv_list,
    CONV_APPEND: _conv_append, CONV_SET_TITLE: _conv_set_title,
    CONV_SET_CATEGORY: _conv_set_category,
    CONV_SET_NOTIFICATION_MODE: _conv_set_notification_mode,
    CONV_LOAD: _conv_load, CONV_CLEAR: _conv_clear,
    CONV_DELETE: _conv_delete,
    SESSION_GET: _session_get, SESSION_LIST: _session_list,
    SESSION_PUSH: _session_push, SESSION_STATE_GET: _session_state_get,
    SESSION_STATE_SET: _session_state_set, SESSION_CANCEL: _session_cancel,
    SESSION_ADD_TOOL: _session_add_tool,
    SESSION_REMOVE_TOOL: _session_remove_tool,
    SESSION_ADD_PROMPT: _session_add_prompt,
    SESSION_REMOVE_PROMPT: _session_remove_prompt,
    SESSION_ADD_ATTACHMENT: _session_add_attachment,
    SESSION_SET_MODE: _session_set_mode,
    UI_ASK: _ui_ask, UI_APPROVE: _ui_approve, UI_RENDER: _ui_render,
    CONFIG_READ: _config_read, CONFIG_WRITE: _config_write,
    PATH_GET: _path_get,
    USER_READ: _user_read, USER_LIST: _user_list, USER_WRITE: _user_write,
    PLUGIN_LIST: _plugin_list, PLUGIN_DESCRIBE: _plugin_describe,
    PLUGIN_VALIDATE: _plugin_validate,
    PLUGIN_REGISTER: _plugin_register, PLUGIN_UNREGISTER: _plugin_unregister,
    PLUGIN_RELOAD: _plugin_reload,
    PLUGIN_INSTALL: _plugin_install, PLUGIN_UNINSTALL: _plugin_uninstall,
    PLUGIN_UPDATE: _plugin_update,
    SERVICE_LIST: _service_list, SERVICE_CALL: _service_call,
    SERVICE_LOAD: _service_load, SERVICE_UNLOAD: _service_unload,
    TOOL_LIST: _tool_list, TOOL_CALL: _tool_call,
    COMMAND_LIST: _command_list, COMMAND_CALL: _command_call,
    AGENT_COMPLETE: _agent_complete,
    AGENT_SPAWN: _agent_spawn, AGENT_COLLECT: _agent_collect,
    AGENT_STOP: _agent_stop, AGENT_SCHEDULE: _agent_schedule,
    LLM_PROCEED: _model_proceed, LLM_DELTA: _model_delta,
    LLM_LIST: _llm_list, LLM_LOAD: _llm_load, LLM_UNLOAD: _llm_unload,
    CRON_LIST: _cron_list, CRON_GET: _cron_get, CRON_CREATE: _cron_create,
    CRON_UPDATE: _cron_update, CRON_REMOVE: _cron_remove,
    CRON_ENABLE: _cron_enable,
    EVENT_EMIT: _event_emit, EVENT_REQUEST: _event_request,
    FRONTEND_SUBMIT: _frontend_submit, FRONTEND_CANCEL: _frontend_cancel,
    FRONTEND_BIND: _frontend_bind, FRONTEND_ATTEND: _frontend_attend,
    FRONTEND_RESOLVE: _frontend_resolve, FRONTEND_PENDING: _frontend_pending,
    FRONTEND_ACT: _frontend_act, FRONTEND_COLLECT: _frontend_collect,
    CONSOLE_READ: _console_read, CONSOLE_WRITE: _console_write,
    HTTP_DRAIN: _http_drain, HTTP_RESPOND: _http_respond,
    HTTP_PUSH: _http_push, HTTP_CLOSE: _http_close,
    TASK_ENQUEUE: _task_enqueue, TASK_STATUS: _task_status,
    TASK_OUTPUT: _task_output, TASK_LIST: _task_list,
    TASK_GRAPH: _task_graph, TASK_PAUSE: _task_pause,
    TASK_RESET: _task_reset, TASK_TRIGGER: _task_trigger,
    FILE_REGISTER: _file_register, FILE_LIST: _file_list,
    PARSE_FILE: _parse_file, PARSE_MODALITY: _parse_modality,
    LEDGER_RECORD: _ledger_record, LEDGER_READ: _ledger_read,
    SCRIPT_RUN: _script_run, SCRIPT_COLLECT: _script_collect,
    SCRIPT_STOP: _script_stop,
    SELF_BUDGET: _self_budget,
    APP_STOP: _app_stop,
}
