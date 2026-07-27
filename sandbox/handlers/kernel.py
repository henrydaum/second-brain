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

from ..guest.requests import (AGENT_COMPLETE, COMMAND_CALL, COMMAND_LIST,
                              MODEL_DELTA, MODEL_PROCEED,
                              CONFIG_READ, CONFIG_WRITE, CONV_APPEND, CONV_CLEAR,
                              CONV_CREATE, CONV_DELETE, CONV_LIST, CONV_READ,
                              CONV_SET_CATEGORY, CONV_SET_TITLE, CRON_CREATE,
                              CRON_ENABLE, CRON_GET, CRON_LIST, CRON_REMOVE,
                              CONSOLE_READ, CONSOLE_WRITE,
                              CRON_UPDATE, DB_DEFINE, DB_QUERY, DB_WRITE,
                              EVENT_EMIT, EVENT_REQUEST, FILE_LIST,
                              FILE_REGISTER, FRONTEND_ATTEND, FRONTEND_BIND,
                              FRONTEND_CANCEL, FRONTEND_PENDING,
                              FRONTEND_RESOLVE,
                              FRONTEND_SUBMIT, LEDGER_READ, LEDGER_RECORD,
                              PARSE_FILE, PARSE_MODALITY, PATH_GET, PLUGIN_DESCRIBE,
                              PLUGIN_INSTALL, PLUGIN_LIST, PLUGIN_UNINSTALL,
                              PLUGIN_UPDATE, SERVICE_CALL, SERVICE_LIST,
                              SESSION_ADD_PROMPT, SESSION_ADD_TOOL,
                              SESSION_CANCEL, SESSION_GET, SESSION_LIST,
                              SESSION_PUSH, SESSION_REMOVE_PROMPT,
                              SESSION_REMOVE_TOOL, SESSION_STATE_GET,
                              SESSION_STATE_SET, TASK_ENQUEUE, TASK_OUTPUT,
                              TASK_STATUS, TOOL_CALL, TOOL_LIST, UI_APPROVE,
                              UI_ASK, UI_RENDER, USER_LIST, USER_READ,
                              USER_WRITE, Result)
from ..secrets import redact
from ..users import scope_sql

# Never returned by any Request, at any level.
HIDDEN_USER_COLUMNS = {"password_hash"}


def _need(value, what: str):
    """Return a Result explaining an absent capability, or None.

    Callers must compare against None. A failure Result is *falsy* by design
    — that is the whole point of the return contract — so ``if (bad := _need(
    ...)):`` silently does nothing, which is the opposite of a guard.
    """
    if value is None:
        return Result.failure(f"{what} is not available in this kernel")
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
        return Result(data=_rows(db.query(scoped, params)))
    except Exception as exc:
        return Result.failure(f"query failed: {exc}")


def _db_write(ctx, args: dict) -> Result:
    """Insert, update or delete."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    sql = args.get("sql")
    if not sql:
        return Result.failure("db.write requires sql")
    try:
        db.execute_write(sql, args.get("params") or [])
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"write failed: {exc}")


def _db_define(ctx, args: dict) -> Result:
    """Create a plugin-owned table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    ddl = args.get("ddl")
    if not ddl:
        return Result.failure("db.define requires ddl")
    try:
        db.execute_write(ddl, [])
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"define failed: {exc}")


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
    try:
        cid = db.create_conversation(args.get("title") or "New conversation")
        return Result(data=cid)
    except Exception as exc:
        return Result.failure(f"create failed: {exc}")


def _conv_read(ctx, args: dict) -> Result:
    """Messages and metadata for one conversation."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    try:
        return Result(data={
            "conversation": dict(db.get_conversation(cid) or {}),
            "messages": _rows(db.get_conversation_messages(cid)),
        })
    except Exception as exc:
        return Result.failure(f"read failed: {exc}")


def _conv_list(ctx, args: dict) -> Result:
    """Conversations belonging to the current user."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        user_id = getattr(ctx, "user_id", None)
        if user_id is not None and hasattr(db, "list_user_conversations"):
            return Result(data=_rows(db.list_user_conversations(user_id)))
        return Result(data=_rows(db.list_conversations()))
    except Exception as exc:
        return Result.failure(f"list failed: {exc}")


def _conv_append(ctx, args: dict) -> Result:
    """Add a message."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    try:
        db.save_message(cid, args.get("role") or "user", args.get("content") or "")
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"append failed: {exc}")


def _conv_set_title(ctx, args: dict) -> Result:
    """Retitle a conversation."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    cid = args.get("id")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused
    try:
        db.update_conversation_title(cid, args.get("title") or "")
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"retitle failed: {exc}")


def _conv_set_category(ctx, args: dict) -> Result:
    """Categorize a conversation."""
    runtime = _runtime(ctx)
    setter = getattr(runtime, "set_conversation_category", None)
    if (bad := _need(setter, "conversation categories")) is not None:
        return bad
    try:
        setter(args.get("id"), args.get("category") or "")
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"categorize failed: {exc}")


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
        return Result.failure("no conversation loaded")
    if (refused := _check_access(ctx, cid)) is not None:
        return refused

    try:
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
    except Exception as exc:
        return Result.failure(f"clear failed: {exc}")


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
    try:
        deleter(cid)
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"delete failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Sessions.
# ──────────────────────────────────────────────────────────────────────

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
    }
    if args.get("details"):
        if machine is None:
            data["debug"] = None
            return Result(data=data)
        from state_machine.debug import format_recent_events, format_state

        flags = [
            flag
            for service in (getattr(ctx, "services", None) or {}).values()
            for flag in (
                service.debug_flags(session)
                if callable(getattr(service, "debug_flags", None)) else []
            )
        ]
        data["debug"] = {
            "state": format_state(machine),
            "service_flags": flags,
            "recent_events": format_recent_events(machine),
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
    try:
        push(key, args.get("message") or "")
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"push failed: {exc}")


def _session_state_get(ctx, args: dict) -> Result:
    """Read this plugin's per-session scratch state."""
    runtime = _runtime(ctx)
    getter = getattr(runtime, "get_session_plugin_state", None)
    if (bad := _need(getter, "session state")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    try:
        return Result(data=getter(key, args.get("namespace") or "sandbox"))
    except Exception as exc:
        return Result.failure(f"state read failed: {exc}")


def _session_state_set(ctx, args: dict) -> Result:
    """Write this plugin's per-session scratch state."""
    runtime = _runtime(ctx)
    setter = getattr(runtime, "update_session_plugin_state", None)
    if (bad := _need(setter, "session state")) is not None:
        return bad
    key = args.get("key") or getattr(ctx, "session_key", None)
    try:
        setter(key, args.get("namespace") or "sandbox", args.get("value"))
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"state write failed: {exc}")


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


def _session_add_prompt(ctx, args: dict) -> Result:
    """Inject system prompt text for this session."""
    runtime = _runtime(ctx)
    adder = getattr(runtime, "add_system_prompt_extra", None)
    if (bad := _need(adder, "prompt extras")) is not None:
        return bad
    handle = adder(args.get("key") or getattr(ctx, "session_key", None),
                   args.get("text") or "")
    return Result(data=handle)


def _session_remove_prompt(ctx, args: dict) -> Result:
    """Withdraw injected prompt text."""
    runtime = _runtime(ctx)
    remover = getattr(runtime, "remove_system_prompt_extra", None)
    if (bad := _need(remover, "prompt extras")) is not None:
        return bad
    remover(args.get("key") or getattr(ctx, "session_key", None),
            args.get("handle"))
    return Result(data=True)


# ──────────────────────────────────────────────────────────────────────
# Talking to the user.
# ──────────────────────────────────────────────────────────────────────

def _ui_ask(ctx, args: dict) -> Result:
    """Ask a question and wait for the answer."""
    asker = getattr(ctx, "request_user_input", None)
    runtime = _runtime(ctx)
    key = getattr(ctx, "session_key", None)
    if asker is None and runtime is not None and key:
        def asker(title, prompt, **kw):
            """Fall back to the runtime's own prompt."""
            return runtime.request_input(key, title, prompt, **kw)
    if (bad := _need(asker, "asking the user")) is not None:
        return bad

    try:
        request = asker(args.get("title") or "Question",
                        args.get("prompt") or "",
                        type=args.get("type") or "text",
                        choices=args.get("choices") or None)
        if not request.wait(timeout=float(args.get("timeout") or 300.0)):
            return Result.failure("the user did not answer", retryable=True)
        if request.metadata.get("cancelled"):
            return Result.refusal("the user cancelled")
        return Result(data=getattr(request, "value", None)
                      if hasattr(request, "value") else request.approved)
    except Exception as exc:
        return Result.failure(f"could not ask: {exc}")


def _ui_approve(ctx, args: dict) -> Result:
    """Ask the user to approve a described action."""
    approve = getattr(ctx, "approve_command", None)
    if (bad := _need(approve, "approval")) is not None:
        return bad
    try:
        allowed = approve(args.get("action") or "",
                          args.get("justification") or "")
        return Result(data=bool(allowed))
    except Exception as exc:
        return Result.failure(f"could not ask: {exc}")


def _ui_render(ctx, args: dict) -> Result:
    """Show files to the user in chat."""
    runtime = _runtime(ctx)
    push = getattr(runtime, "push_message", None)
    if (bad := _need(push, "rendering to the user")) is not None:
        return bad
    paths = args.get("paths") or []
    caption = args.get("caption") or ""
    try:
        push(getattr(ctx, "session_key", None),
             caption or f"{len(paths)} file(s)")
        return Result(data={"rendered": len(paths)})
    except Exception as exc:
        return Result.failure(f"render failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Config, users.
# ──────────────────────────────────────────────────────────────────────

def _config_read(ctx, args: dict) -> Result:
    """Read a setting, redacting credentials into handles."""
    config = getattr(ctx, "config", None) or {}
    key = args.get("key")
    if key is None:
        return Result(data={k: redact(k, v) for k, v in config.items()})
    if key not in config:
        return Result(data=None)
    return Result(data=redact(key, config[key]))


def _config_write(ctx, args: dict) -> Result:
    """Change a setting."""
    key = args.get("key")
    if not key:
        return Result.failure("config.write requires a key")
    config = getattr(ctx, "config", None)
    if (bad := _need(config, "config")) is not None:
        return bad
    try:
        from config import config_manager
        config[key] = args.get("value")
        config_manager.save(config)
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"config write failed: {exc}")


def _path_get(ctx, args: dict) -> Result:
    """Resolve one of the application locations exposed to plugins."""
    from paths import DATA_DIR, INSTALLED_PLUGINS, ROOT_DIR, SANDBOX_PLUGINS

    locations = {
        "project": getattr(ctx, "root_dir", None) or ROOT_DIR,
        "data": DATA_DIR,
        "installed_plugins": INSTALLED_PLUGINS,
        "sandbox_plugins": SANDBOX_PLUGINS,
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
    try:
        return Result(data=_visible_user(db.get_user(uid)))
    except Exception as exc:
        return Result.failure(f"user read failed: {exc}")


def _user_list(ctx, args: dict) -> Result:
    """Every user, minus anything never returned."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        return Result(data=[_visible_user(r) for r in db.list_users() or []])
    except Exception as exc:
        return Result.failure(f"user list failed: {exc}")


def _user_write(ctx, args: dict) -> Result:
    """Update a user's config blob or type."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    uid = args.get("id", getattr(ctx, "user_id", None))
    try:
        if "config" in args:
            db.set_user_config(uid, args["config"])
        if "user_type" in args:
            db.set_user_type(uid, args["user_type"])
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"user write failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Plugins, services, tools, commands.
# ──────────────────────────────────────────────────────────────────────

def _plugin_list(ctx, args: dict) -> Result:
    """Everything currently registered, by family."""
    source = args.get("source") or "registered"
    if source != "registered":
        try:
            from paths import ROOT_DIR
            from plugins.commands.helpers import package_manager

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
            return Result.failure(str(exc))

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
        from plugins.commands.helpers import package_manager

        root = getattr(ctx, "root_dir", None) or ROOT_DIR
        outcome = package_manager.install_package(
            root, args.get("package_id") or "", ctx,
            progress=_package_progress(ctx))
        return Result(data=outcome.text())
    except Exception as exc:
        return Result.failure(str(exc))


def _plugin_uninstall(ctx, args: dict) -> Result:
    """Uninstall one package through the kernel package manager."""
    try:
        from paths import ROOT_DIR
        from plugins.commands.helpers import package_manager

        root = getattr(ctx, "root_dir", None) or ROOT_DIR
        outcome = package_manager.uninstall_package(
            args.get("package_id") or "", ctx,
            progress=_package_progress(ctx), root_dir=root)
        return Result(data=outcome.text())
    except Exception as exc:
        return Result.failure(str(exc))


def _plugin_update(ctx, args: dict) -> Result:
    """Update all installed packages through the kernel package manager."""
    try:
        from paths import ROOT_DIR
        from plugins.commands.helpers import package_manager

        root = getattr(ctx, "root_dir", None) or ROOT_DIR
        outcome = package_manager.update_packages(
            root, ctx, progress=_package_progress(ctx))
        return Result(data=outcome.text())
    except Exception as exc:
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


def _service_list(ctx, args: dict) -> Result:
    """Loaded services and whether each is ready."""
    services = getattr(ctx, "services", None) or {}
    return Result(data={
        name: bool(getattr(service, "loaded", False))
        for name, service in services.items()
    })


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
        return Result.failure(f"service {name!r} is not loaded")

    exports = getattr(service, "exports", None)
    if exports is not None and method not in exports:
        return Result.refusal(
            f"{name}.{method} is not exported; {sorted(exports)} are")

    fn = getattr(service, method or "", None)
    if not callable(fn):
        return Result.failure(f"{name} has no method {method!r}")
    try:
        return Result(data=fn(**(args.get("kwargs") or {})))
    except Exception as exc:
        return Result.failure(f"{name}.{method} failed: {exc}")


def _tool_list(ctx, args: dict) -> Result:
    """Tools the current scope exposes."""
    registry = getattr(ctx, "tool_registry", None)
    if (bad := _need(registry, "the tool registry")) is not None:
        return bad
    return Result(data=sorted(registry.list_tools()))


def _tool_call(ctx, args: dict) -> Result:
    """Call another tool. The Request that makes a chain two links deep."""
    call = getattr(ctx, "call_tool", None)
    if (bad := _need(call, "tool-to-tool calls")) is not None:
        return bad
    name = args.get("name")
    try:
        outcome = call(name, **(args.get("kwargs") or {}))
        return Result(ok=bool(getattr(outcome, "success", True)),
                      data=getattr(outcome, "data", outcome),
                      error=str(getattr(outcome, "error", "")))
    except Exception as exc:
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
        from plugins.frontends.helpers.command_registry import (
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
    """Run a slash command in one shot."""
    registry = getattr(ctx, "command_registry", None)
    runner = getattr(registry, "run", None) or getattr(registry, "execute", None)
    if (bad := _need(runner, "running commands")) is not None:
        return bad
    try:
        return Result(data=runner(args.get("name"), args.get("args") or {}))
    except Exception as exc:
        return Result.failure(f"command failed: {exc}")


# ──────────────────────────────────────────────────────────────────────
# Agent, scheduling, events, pipeline, parsing, ledger.
# ──────────────────────────────────────────────────────────────────────

def _model_proceed(ctx, args: dict) -> Result:
    """Place the model call an escort is holding.

    Unlike every other handler this one resolves through a token rather than
    a static table, because what it invokes is a closure the kernel built for
    one particular call and will discard the moment the escort returns. Code
    that is not standing at the ``model_call`` doorway holds no token, reaches
    no closure, and is refused — which is the correct answer, not an omission.
    """
    from ..hooks import phone

    dial = phone(args.get("token") or "")
    if dial is None:
        return Result.refusal(
            "model.proceed is only available inside a model_call hook")
    try:
        return Result(data=dial(args.get("request")))
    except Exception as exc:
        return Result.failure(f"model call failed: {exc}")


def _model_delta(ctx, args: dict) -> Result:
    """Carry one fragment of streamed assistant text out of a backend's box.

    Token-scoped exactly like ``model.proceed``, and one-way: the answer says
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
            "model.delta is only available inside an LLM backend's chat call")
    return Result(data=True)


def _agent_complete(ctx, args: dict) -> Result:
    """A model call.

    Its own Request, never a generic ``service.call``: keys, sockets and
    provider details stay kernel-side and the sandbox sees a prompt.
    """
    llm = _service(ctx, "llm")
    if (bad := _need(llm, "an LLM")) is not None:
        return bad
    messages = args.get("messages")
    if not messages:
        prompt = args.get("prompt") or ""
        messages = [{"role": "user", "content": prompt}]
    try:
        response = llm.chat_with_tools(messages)
        return Result(ok=not getattr(response, "is_error", False),
                      data={"content": getattr(response, "content", ""),
                            "tool_calls": getattr(response, "tool_calls", [])},
                      error=str(getattr(response, "error", "") or ""))
    except Exception as exc:
        return Result.failure(f"model call failed: {exc}")


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
    """Publish on a bus channel."""
    try:
        from events.event_bus import bus
        bus.emit(args.get("channel"), args.get("payload"))
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"emit failed: {exc}")


def _event_request(ctx, args: dict) -> Result:
    """Publish and wait for one answer."""
    try:
        from events.event_bus import bus
        return Result(data=bus.request(args.get("channel"),
                                       args.get("payload") or {},
                                       timeout=float(args.get("timeout") or 120.0)))
    except Exception as exc:
        return Result.failure(f"request failed: {exc}", retryable=True)


# ──────────────────────────────────────────────────────────────────────
# Frontends: carrying what a person did into the state machine.
#
# Every one of these resolves through a token to the calling frontend's own
# adapter, so the authority is the frontend's identity rather than anything
# the Request says about itself. A caller that is not a loaded frontend holds
# no token, reaches no adapter, and is refused — which is the correct answer
# rather than an omission, exactly as it is for ``model.proceed``.
# ──────────────────────────────────────────────────────────────────────

def _at_desk(args: dict):
    """The adapter behind a frontend Request, or a refusal explaining why not."""
    from ..frontends import desk

    adapter = desk(args.get("token") or "")
    if adapter is None:
        return None, Result.refusal(
            "sdk.frontend is only available inside a loaded frontend")
    return adapter, None


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
    def submit():
        if kind == "text":
            return adapter.submit_text(session_key, args.get("text") or "")
        elif kind == "attachment":
            return adapter.submit_attachment(
                session_key, args.get("path") or "",
                args.get("extension") or None)
        elif kind == "action":
            return adapter.submit(session_key,
                                  args.get("action_type") or "",
                                  args.get("payload"))
        raise ValueError(f"unknown submit kind {kind!r}")

    if getattr(adapter, "background_submit", False):
        import threading

        def run():
            try:
                submit()
            except Exception:
                logger.exception("background frontend submit failed")

        threading.Thread(
            target=run, daemon=True,
            name=f"{getattr(adapter, 'name', 'frontend')}-submit",
        ).start()
        return Result(data=True)

    try:
        result = submit()
    except Exception as exc:
        return Result.failure(f"submit failed: {exc}")

    # A RuntimeResult is a live object. What a frontend needs back is whether
    # it landed, and the rest reaches it as a render call like everything else.
    return Result(data=bool(getattr(result, "ok", result is not None)))


def _frontend_cancel(ctx, args: dict) -> Result:
    """Stop whatever a session is doing."""
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal
    try:
        adapter.cancel(str(args.get("session_key") or ""))
        return Result(data=True)
    except Exception as exc:
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
    try:
        if external_id is None:
            return Result(data=adapter.bind_session(session_key))
        return Result(data=adapter.identify(
            session_key, external_id, args.get("config") or None,
            user_type=str(args.get("user_type") or "user")))
    except Exception as exc:
        return Result.failure(f"bind failed: {exc}")


def _frontend_attend(ctx, args: dict) -> Result:
    """Say whether a person is watching a session."""
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    try:
        if args.get("present"):
            adapter.mark_attended(session_key)
        else:
            adapter.mark_unattended(session_key)
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"attendance failed: {exc}")


def _frontend_resolve(ctx, args: dict) -> Result:
    """Answer a pending approval by id, or the session's next one."""
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    request_id = str(args.get("request_id") or "")
    try:
        if request_id:
            return Result(data=bool(adapter.resolve_approval(
                session_key, request_id, args.get("value"))))
        return Result(data=bool(adapter.resolve_next_approval(
            session_key, args.get("value"))))
    except Exception as exc:
        return Result.failure(f"resolve failed: {exc}")


def _frontend_pending(ctx, args: dict) -> Result:
    """The id of the approval a session is waiting on, or None.

    Asked rather than remembered. A frontend knows an approval exists — it was
    handed one to render — but not when it stops existing: another frontend can
    answer it, or it can time out. A frontend acting on a stale record would
    swallow the next thing a person typed as a yes/no.
    """
    adapter, refusal = _at_desk(args)
    if refusal is not None:
        return refusal

    session_key = str(args.get("session_key") or "")
    try:
        if not adapter.has_pending_approval(session_key):
            return Result(data=None)
        order = getattr(adapter, "_pending_approval_order", None) or {}
        waiting = list(order.get(session_key) or [])
        # The id is enough to answer and only enough to answer — the same
        # projection the ``approval`` render makes.
        return Result(data=waiting[0] if waiting else True)
    except Exception as exc:
        return Result.failure(f"pending lookup failed: {exc}")


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
    try:
        CONSOLE.write(str(args.get("text") or ""),
                      end=str(args.get("end", "\n")))
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"console write failed: {exc}")


def _task_enqueue(ctx, args: dict) -> Result:
    """Queue pipeline work."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        for path in args.get("paths") or []:
            db.enqueue_task(args.get("name"), path)
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"enqueue failed: {exc}")


def _task_status(ctx, args: dict) -> Result:
    """Where one task stands for one path."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        return Result(data=db.get_task_status(args.get("name"),
                                              args.get("path")))
    except Exception as exc:
        return Result.failure(f"status failed: {exc}")


def _task_output(ctx, args: dict) -> Result:
    """Read a task's output table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        return Result(data=_rows(db.get_task_output(args.get("name"),
                                                    args.get("path"))))
    except Exception as exc:
        return Result.failure(f"output failed: {exc}")


def _file_register(ctx, args: dict) -> Result:
    """Add a path to the watched-file table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        db.upsert_file(args.get("path"), **(args.get("meta") or {}))
        return Result(data=True)
    except Exception as exc:
        return Result.failure(f"register failed: {exc}")


def _file_list(ctx, args: dict) -> Result:
    """Query the watched-file table."""
    db = _db(ctx)
    if (bad := _need(db, "the database")) is not None:
        return bad
    try:
        modality = args.get("modality")
        rows = (db.get_files_by_modality(modality) if modality
                else db.get_all_files())
        return Result(data=_rows(rows))
    except Exception as exc:
        return Result.failure(f"file list failed: {exc}")


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
        return Result.failure(
            f"{modality!r} parsing produces live objects that cannot cross the "
            f"sandbox boundary; import the parser into your own box, or ask "
            f"for {sorted(CROSSABLE_MODALITIES)}")

    try:
        parsed = parsing.parse(args.get("path"), modality)
    except Exception as exc:
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
    try:
        return Result(data=_rows(db.get_ledger_rows(
            limit=int(args.get("limit") or 50))))
    except Exception as exc:
        return Result.failure(f"ledger read failed: {exc}")


HANDLERS = {
    DB_QUERY: _db_query, DB_WRITE: _db_write, DB_DEFINE: _db_define,
    CONV_CREATE: _conv_create, CONV_READ: _conv_read, CONV_LIST: _conv_list,
    CONV_APPEND: _conv_append, CONV_SET_TITLE: _conv_set_title,
    CONV_SET_CATEGORY: _conv_set_category, CONV_CLEAR: _conv_clear,
    CONV_DELETE: _conv_delete,
    SESSION_GET: _session_get, SESSION_LIST: _session_list,
    SESSION_PUSH: _session_push, SESSION_STATE_GET: _session_state_get,
    SESSION_STATE_SET: _session_state_set, SESSION_CANCEL: _session_cancel,
    SESSION_ADD_TOOL: _session_add_tool,
    SESSION_REMOVE_TOOL: _session_remove_tool,
    SESSION_ADD_PROMPT: _session_add_prompt,
    SESSION_REMOVE_PROMPT: _session_remove_prompt,
    UI_ASK: _ui_ask, UI_APPROVE: _ui_approve, UI_RENDER: _ui_render,
    CONFIG_READ: _config_read, CONFIG_WRITE: _config_write,
    PATH_GET: _path_get,
    USER_READ: _user_read, USER_LIST: _user_list, USER_WRITE: _user_write,
    PLUGIN_LIST: _plugin_list, PLUGIN_DESCRIBE: _plugin_describe,
    PLUGIN_INSTALL: _plugin_install, PLUGIN_UNINSTALL: _plugin_uninstall,
    PLUGIN_UPDATE: _plugin_update,
    SERVICE_LIST: _service_list, SERVICE_CALL: _service_call,
    TOOL_LIST: _tool_list, TOOL_CALL: _tool_call,
    COMMAND_LIST: _command_list, COMMAND_CALL: _command_call,
    AGENT_COMPLETE: _agent_complete,
    MODEL_PROCEED: _model_proceed, MODEL_DELTA: _model_delta,
    CRON_LIST: _cron_list, CRON_GET: _cron_get, CRON_CREATE: _cron_create,
    CRON_UPDATE: _cron_update, CRON_REMOVE: _cron_remove,
    CRON_ENABLE: _cron_enable,
    EVENT_EMIT: _event_emit, EVENT_REQUEST: _event_request,
    FRONTEND_SUBMIT: _frontend_submit, FRONTEND_CANCEL: _frontend_cancel,
    FRONTEND_BIND: _frontend_bind, FRONTEND_ATTEND: _frontend_attend,
    FRONTEND_RESOLVE: _frontend_resolve, FRONTEND_PENDING: _frontend_pending,
    CONSOLE_READ: _console_read, CONSOLE_WRITE: _console_write,
    TASK_ENQUEUE: _task_enqueue, TASK_STATUS: _task_status,
    TASK_OUTPUT: _task_output,
    FILE_REGISTER: _file_register, FILE_LIST: _file_list,
    PARSE_FILE: _parse_file, PARSE_MODALITY: _parse_modality,
    LEDGER_RECORD: _ledger_record, LEDGER_READ: _ledger_read,
}
