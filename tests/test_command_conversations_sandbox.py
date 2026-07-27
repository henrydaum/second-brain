"""Conversation lifecycle SDK and sandboxed picker coverage."""

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from pipeline.database import Database
from plugins.frontends.helpers.command_registry import CommandRegistry
from plugins.plugin_discovery import discover_commands
from runtime.context import build_context
from runtime.conversation_runtime import ConversationRuntime
from sandbox import Sandbox
from sandbox.bridge import adapt, configure
from sandbox.console import CONSOLE
from sandbox.guest.requests import (
    CONV_LOAD,
    CONV_SET_NOTIFICATION_MODE,
    Request,
)
from sandbox.policy import ALWAYS_SAFE
from state_machine.serialization import latest_state, save_state_marker


def _context(tmp_path, *, config=None):
    db = Database(str(tmp_path / "conversations.db"))
    config = config if config is not None else {
        "llm_profiles": {"test": {}},
        "default_llm_profile": "test",
    }
    runtime = ConversationRuntime(db=db, services={}, config=config)
    runtime.set_session_user("chat", 1)
    context = build_context(
        db,
        config,
        {},
        runtime=runtime,
        root_dir=tmp_path,
        session_key="chat",
    )
    context.command_registry = object()
    return context, runtime, db


def _run(context, args, *, entry="ConversationsCommand", method="run",
         approve=None):
    filename = (
        "command_new.py" if entry == "NewCommand"
        else "command_conversations.py"
    )
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            f"plugins/commands/{filename}",
            entry,
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def _conversation(db, *, title="A chat", category=None):
    cid = db.create_conversation(title, category=category, user_id=1)
    db.save_message(cid, "user", "hello from the user")
    db.save_message(cid, "assistant", "hello from the assistant")
    save_state_marker(db, cid, {
        "active_agent_profile": "writer",
        "notification_mode": "off",
    })
    return cid


def test_conversation_requests_are_safe_but_not_read_only():
    assert {CONV_LOAD, CONV_SET_NOTIFICATION_MODE} <= ALWAYS_SAFE
    assert not Request(CONV_LOAD).read_only
    assert not Request(CONV_SET_NOTIFICATION_MODE).read_only


def test_conversations_form_builds_category_preview_and_actions(tmp_path):
    context, _, db = _context(tmp_path)
    cid = _conversation(db)

    initial = _run(context, {}, method="form")
    category = _run(context, {"category": "Main"}, method="form")
    selected = _run(
        context,
        {"category": "Main", "conversation_id": str(cid)},
        method="form",
    )
    recategorize = _run(
        context,
        {
            "category": "Main",
            "conversation_id": str(cid),
            "action": "Change category",
            "target_category": "Add New category",
        },
        method="form",
    )

    assert initial.data[0]["enum"] == ["Main"]
    assert category.data[1]["enum"] == [str(cid)]
    assert category.data[1]["enum_labels"][0].startswith("A chat  (")
    prompt = selected.data[2]["prompt"]
    assert "| A chat |  |" in prompt
    assert "| Agent | writer |" in prompt
    assert "| Notifications | off |" in prompt
    assert "> user: hello from the user" in prompt
    assert "> assistant: hello from the assistant" in prompt
    assert [step["name"] for step in recategorize.data] == [
        "category", "conversation_id", "action",
        "target_category", "custom_category",
    ]


def test_conversation_lifecycle_actions_preserve_outputs(tmp_path):
    context, runtime, db = _context(tmp_path)
    cid = _conversation(db)

    loaded = _run(
        context,
        {"conversation_id": cid, "action": "Load conversation"},
    )
    loaded_cid = runtime.sessions["chat"].conversation_id
    notified = _run(
        context,
        {
            "conversation_id": cid,
            "action": "Change notification mode",
            "mode": "on",
        },
    )
    notification_marker = latest_state(
        db.get_conversation_messages(cid))
    moved = _run(
        context,
        {
            "conversation_id": cid,
            "action": "Change category",
            "target_category": "Work",
        },
    )
    deleted = _run(
        context,
        {"conversation_id": cid, "action": "Delete conversation"},
        approve=lambda *_: True,
    )

    assert loaded.data
    assert loaded_cid == cid
    assert notified.data == f"Notifications for #{cid} → on."
    assert notification_marker["notification_mode"] == "on"
    assert moved.data == f"Conversation #{cid} moved to 'Work'."
    assert deleted.data == f"Deleted conversation #{cid}."
    assert db.get_conversation(cid) is None


def test_conversations_do_not_leak_cross_user_rows(tmp_path):
    context, _, db = _context(tmp_path)
    other = db.upsert_user("web", "other")
    cid = db.create_conversation("Private", user_id=other)

    form = _run(context, {}, method="form")
    loaded = _run(context, {"conversation_id": cid})

    assert form.data[0]["enum"] == []
    assert loaded.data == "No such conversation."


def test_new_requires_llm_and_creates_for_current_user(tmp_path):
    missing_context, _, _ = _context(tmp_path, config={})
    missing = _run(missing_context, {}, entry="NewCommand")

    ready = tmp_path / "ready"
    ready.mkdir()
    context, runtime, db = _context(ready)
    created = _run(context, {}, entry="NewCommand")
    cid = runtime.sessions["chat"].conversation_id

    assert missing.data == (
        "No LLM is configured yet. Run /setup to add one before starting "
        "a conversation."
    )
    assert created.data == (
        f"Started new conversation #{cid} under 'Main'.\nAgent: default")
    assert db.get_conversation(cid)["user_id"] == 1


def test_live_repl_collects_conversation_picker(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "conversations-live.db"))
    config = {"llm_profiles": {"test": {}}}
    services = {}
    cid = _conversation(db)
    holder = {}
    commands = CommandRegistry(
        lambda key=None: build_context(
            db, config, services, runtime=holder.get("runtime"),
            root_dir=tmp_path, session_key=key,
        )
    )
    discover_commands(tmp_path, commands, config)
    runtime = ConversationRuntime(
        db=db,
        services=services,
        config=config,
        commands=commands.to_callable_specs(),
    )
    runtime.set_session_user("default", 1)
    holder["runtime"] = runtime

    sandbox = Sandbox()
    configure(sandbox)
    written = []
    original_claim = CONSOLE.claim

    class PacedInput(io.StringIO):
        def readline(self, *args, **kwargs):
            if self.tell():
                time.sleep(0.25)
            return super().readline(*args, **kwargs)

    def claim(token, source=None, writer=None):
        return original_claim(
            token,
            source=PacedInput(
                f"/conversations\nMain\n{cid}\nLoad conversation\n"),
            writer=written.append,
        )

    monkeypatch.setattr(CONSOLE, "claim", claim)
    module = adapt(Path("plugins/frontends/frontend_repl.py").resolve())
    frontend_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    frontend = frontend_cls(shutdown_event=threading.Event())
    frontend.bind(runtime, commands, config)
    thread = threading.Thread(target=frontend.start, daemon=True)

    try:
        thread.start()
        deadline = time.time() + 5
        while time.time() < deadline and not any(
                "Loaded conversation" in text for text in written):
            time.sleep(0.01)
        output = "".join(written)
        assert "Choose a conversation category." in output
        assert "Choose a recent conversation under 'Main'." in output
        assert "What do you want to do with this conversation?" in output
        assert "Loaded conversation" in output
    finally:
        frontend.unbind()
        frontend.stop()
        thread.join(timeout=2)
        sandbox.shutdown()
        configure(None)
