"""Sandboxed `/agent` profile-management coverage."""

from types import SimpleNamespace

from sandbox import Sandbox


class FakeDb:
    def __init__(self):
        self.user_config = {"active_agent_profile": "builder"}

    def get_user_config(self, _user_id):
        return dict(self.user_config)

    def set_user_config(self, _user_id, values):
        self.user_config = dict(values)


def _context(monkeypatch):
    profiles = {
        "builder": {
            "llm": "default",
            "prompt_suffix": "",
            "whitelist_or_blacklist_tools": "blacklist",
            "tools_list": ["read_file"],
        },
        "default": {
            "llm": "default",
            "prompt_suffix": "",
            "whitelist_or_blacklist_tools": "blacklist",
            "tools_list": [],
        },
    }
    config = {
        "agent_profiles": profiles,
        "active_agent_profile": "builder",
        "llm_profiles": {"fast": {}},
    }
    saved = {}
    monkeypatch.setattr(
        "config.config_manager.save", lambda values: saved.update(values))
    session = SimpleNamespace(
        active_agent_profile="builder", profile_override="builder")
    switched = []

    def set_agent_profile(key, profile):
        switched.append((key, profile))
        session.active_agent_profile = profile
        session.profile_override = profile
        return True

    runtime = SimpleNamespace(
        config=config,
        sessions={"chat": session},
        set_agent_profile=set_agent_profile,
        refresh_session_specs=lambda: None,
    )
    db = FakeDb()
    context = SimpleNamespace(
        config=dict(config),
        runtime=runtime,
        db=db,
        user_id=1,
        session_key="chat",
        tool_registry=SimpleNamespace(
            tools={"read_file": object(), "grep": object()},
            list_tools=lambda: ["read_file", "grep"]),
        services={},
    )
    return context, db, session, switched, saved


def _run(context, args, *, method="run", approve=None):
    sandbox = Sandbox(context=context, approve=approve)
    try:
        return sandbox.run(
            "plugins/commands/command_agent.py",
            "AgentCommand",
            kwargs={"args": args},
            method=method,
        )
    finally:
        sandbox.shutdown()


def test_agent_form_covers_add_select_and_edit(monkeypatch):
    context, _, _, _, _ = _context(monkeypatch)

    selected = _run(
        context, {"profile_name": "builder"}, method="form")
    adding = _run(
        context, {"profile_name": "add"}, method="form")
    editing = _run(
        context,
        {"profile_name": "builder", "action": "edit"},
        method="form",
    )

    assert selected.data[0]["enum_labels"][0] == "builder (active)"
    assert selected.data[1]["enum"] == ["switch", "edit", "remove"]
    assert adding.data[2]["enum"] == ["default", "fast"]
    assert "grep, read_file" in adding.data[-1]["prompt"]
    assert editing.data[-1]["type"] == "string"


def test_agent_add_and_switch_preserve_exact_outputs(monkeypatch):
    context, db, session, switched, saved = _context(monkeypatch)

    added = _run(
        context,
        {
            "profile_name": "add",
            "new_profile_name": "writer",
            "llm": "fast",
            "prompt_suffix": "Be concise.",
            "whitelist_or_blacklist_tools": "whitelist",
            "tools_list": ["read_file"],
        },
        approve=lambda *_: True,
    )
    context.config.update(saved)
    switched_result = _run(
        context,
        {"profile_name": "writer", "action": "switch"},
        approve=lambda *_: True,
    )

    assert added.data == "Added agent profile: writer"
    assert switched_result.data == "Switched agent profile to: writer"
    assert db.user_config["active_agent_profile"] == "writer"
    assert session.active_agent_profile == "writer"
    assert switched == [("chat", "writer")]


def test_agent_rename_updates_active_and_live_session_refs(monkeypatch):
    context, db, session, switched, saved = _context(monkeypatch)

    result = _run(
        context,
        {
            "profile_name": "builder",
            "action": "edit",
            "field": "agent_profile_name",
            "value": "writer",
        },
        approve=lambda *_: True,
    )

    assert result.data == "Updated agent profile: writer"
    assert "builder" not in saved["agent_profiles"]
    assert session.active_agent_profile == "writer"
    assert session.profile_override == "writer"
    assert db.user_config["active_agent_profile"] == "writer"
    assert switched[-1] == ("chat", "writer")


def test_agent_edit_tools_and_remove_active_profile(monkeypatch):
    context, db, session, _, saved = _context(monkeypatch)

    edited = _run(
        context,
        {
            "profile_name": "builder",
            "action": "edit",
            "field": "tools_list",
            "value": ["grep"],
        },
        approve=lambda *_: True,
    )
    context.config.update(saved)
    removed = _run(
        context,
        {"profile_name": "builder", "action": "remove"},
        approve=lambda *_: True,
    )

    assert edited.data == "Updated agent profile: builder"
    assert saved["agent_profiles"].get("builder") is None
    assert removed.data == "Removed agent profile: builder"
    assert db.user_config["active_agent_profile"] == "default"
    assert session.active_agent_profile == "default"


def test_agent_guards_default_and_unknown_profiles(monkeypatch):
    context, _, _, _, _ = _context(monkeypatch)

    guarded = _run(
        context, {"profile_name": "default", "action": "remove"})
    unknown = _run(
        context, {"profile_name": "missing", "action": "switch"})

    assert guarded.data == "Cannot remove the default agent profile."
    assert unknown.data == "Unknown agent profile."
