from types import SimpleNamespace
from unittest.mock import patch

from events.event_bus import bus
from events.event_channels import SESSION_TURN_COMPLETED
from pipeline.database import Database
from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.command_registry import CommandRegistry, parse_command_line
from plugins.plugin_discovery import discover_commands
from plugins.tasks.task_run_subagent import RunSubagent
from plugins.tools.tool_slash_command import SlashCommand
from plugins.tools.tool_ask_subagent import AskSubagent
from state_machine.conversationClass import FormStep


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def create_conversation(self, title, *, kind="user"):
        self.calls.append(("create_conversation", title, kind))
        return 7

    def subagent_session_key(self, job_name):
        return f"subagent:{job_name}"

    def load_conversation(self, session_key, conversation_id, **kwargs):
        self.calls.append(("load_conversation", session_key, conversation_id, kwargs))

    def add_session_tool(self, session_key, tool):
        self.calls.append(("add_session_tool", session_key, tool.name))

    def iterate_agent_turn(self, session_key, prompt, *, image_paths=None, actor_id="user"):
        self.calls.append(("iterate_agent_turn", session_key, prompt, image_paths, actor_id))
        bus.emit(SESSION_TURN_COMPLETED, {"session_key": session_key, "conversation_id": 7, "final_text": "done", "new_messages": [], "attachments": []})
        return SimpleNamespace(ok=True, error=None, messages=["done"], attachments=[], data={})

    def unload_conversation(self, session_key):
        self.calls.append(("unload_conversation", session_key))

    def inject_user_message(self, session_key, text, *, conversation_id=None, actor_id="user"):
        self.calls.append(("inject_user_message", session_key, text, conversation_id, actor_id))


class FakeDB:
    def get_conversation(self, conversation_id):
        return {"id": conversation_id, "kind": "subagent"}

    def replace_conversation_messages(self, *_):
        raise AssertionError("task_run_subagent should not replace history directly")


def test_run_subagent_uses_public_runtime_api():
    runtime = FakeRuntime()
    context = SimpleNamespace(
        runtime=runtime,
        db=FakeDB(),
        config={"agent_profiles": {"default": {}}},
        services={},
    )

    result = RunSubagent().run_event("run-1", {"prompt": "do it", "title": "Job"}, context)

    assert result.success
    assert [c[0] for c in runtime.calls] == [
        "create_conversation",
        "load_conversation",
        "add_session_tool",
        "iterate_agent_turn",
        "unload_conversation",
    ]
    assert runtime.calls[0][2] == "subagent"


def test_subagent_conversations_are_hidden_from_user_history():
    db = Database(":memory:")
    user_id = db.create_conversation("User")
    sub_id = db.create_conversation("Background", kind="subagent")

    ids = [row["id"] for row in db.list_user_conversations()]

    assert user_id in ids
    assert sub_id not in ids


def test_ask_subagent_reads_final_answer_from_conversation():
    db = SimpleNamespace(get_conversation_messages=lambda _cid: [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "done"},
    ])
    context = SimpleNamespace(db=db)

    result = AskSubagent()._build_success_result(context, "run-1", 7, "do it", "Job", [])

    assert result.success
    assert result.data["conversation_id"] == 7
    assert result.data["final_answer"] == "done"


def test_command_discovery_loads_minimal_builtin_commands():
    import plugins.plugin_discovery as discovery

    old_sandbox = discovery._COMMAND_CONFIG["sandbox_dir"]
    discovery._COMMAND_CONFIG["sandbox_dir"] = discovery.ROOT_DIR / ".missing_sandbox_commands"
    registry = CommandRegistry()
    try:
        discover_commands(".", registry)
        assert [cmd.name for cmd in registry.all_commands()] == ["agent", "cancel", "commands", "config", "frontends", "history", "llm", "locations", "new", "services", "tasks", "tools", "update"]
    finally:
        discovery._COMMAND_CONFIG["sandbox_dir"] = old_sandbox


def test_host_commands_are_visible_and_user_approved():
    from pathlib import Path
    from plugins.frontends.bootstrap import _conversation_runtime

    class ToolRegistry:
        tools = {}
        def get_all_schemas(self): return []

    scaffold = SimpleNamespace(
        db=None,
        config={},
        orchestrator=SimpleNamespace(tasks={}, runtime=None),
        restart=lambda: None,
    )
    runtime = _conversation_runtime(scaffold, lambda: None, ToolRegistry(), {}, {}, Path("."))

    names = sorted(runtime.command_registry._commands)
    visible = [cmd.name for cmd in runtime.command_registry.visible_commands()]

    assert names == ["agent", "cancel", "commands", "config", "frontends", "history", "llm", "locations", "new", "quit", "restart", "services", "tasks", "tools", "update"]
    assert visible == ["agent", "cancel", "commands", "config", "frontends", "history", "llm", "locations", "new", "quit", "restart", "services", "tasks", "tools", "update"]
    assert not runtime.commands["cancel"].require_approval
    assert not runtime.commands["commands"].require_approval
    assert runtime.commands["quit"].require_approval
    assert runtime.commands["quit"].approval_actor_id == "user"
    assert runtime.commands["restart"].require_approval
    assert runtime.commands["restart"].approval_actor_id == "user"
    assert runtime.commands["update"].require_approval
    assert runtime.commands["update"].approval_actor_id == "user"

    text = runtime.command_registry.dispatch_dict("commands", {}, session_key="default", _emit=False)
    for name in visible:
        assert f"/{name}" in text


def test_quit_host_command_delays_shutdown_until_after_command_finish():
    from plugins.frontends.bootstrap import _quit
    called = []

    with patch("plugins.frontends.bootstrap.threading.Timer") as timer:
        timer.return_value.start.side_effect = lambda: called.append("started")
        assert _quit(lambda: called.append("shutdown")) == "Shutting down."

    timer.assert_called_once()
    assert timer.call_args.args[0] == 0.75
    assert called == ["started"]


def test_resource_commands_share_action_target_pattern():
    from plugins.commands.command_frontends import FrontendsCommand
    from plugins.commands.command_services import ServicesCommand
    from plugins.commands.command_tasks import TasksCommand

    class Service:
        loaded = False
        model_name = "svc"
        def load(self): self.loaded = True
        def unload(self): self.loaded = False

    svc = Service()
    orch = SimpleNamespace(tasks={"index": object()}, paused=set(), clear_skip_cache=lambda *_: None, dependency_pipeline_graph=lambda: "Path Pipeline:\n  index")
    reset = []
    db = SimpleNamespace(get_system_stats=lambda: {"tasks": {}}, get_run_stats=lambda: {}, reset_task=lambda n: reset.append(("reset", n)), reset_failed_tasks=lambda n: reset.append(("retry", n)))
    ctx = SimpleNamespace(orchestrator=orch, db=db, services={"svc": svc}, config={"enabled_frontends": ["repl"]})

    assert [s.name for s in TasksCommand().form({}, ctx)] == ["task_name"]
    assert TasksCommand().form({}, ctx)[0].enum == ["index", "pipeline"]
    assert TasksCommand().form({}, ctx)[0].required
    assert [s.name for s in TasksCommand().form({"task_name": "index"}, ctx)] == ["task_name", "action"]
    assert TasksCommand().form({"task_name": "index"}, ctx)[1].enum == ["pause", "unpause", "reset", "retry"]
    assert "Pending:" in TasksCommand().form({"task_name": "index"}, ctx)[1].prompt
    assert TasksCommand().run({"task_name": "index", "action": "pause"}, ctx) == "Paused task: index"
    assert "index" in orch.paused
    assert TasksCommand().run({"task_name": "index", "action": "reset"}, ctx) == "Reset task: index"
    assert TasksCommand().run({"task_name": "index", "action": "retry"}, ctx) == "Retried failed entries for task: index"
    assert reset == [("reset", "index"), ("retry", "index")]
    assert TasksCommand().form({"task_name": "pipeline"}, ctx)[0].enum == ["index", "pipeline"]
    assert TasksCommand().run({"task_name": "pipeline"}, ctx) == "Path Pipeline:\n  index"
    assert ServicesCommand().form({"service_name": "svc"}, ctx)[1].enum == ["load", "unload"]
    assert ServicesCommand().run({"service_name": "svc", "action": "load"}, ctx) == "Loaded service: svc"
    assert svc.loaded
    assert FrontendsCommand().form({"frontend_name": "repl"}, ctx)[1].enum == ["enable", "disable"]
    with patch("config.config_manager.save"):
        assert FrontendsCommand().run({"frontend_name": "repl", "action": "disable"}, ctx) == "Cannot disable the last enabled frontend."


def test_tasks_trigger_only_for_event_tasks():
    from plugins.commands.command_tasks import TasksCommand

    created = []
    path_task = SimpleNamespace(name="path", trigger="path", requires_services=[], trigger_channels=[])
    event_task = SimpleNamespace(
        name="event",
        trigger="event",
        requires_services=[],
        trigger_channels=["manual"],
        event_payload_schema={"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
    )
    db = SimpleNamespace(
        get_system_stats=lambda: {"tasks": {}},
        get_run_stats=lambda: {},
        create_run=lambda *args, **kwargs: created.append((args, kwargs)),
    )
    orch = SimpleNamespace(
        tasks={"path": path_task, "event": event_task},
        paused=set(),
        clear_skip_cache=lambda *_: None,
        on_run_enqueued=lambda run_id, task_name: created.append(((run_id, task_name), {"enqueued": True})),
    )
    tk = SimpleNamespace(
        loaded=True,
        list_jobs=lambda: {"daily-event": {"channel": "manual", "cron": "0 9 * * *", "enabled": True}},
        cron_to_text=lambda _cron: "every day at 09:00",
        get_next_fire_at=lambda _name: None,
    )
    ctx = SimpleNamespace(orchestrator=orch, db=db, services={"timekeeper": tk})

    assert TasksCommand().form({"task_name": "path"}, ctx)[1].enum == ["pause", "unpause", "reset", "retry"]
    event_action = TasksCommand().form({"task_name": "event"}, ctx)[1]
    assert event_action.enum == ["pause", "unpause", "trigger", "schedule", "unschedule"]
    assert "daily-event" in event_action.prompt
    assert [s.name for s in TasksCommand().form({"task_name": "event", "action": "trigger"}, ctx)] == ["task_name", "action", "prompt"]
    assert TasksCommand().run({"task_name": "event", "action": "trigger", "prompt": "go"}, ctx).startswith("Triggered task: event")
    assert '"prompt": "go"' in created[0][1]["payload_json"]


def test_tools_call_action_uses_tool_schema_form():
    from plugins.BaseTool import ToolResult
    from plugins.commands.command_tools import ToolsCommand

    class Tool:
        name = "echo"
        requires_services = []
        def to_schema(self):
            return {"function": {"name": "echo", "description": "Echo text", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "text": {"type": "string"}}, "required": ["name", "text"]}}}

    calls = []
    registry = SimpleNamespace(
        tools={"echo": Tool()},
        call=lambda tool_name, **kwargs: calls.append((tool_name, kwargs)) or ToolResult(data={"ok": kwargs}, llm_summary="echoed"),
    )
    ctx = SimpleNamespace(tool_registry=registry)

    assert [s.name for s in ToolsCommand().form({}, ctx)] == ["tool_name"]
    assert [s.name for s in ToolsCommand().form({"tool_name": "echo"}, ctx)] == ["tool_name", "action"]
    assert [s.name for s in ToolsCommand().form({"tool_name": "echo", "action": "call"}, ctx)] == ["tool_name", "action", "name", "text"]
    assert '"text": "hi"' in ToolsCommand().run({"tool_name": "echo", "action": "call", "name": "doc", "text": "hi"}, ctx)
    assert calls == [("echo", {"name": "doc", "text": "hi"})]


def test_profile_commands_use_add_then_item_actions():
    from plugins.commands.command_agent import AgentCommand
    from plugins.commands.command_llm import LlmCommand

    ctx = SimpleNamespace(
        config={
            "llm_profiles": {"gpt": {"llm_service_class": "OpenAILLM", "llm_context_size": 1}},
            "default_llm_profile": "gpt",
            "agent_profiles": {"default": {"llm": "default", "prompt_suffix": "", "whitelist_or_blacklist_tools": "blacklist", "tools_list": []}},
            "active_agent_profile": "default",
        },
        services={"gpt": SimpleNamespace(loaded=True), "llm": SimpleNamespace(add_llm=lambda *_: None, remove_llm=lambda *_: None)},
        tool_registry=SimpleNamespace(tools={"read_file": object()}),
    )

    assert LlmCommand().form({}, ctx)[0].enum == ["gpt", "add"]
    assert LlmCommand().form({"model_name": "gpt"}, ctx)[1].enum == ["edit", "remove"]
    assert AgentCommand().form({}, ctx)[0].enum == ["default", "add"]
    assert AgentCommand().form({"profile_name": "default"}, ctx)[1].enum == ["switch", "edit", "remove"]
    with patch("config.config_manager.save"), patch("config.config_manager.load_plugin_config", return_value={}), patch("config.config_manager.save_plugin_config"):
        assert AgentCommand().run({"profile_name": "add", "new_profile_name": "builder", "llm": "default", "prompt_suffix": "", "whitelist_or_blacklist_tools": "blacklist", "tools_list": []}, ctx) == "Added agent profile: builder"
        assert LlmCommand().run({"model_name": "gpt", "action": "edit", "field": "llm_context_size", "value": "2"}, ctx) == "Updated LLM profile: gpt"
    assert ctx.config["agent_profiles"]["builder"]["llm"] == "default"
    assert ctx.config["llm_profiles"]["gpt"]["llm_context_size"] == 2


def test_config_command_uses_pick_then_edit_pattern():
    from plugins.commands.command_config import ConfigCommand

    refreshed = []
    ctx = SimpleNamespace(config={"max_workers": 4, "plugin_rate": 1}, runtime=SimpleNamespace(refresh_session_specs=lambda: refreshed.append(True)))
    with patch("plugins.commands.command_config.get_plugin_settings", return_value=[("Plugin Rate", "plugin_rate", "Plugin setting", 1, {})]), \
         patch("config.config_manager.save"), patch("config.config_manager.load_plugin_config", return_value={}), patch("config.config_manager.save_plugin_config") as save_plugin:
        steps = ConfigCommand().form({}, ctx)
        assert steps[0].columns == 2
        assert "max_workers" in steps[0].enum and "plugin_rate" in steps[0].enum
        assert ConfigCommand().form({"setting_name": "max_workers"}, ctx)[1].enum == ["edit"]
        assert ConfigCommand().run({"setting_name": "plugin_rate", "action": "edit", "value": "2"}, ctx) == "Set plugin_rate = 2"
        save_plugin.assert_called_once_with({"plugin_rate": 2})
    assert ctx.config["plugin_rate"] == 2
    assert refreshed == [True]


def test_agent_switch_uses_runtime_session_profile():
    from plugins.commands.command_agent import AgentCommand

    runtime = SimpleNamespace(calls=[], set_agent_profile=lambda session_key, profile: runtime.calls.append((session_key, profile)) or True)
    ctx = SimpleNamespace(
        runtime=runtime,
        session_key="chat:1",
        config={"agent_profiles": {"default": {}, "builder": {}}},
    )

    assert AgentCommand().run({"profile_name": "builder", "action": "switch"}, ctx) == "Switched agent profile to: builder"
    assert runtime.calls == [("chat:1", "builder")]


def test_runtime_agent_switch_keeps_session_override_metadata():
    from state_machine.runtime import ConversationRuntime

    runtime = ConversationRuntime(config={"active_agent_profile": "default"})
    session = runtime.get_session("s")

    assert runtime.set_agent_profile("s", "builder")
    assert session.profile_override == "builder"
    assert session.active_agent_profile == "builder"


def test_agent_llm_edit_reroutes_active_session():
    from plugins.commands.command_agent import AgentCommand
    from state_machine.runtime import ConversationRuntime

    config = {
        "agent_profiles": {"builder": {"llm": "old", "prompt_suffix": "", "whitelist_or_blacklist_tools": "blacklist", "tools_list": []}},
        "active_agent_profile": "default",
        "default_llm_profile": "old",
    }
    runtime = ConversationRuntime(config=config, services={"old": object(), "new": object()})
    runtime.get_session("s")
    runtime.set_agent_profile("s", "builder")
    ctx = SimpleNamespace(config=config, runtime=runtime, session_key="s")

    with patch("config.config_manager.save"):
        assert AgentCommand().run({"profile_name": "builder", "action": "edit", "field": "llm", "value": "new"}, ctx) == "Updated agent profile: builder"

    assert runtime._active_llm(runtime.sessions["s"]) is runtime.services["new"]


def test_telegram_form_echo_shows_command_fragment():
    from plugins.frontends.frontend_telegram import TelegramFrontend

    tg = TelegramFrontend()
    form = {
        "name": "agent",
        "action_type": "call_command",
        "collected": {"profile_name": "default"},
        "field": {"name": "action", "enum": ["edit", "remove"]},
    }

    assert tg._form_echo(form, "edit") == "/agent default edit"


def test_telegram_enum_markup_honors_form_columns():
    from plugins.frontends.frontend_telegram import TelegramFrontend

    markup = TelegramFrontend()._enum_markup("s", {"field": {"enum": ["a", "b", "c"], "columns": 2}})

    assert [[b.text for b in row] for row in markup.inline_keyboard] == [["a", "b"], ["c"], ["Cancel"]]


class FakeCommand(BaseCommand):
    name = "fake"
    description = "fake"

    def form(self, args, context):
        steps = [FormStep("tool_name", "Tool", True, enum=sorted(context.tool_registry.tools))]
        if args.get("tool_name") == "echo":
            steps.append(FormStep("text", "Text", True))
        return steps

    def run(self, args, context):
        context.calls.append(args)
        return args["text"]


def test_context_aware_command_form_uses_runtime_state():
    context = SimpleNamespace(tool_registry=SimpleNamespace(tools={"echo": object()}), calls=[])
    registry = CommandRegistry(lambda _session_key=None: context)
    registry.register(FakeCommand())

    spec = registry.to_callable_specs()["fake"]
    steps = spec.form_factory({}, SimpleNamespace(cache={"session_key": "s"}))

    assert steps[0].enum == ["echo"]


def test_dispatch_dict_invokes_command_with_structured_args():
    context = SimpleNamespace(tool_registry=SimpleNamespace(tools={"echo": object()}), calls=[])
    registry = CommandRegistry(lambda _session_key=None: context)
    registry.register(FakeCommand())

    assert registry.dispatch_dict("fake", {"tool_name": "echo", "text": "hi"}, _emit=False) == "hi"
    assert context.calls == [{"tool_name": "echo", "text": "hi"}]


def test_parse_command_line_handles_dynamic_forms_json_and_text():
    def form(args, _context):
        steps = [FormStep("subcommand", "Subcommand", True, enum=["run"])]
        if args.get("subcommand") == "run":
            steps += [FormStep("payload", "Payload", True, "object"), FormStep("text", "Text", True)]
        return steps

    args = parse_command_line('run {"x": 1, "y": ["z"]} hello there', form)

    assert args == {"subcommand": "run", "payload": {"x": 1, "y": ["z"]}, "text": "hello there"}


def test_slash_command_tool_uses_dispatch_dict_once():
    context = SimpleNamespace(tool_registry=SimpleNamespace(tools={"echo": object()}), calls=[])
    registry = CommandRegistry(lambda _session_key=None: context)
    registry.register(FakeCommand())
    runtime = SimpleNamespace(command_registry=registry, sessions={"s": object()})
    tool_context = SimpleNamespace(runtime=runtime, tool_registry=SimpleNamespace(runtime=runtime), approve_command=lambda *_: True)

    result = SlashCommand().run(tool_context, name="fake", args={"tool_name": "echo", "text": "done"})

    assert result.success
    assert result.data == {"command": "fake", "output": "done"}
    assert context.calls == [{"tool_name": "echo", "text": "done"}]


def test_slash_command_tool_blocks_approval_required_commands():
    cmd = FakeCommand()
    cmd.require_approval = True
    registry = CommandRegistry()
    registry.register(cmd)
    runtime = SimpleNamespace(command_registry=registry, sessions={})

    result = SlashCommand().run(SimpleNamespace(runtime=runtime), name="fake", args={})

    assert not result.success
    assert "requires user approval" in result.error
