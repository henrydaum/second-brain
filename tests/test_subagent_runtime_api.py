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
        assert [cmd.name for cmd in registry.all_commands()] == ["agent", "cancel", "commands", "frontends", "llm", "services", "tasks", "tools"]
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

    assert names == ["agent", "cancel", "commands", "frontends", "llm", "quit", "restart", "services", "tasks", "tools"]
    assert visible == ["agent", "cancel", "commands", "frontends", "llm", "quit", "restart", "services", "tasks", "tools"]
    assert not runtime.commands["cancel"].require_approval
    assert not runtime.commands["commands"].require_approval
    assert runtime.commands["quit"].require_approval
    assert runtime.commands["quit"].approval_actor_id == "user"
    assert runtime.commands["restart"].require_approval
    assert runtime.commands["restart"].approval_actor_id == "user"

    text = runtime.command_registry.dispatch_dict("commands", {}, session_key="default", _emit=False)
    for name in names:
        assert f"/{name}" in text


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
    orch = SimpleNamespace(tasks={"index": object()}, paused=set(), clear_skip_cache=lambda *_: None)
    db = SimpleNamespace(get_system_stats=lambda: {"tasks": {}}, get_run_stats=lambda: {})
    ctx = SimpleNamespace(orchestrator=orch, db=db, services={"svc": svc}, config={"enabled_frontends": ["repl"]})

    assert [s.name for s in TasksCommand().form({}, ctx)] == ["task_name"]
    assert TasksCommand().form({}, ctx)[0].required
    assert [s.name for s in TasksCommand().form({"task_name": "index"}, ctx)] == ["task_name", "action"]
    assert TasksCommand().form({"task_name": "index"}, ctx)[1].enum == ["pause", "unpause"]
    assert "Pending:" in TasksCommand().form({"task_name": "index"}, ctx)[1].prompt
    assert TasksCommand().run({"task_name": "index", "action": "pause"}, ctx) == "Paused task: index"
    assert "index" in orch.paused
    assert ServicesCommand().form({"service_name": "svc"}, ctx)[1].enum == ["load", "unload"]
    assert ServicesCommand().run({"service_name": "svc", "action": "load"}, ctx) == "Loaded service: svc"
    assert svc.loaded
    assert FrontendsCommand().form({"frontend_name": "repl"}, ctx)[1].enum == ["enable", "disable"]
    with patch("config.config_manager.save"):
        assert FrontendsCommand().run({"frontend_name": "repl", "action": "disable"}, ctx) == "Cannot disable the last enabled frontend."


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
    from plugins.frontends.telegram_frontend import TelegramFrontend

    tg = TelegramFrontend()
    form = {
        "name": "agent",
        "action_type": "call_command",
        "collected": {"profile_name": "default"},
        "field": {"name": "action", "enum": ["edit", "remove"]},
    }

    assert tg._form_echo(form, "edit") == "/agent default edit"


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
    tool_context = SimpleNamespace(runtime=runtime, tool_registry=SimpleNamespace(runtime=runtime))

    result = SlashCommand().run(tool_context, name="fake", args={"tool_name": "echo", "text": "done"})

    assert result.success
    assert result.data == {"command": "fake", "output": "done"}
    assert context.calls == [{"tool_name": "echo", "text": "done"}]
