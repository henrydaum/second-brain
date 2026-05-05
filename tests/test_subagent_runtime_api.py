from types import SimpleNamespace

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
        assert [cmd.name for cmd in registry.all_commands()] == ["cancel", "help"]
    finally:
        discovery._COMMAND_CONFIG["sandbox_dir"] = old_sandbox


def test_host_commands_are_visible_and_user_approved():
    from pathlib import Path
    from plugins.frontends.bootstrap import _conversation_runtime

    class ToolRegistry:
        tools = {}
        def get_all_schemas(self): return []

    ctrl = SimpleNamespace(
        db=None,
        config={},
        orchestrator=SimpleNamespace(tasks={}, runtime=None),
        maybe_generate_conversation_title_async=None,
        restart=lambda: None,
    )
    runtime = _conversation_runtime(ctrl, lambda: None, ToolRegistry(), {}, {}, Path("."))

    names = sorted(runtime.command_registry._commands)
    visible = [cmd.name for cmd in runtime.command_registry.visible_commands()]

    assert names == ["cancel", "help", "quit", "restart"]
    assert visible == ["cancel", "help", "quit", "restart"]
    assert not runtime.commands["cancel"].require_approval
    assert not runtime.commands["help"].require_approval
    assert runtime.commands["quit"].require_approval
    assert runtime.commands["quit"].approval_actor_id == "user"
    assert runtime.commands["restart"].require_approval
    assert runtime.commands["restart"].approval_actor_id == "user"

    help_text = runtime.command_registry.dispatch_dict("help", {}, session_key="default", _emit=False)
    assert "/cancel" in help_text
    assert "/help" in help_text
    assert "/quit" in help_text
    assert "/restart" in help_text


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
