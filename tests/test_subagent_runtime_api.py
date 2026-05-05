from types import SimpleNamespace
import ast
import inspect
import textwrap

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


def test_message_command_injects_user_message_through_runtime():
    class Timekeeper:
        loaded = True

        def __init__(self):
            self.payload = {"title": "Job"}

        def get_job(self, name):
            return {"channel": "subagent.run", "payload": self.payload} if name == "job" else None

        def update_job(self, name, patch):
            self.payload = patch["payload"]

    runtime = FakeRuntime()
    ctrl = SimpleNamespace(
        db=SimpleNamespace(get_conversation=lambda _cid: None, count_pending_inbox=lambda _cid: 1),
        config={},
        list_task_names=lambda *a, **k: [],
        list_tasks=lambda: [],
        frontend_runtime=runtime,
        orchestrator=SimpleNamespace(tasks={}),
    )
    tool_registry = SimpleNamespace(tools={}, runtime=runtime, get_schema=lambda _name: None)
    registry = CommandRegistry(lambda _session_key=None: SimpleNamespace(
        controller=ctrl,
        db=ctrl.db,
        config=ctrl.config,
        services={"timekeeper": Timekeeper()},
        tool_registry=tool_registry,
        orchestrator=ctrl.orchestrator,
        runtime=runtime,
        root_dir=".",
    ))
    discover_commands(".", registry)

    output = registry.dispatch_dict("message", {"job_name": "job", "text": "hello there"}, _emit=False)

    assert "Queued for 'job'" in output
    assert ("create_conversation", "Job", "subagent") in runtime.calls
    assert ("inject_user_message", "subagent:job", "hello there", 7, "user") in runtime.calls
    assert ("unload_conversation", "subagent:job") in runtime.calls


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


def test_builtin_command_handlers_read_form_fields():
    from plugins.BaseCommand import BaseCommand
    from plugins.commands import command_core

    class Ctrl:
        config = {}
        db = SimpleNamespace()
        orchestrator = SimpleNamespace(tasks={}, dependency_pipeline_graph=lambda: "")
        def list_task_names(self, trigger=None): return []

    ctx = SimpleNamespace(
        controller=Ctrl(),
        config={},
        services={},
        tool_registry=SimpleNamespace(tools={}, get_schema=lambda _name: None),
        orchestrator=SimpleNamespace(tasks={}),
    )

    def command_classes(cls=BaseCommand):
        for sub in cls.__subclasses__():
            yield sub
            yield from command_classes(sub)

    for cls in command_classes():
        if cls.__module__ != command_core.__name__ or not getattr(cls, "name", ""):
            continue
        command = cls()
        forms = [command.form({}, ctx)]
        first = forms[0][0] if forms[0] else None
        if first and first.name == "subcommand":
            forms += [command.form({"subcommand": sub}, ctx) for sub in first.enum or []]
        form_keys = {step.name for form in forms for step in form}
        source = inspect.getsource(cls.run)
        tree = ast.parse(textwrap.dedent(source))
        read_keys = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "args"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        read_keys |= {
            node.slice.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        assert read_keys <= form_keys, f"{cls.__name__} reads keys not in form: {read_keys - form_keys}"
