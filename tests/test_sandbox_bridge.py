"""The dual-mode loader: migrated and unmigrated plugins side by side.

The claim under test is the one the whole migration plan rests on — that a
migrated plugin is indistinguishable from a native one to everything
downstream, so the app keeps working with any mix of the two.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.bridge import adapt, configure, family_of, is_sandboxed

MIGRATED_TOOL = '''
"""A migrated tool."""

from guest.bases import BaseTool


class WordCount(BaseTool):
    """Count words in some text."""

    name = "word_count"
    description = "Count the words in a string."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    max_calls = 5

    def run(self, sdk, text=""):
        """Count them."""
        words = len(text.split())
        return sdk.ok({"words": words}, llm_summary=f"{words} words")
'''

NATIVE_TOOL = '''
"""An unmigrated tool."""

from plugins.BaseTool import BaseTool, ToolResult


class Shout(BaseTool):
    """Uppercase some text."""

    name = "shout"
    description = "Uppercase a string."

    def run(self, context, text=""):
        """Shout it."""
        return ToolResult(data=text.upper())
'''

MIGRATED_TASK = '''
"""A migrated task."""

from guest.bases import BaseTask


class Extract(BaseTask):
    """Pretend to extract text."""

    name = "extract"
    writes = ["text_docs"]

    def run(self, sdk, paths):
        """Report what it was given."""
        return sdk.ok([{"path": p} for p in paths],
                      discovered_paths=["/found/extra.txt"])
'''

MIGRATED_COMMAND = '''
"""A migrated command."""

from guest.bases import BaseCommand


class Status(BaseCommand):
    """Report status."""

    name = "status"
    description = "Show status."

    def run(self, sdk, args):
        """Render markdown."""
        return sdk.ok(f"**Status:** {args.get('mode', 'normal')}")
'''


@pytest.fixture
def box():
    """A sandbox the bridge routes migrated plugins through."""
    made = Sandbox()
    configure(made)
    yield made
    configure(None)
    made.shutdown()


@pytest.fixture(autouse=True)
def clean_boxes():
    """Boxes are module caches; a leak hides staleness."""
    yield
    for name in ("tool_word_count", "tool_shout", "task_extract",
                 "command_status", "tool_broken", "service_counter",
                 "command_deploy", "frontend_web"):
        unload_box(name)


def _write(tmp_path, filename, source):
    """Put a plugin file on disk."""
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────
# Telling them apart, without importing either.
# ──────────────────────────────────────────────────────────────────────

def test_a_migrated_plugin_is_recognised(tmp_path):
    """Detection is by which contract the file imports."""
    assert is_sandboxed(_write(tmp_path, "tool_word_count.py", MIGRATED_TOOL))


def test_a_native_plugin_is_left_alone(tmp_path):
    """An unmigrated plugin must route the ordinary way."""
    path = _write(tmp_path, "tool_shout.py", NATIVE_TOOL)
    assert not is_sandboxed(path)
    assert adapt(path) is None


def test_detection_never_imports_the_file(tmp_path):
    """Asking the question must not run anything."""
    marker = tmp_path / "ran.txt"
    path = _write(tmp_path, "tool_evil.py",
                  f"open({str(marker)!r}, 'w').write('x')\n")
    is_sandboxed(path)
    assert not marker.exists()


def test_family_comes_from_the_filename(tmp_path):
    """Discovery finds plugins by filename, so the bridge agrees with it."""
    assert family_of("tool_x.py") == "tool"
    assert family_of("command_y.py") == "command"
    assert family_of("helper_z.py") == ""


# ──────────────────────────────────────────────────────────────────────
# The adapter looks native.
# ──────────────────────────────────────────────────────────────────────

def test_the_adapter_subclasses_the_native_base(tmp_path, box):
    """The registry type-checks against BaseTool; the adapter must pass."""
    from plugins.BaseTool import BaseTool

    module = adapt(_write(tmp_path, "tool_word_count.py", MIGRATED_TOOL))
    adapter = next(v for v in vars(module).values()
                   if isinstance(v, type) and issubclass(v, BaseTool))
    assert issubclass(adapter, BaseTool)
    assert adapter()._sandboxed is True


def test_declarations_are_carried_onto_the_adapter(tmp_path, box):
    """A plugin advertised with no schema is a plugin the agent cannot call."""
    module = adapt(_write(tmp_path, "tool_word_count.py", MIGRATED_TOOL))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    assert instance.name == "word_count"
    assert instance.description == "Count the words in a string."
    assert instance.parameters["required"] == ["text"]
    assert instance.max_calls == 5


def test_a_tool_runs_through_the_adapter(tmp_path, box):
    """The native contract in, a ToolResult out."""
    module = adapt(_write(tmp_path, "tool_word_count.py", MIGRATED_TOOL))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    outcome = instance.run(SimpleNamespace(config={}), text="one two three")
    assert outcome.success
    assert outcome.data == {"words": 3}
    assert outcome.llm_summary == "3 words"


def test_a_failing_tool_becomes_a_failed_toolresult(tmp_path, box):
    """Failure has to translate too, not just success."""
    source = MIGRATED_TOOL.replace(
        'return sdk.ok({"words": words}, llm_summary=f"{words} words")',
        'return sdk.fail("no text given")')
    module = adapt(_write(tmp_path, "tool_word_count.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    outcome = instance.run(SimpleNamespace(config={}), text="x")
    assert not outcome.success
    assert "no text given" in outcome.error


# ──────────────────────────────────────────────────────────────────────
# The families disagree about argument order.
# ──────────────────────────────────────────────────────────────────────

def test_a_task_gets_its_paths_not_its_context(tmp_path, box):
    """run(paths, context) - a generic signature would bind these wrongly."""
    module = adapt(_write(tmp_path, "task_extract.py", MIGRATED_TASK))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    outcome = instance.run(["/a.txt", "/b.txt"], SimpleNamespace(config={}))
    assert outcome.success
    assert outcome.data == [{"path": "/a.txt"}, {"path": "/b.txt"}]
    assert outcome.discovered_paths == ["/found/extra.txt"]


def test_a_command_returns_markdown(tmp_path, box):
    """Commands answer with a string, not a Result."""
    module = adapt(_write(tmp_path, "command_status.py", MIGRATED_COMMAND))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    rendered = instance.run({"mode": "quiet"}, SimpleNamespace(config={}))
    assert rendered == "**Status:** quiet"


# ──────────────────────────────────────────────────────────────────────
# The context travels per call.
# ──────────────────────────────────────────────────────────────────────

def test_each_call_answers_from_its_own_context(tmp_path, box):
    """Two sessions in flight must not answer from each other's world."""
    source = '''
"""Reads a setting."""

from guest.bases import BaseTool


class ReadSetting(BaseTool):
    """Read one setting."""

    name = "read_setting"

    def run(self, sdk, key=""):
        """Through the gate."""
        return sdk.config.read(key)
'''
    module = adapt(_write(tmp_path, "tool_word_count.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    first = instance.run(SimpleNamespace(config={"model": "opus"}), key="model")
    second = instance.run(SimpleNamespace(config={"model": "haiku"}),
                          key="model")
    assert first.data == "opus"
    assert second.data == "haiku"


def test_the_chain_records_what_caused_the_call(tmp_path, box):
    """Provenance has to survive the bridge, or the dialog loses its root."""
    seen = []
    box.interpreter._record = lambda chain, req, dec, res: seen.append(
        chain.render())

    source = MIGRATED_TOOL.replace(
        'words = len(text.split())',
        'sdk.fs.list(".")\n        words = len(text.split())')
    module = adapt(_write(tmp_path, "tool_word_count.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    instance.run(SimpleNamespace(config={}, user_initiated=True), text="a")
    assert seen
    # Exact, not startswith: a doubled push renders as
    # "user -> word_count -> word_count" and would pass a prefix check while
    # tripping the cycle detector on every Request.
    assert seen[0] == "user -> word_count"


# ──────────────────────────────────────────────────────────────────────
# Refusing to load something broken.
# ──────────────────────────────────────────────────────────────────────

def test_an_invalid_migrated_plugin_does_not_load(tmp_path, box):
    """The validator gates loading, not just advises."""
    source = MIGRATED_TOOL.replace(
        "words = len(text.split())", "words = len(open(text).read())")
    assert adapt(_write(tmp_path, "tool_broken.py", source)) is None


def test_a_file_that_is_not_a_plugin_declines_cleanly(tmp_path, box):
    """A file naming no plugin class is a helper or a script, not a plugin.

    All five families are bridged now, so what is left to decline is a file
    the bridge cannot find an entry point in — and it must say so rather than
    build an adapter around nothing.
    """
    source = '''
"""Helpers, not a plugin."""

from guest.bases import BaseFrontend


def helper(sdk):
    """Do something."""
    return 1
'''
    assert adapt(_write(tmp_path, "frontend_web.py", source)) is None


# ──────────────────────────────────────────────────────────────────────
# Services: a residency rather than a call.
# ──────────────────────────────────────────────────────────────────────

MIGRATED_SERVICE = '''
"""A migrated service."""

from guest.bases import BaseService


class Counter(BaseService):
    """Counts things, and remembers between calls."""

    name = "counter"
    exports = ["bump", "total"]
    ISOLATION

    def start(self, sdk):
        """Begin at zero."""
        self._n = 0
        return True

    def bump(self, sdk, by=1):
        """Add to the counter."""
        self._n += by
        return self._n

    def total(self, sdk):
        """Read the counter."""
        return self._n

    def internal(self, sdk):
        """Deliberately not exported."""
        return "unreachable"

    def stop(self, sdk):
        """Forget."""
        self._n = 0
'''


def _service(tmp_path, isolation=""):
    """Build and instantiate a migrated service the way discovery would."""
    source = MIGRATED_SERVICE.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    module = adapt(_write(tmp_path, "service_counter.py", source))
    # Services are found by calling build_services, not by scanning classes,
    # so the synthetic module has to provide one.
    return module.build_services({})["counter"]


@pytest.mark.parametrize("isolation", ["", "subprocess"])
def test_a_migrated_service_keeps_state_between_calls(tmp_path, box, isolation):
    """The point of a service: the box stays open and remembers.

    Both runners, because the promise is that isolation changes nothing a
    caller can observe.
    """
    service = _service(tmp_path, isolation)
    assert service.load() is True
    assert service.loaded is True

    assert service.bump() == 1
    assert service.bump(by=5) == 6
    assert service.total() == 6          # state survived three separate calls

    service.unload()
    assert service.loaded is False


def test_a_migrated_service_looks_native(tmp_path, box):
    """Native callers reach services by attribute access, not .call()."""
    service = _service(tmp_path)
    from plugins.BaseService import BaseService

    assert isinstance(service, BaseService)
    # Named the way the native side names services.
    assert service.model_name == "counter"
    # The box owns the start deadline; a second timer would race it.
    assert service.load_timeout == 0


def test_only_exported_methods_exist(tmp_path, box):
    """``exports`` is the public surface, and it is enforced by absence."""
    service = _service(tmp_path)
    assert callable(getattr(service, "bump", None))
    assert callable(getattr(service, "total", None))
    assert not hasattr(service, "internal")
    # Carried onto the adapter so handlers._service_call refuses unexported
    # methods when the caller is other sandboxed code rather than the kernel.
    assert service.exports == ["bump", "total"]


def test_calling_an_unloaded_service_fails_clearly(tmp_path, box):
    """The failure names the service, rather than surfacing a None box."""
    from sandbox.bridge import ServiceCallFailed

    service = _service(tmp_path)
    with pytest.raises(ServiceCallFailed, match="not loaded"):
        service.bump()

    service.load()
    service.unload()
    with pytest.raises(ServiceCallFailed, match="not loaded"):
        service.bump()


def test_a_failing_export_raises_rather_than_returning(tmp_path, box):
    """Native callers expect a value or an exception, never a Result."""
    from sandbox.bridge import ServiceCallFailed

    source = MIGRATED_SERVICE.replace("ISOLATION", "").replace(
        "self._n += by\n        return self._n",
        "raise ValueError('nope')")
    module = adapt(_write(tmp_path, "service_counter.py", source))
    service = module.build_services({})["counter"]
    service.load()
    with pytest.raises(ServiceCallFailed, match="nope"):
        service.bump()
    service.unload()


def test_unloading_closes_the_box(tmp_path, box):
    """A reloaded service must not leave its old box resident.

    This is what makes the watcher's edit-a-service path safe: discovery
    calls unload() on the old instance, and the box has to go with it or the
    next load talks to a stale process.
    """
    service = _service(tmp_path)
    service.load()
    assert box.box("service_counter") is not None
    service.unload()
    assert box.box("service_counter") is None


# ──────────────────────────────────────────────────────────────────────
# Both kinds coexist in one registry.
# ──────────────────────────────────────────────────────────────────────

def test_migrated_and_native_tools_register_together(tmp_path, box):
    """The claim the migration plan rests on."""
    from agent.tool_registry import ToolRegistry
    from plugins.BaseTool import BaseTool
    from plugins.plugin_discovery import _load_plugin_module

    migrated = _write(tmp_path, "tool_word_count.py", MIGRATED_TOOL)
    native = _write(tmp_path, "tool_shout.py", NATIVE_TOOL)

    registry = ToolRegistry(None, {}, {})
    for path, module_name in ((migrated, "sandboxed_tool_word_count"),
                              (native, "native_tool_shout")):
        module = _load_plugin_module(module_name, path, False, False)
        assert module is not None, path.name
        for value in vars(module).values():
            if (isinstance(value, type) and issubclass(value, BaseTool)
                    and value is not BaseTool):
                registry.register(value())

    assert sorted(registry.list_tools()) == ["shout", "word_count"]

    # And both are callable through the one registry, indistinguishably.
    assert registry.call("shout", text="hey").data == "HEY"
    assert registry.call("word_count", text="a b c").data == {"words": 3}


def test_a_command_form_is_bridged_too(tmp_path, box):
    """A command whose form vanished would silently stop collecting args."""
    source = '''
"""A migrated command with a form."""

from guest.bases import BaseCommand


class Deploy(BaseCommand):
    """Deploy something."""

    name = "deploy"

    def form(self, sdk, args):
        """Ask for the target if it was not given."""
        if args.get("target"):
            return []
        return [{"name": "target", "prompt": "Where to?", "nonsense": 1},
                {"prompt": "no name, so unusable"}]

    def run(self, sdk, args):
        """Do it."""
        return f"deploying {args['target']}"
'''
    module = adapt(_write(tmp_path, "command_deploy.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    # Steps cross the boundary as data and must come back as real FormSteps:
    # the command registry reads step.name and calls step.coerce, so a bare
    # dict would fail at the point of use rather than here.
    steps = instance.form({}, SimpleNamespace(config={}))
    assert [type(s).__name__ for s in steps] == ["FormStep"]
    assert (steps[0].name, steps[0].prompt) == ("target", "Where to?")
    assert instance.form({"target": "prod"}, SimpleNamespace(config={})) == []
    assert instance.run({"target": "prod"},
                        SimpleNamespace(config={})) == "deploying prod"
    unload_box("command_deploy")


def test_a_command_without_a_form_keeps_the_base_default(tmp_path, box):
    """Only forward what the migrated file actually defines."""
    module = adapt(_write(tmp_path, "command_status.py", MIGRATED_COMMAND))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))
    assert instance.form({}, SimpleNamespace(config={})) == []


@pytest.mark.parametrize("filename, source, base_module, base_name", [
    ("tool_word_count.py", MIGRATED_TOOL, "plugins.BaseTool", "BaseTool"),
    ("service_counter.py", MIGRATED_SERVICE.replace("ISOLATION", ""),
     "plugins.BaseService", "BaseService"),
])
def test_a_bridged_plugin_is_visible_to_discovery(tmp_path, box, filename,
                                                  source, base_module,
                                                  base_name):
    """The adapter has to be *findable*, not merely built.

    Discovery only accepts classes belonging to the module it just loaded, and
    a ``type()``-made class claims the module ``type()`` ran in — so every
    adapter used to look foreign and no migrated plugin could be discovered at
    all. The bridge worked; nothing could reach it. Every test here called
    ``adapt`` directly, which is exactly the step that hid it.
    """
    from plugins.plugin_discovery import _find_subclasses

    base = getattr(__import__(base_module, fromlist=[base_name]), base_name)
    module = adapt(_write(tmp_path, filename, source))

    # The name discovery would have asked for, which is *not* the synthetic
    # module's name — that difference is the whole bug.
    found = _find_subclasses(module, base, f"plugins.x.{Path(filename).stem}")
    assert found, "a bridged plugin was invisible to discovery"
    assert issubclass(found[0], base)
