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
                 "command_status", "tool_broken"):
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
        return sdk.ok(sdk.config.read(key).data)
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


def test_an_unbridged_family_declines_cleanly(tmp_path, box):
    """Services and frontends are not bridged yet; say so, do not guess."""
    source = '''
"""A migrated service."""

from guest.bases import BaseService


class Counter(BaseService):
    """Counts."""

    name = "counter"
    exports = ["total"]

    def start(self, sdk):
        """Start."""
        return True
'''
    assert adapt(_write(tmp_path, "service_counter.py", source)) is None


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
