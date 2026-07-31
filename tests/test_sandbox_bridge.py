"""The bridge: SDK code wearing a native face.

The claim under test is that a sandboxed plugin is indistinguishable from a
native one to everything downstream — discovery registers it, the registries
type-check it, the state machine calls it, and none of them learn where the
code actually runs.

This used to be a claim about *coexistence*, because the loader would fall
through to an ordinary import for a file that had not been migrated. That is
gone: the migration is over, and a plugin the bridge will not carry is a
plugin that does not load. ``NATIVE_TOOL`` survives here as the fixture for
what refusal looks like.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.bridge import adapt, configure, family_of
from sandbox.validator import ERROR, validate_file

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
                 "command_deploy", "frontend_web", "service_keeper"):
        unload_box(name)


def _write(tmp_path, filename, source):
    """Put a plugin file on disk."""
    path = tmp_path / filename
    path.write_text(source, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────
# What loads, and what does not — decided without importing either.
# ──────────────────────────────────────────────────────────────────────

def test_a_sandboxed_plugin_adapts(tmp_path):
    """SDK code is what the bridge carries."""
    assert adapt(_write(tmp_path, "tool_word_count.py", MIGRATED_TOOL)) is not None


def test_a_native_plugin_is_declined(tmp_path):
    """The bridge answers None, and the loader turns that into a refusal.

    The refusal now comes from the validator rather than from a separate
    detection pass, which is what lets it say something useful: the native
    base class is no longer a contract module, so importing one is reported
    by name with the import that replaces it.
    """
    path = _write(tmp_path, "tool_shout.py", NATIVE_TOOL)
    assert adapt(path) is None

    report = validate_file(path)
    assert not report.ok
    findings = "\n".join(f.render() for f in report.of(ERROR))
    assert "plugins.BaseTool" in findings
    assert "guest.bases" in findings, (
        "refusing a native plugin has to name the one-line fix; the generic "
        "kernel-module error prescribes 'a Request', which is useless here")


def test_deciding_never_imports_the_file(tmp_path):
    """Asking the question must not run anything."""
    marker = tmp_path / "ran.txt"
    path = _write(tmp_path, "tool_evil.py",
                  f"open({str(marker)!r}, 'w').write('x')\n")
    adapt(path)
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
    from plugins.native.tool import BaseTool

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

    outcomes = instance.run(["/a.txt", "/b.txt"], SimpleNamespace(config={}))
    assert all(o.success for o in outcomes)
    assert outcomes[0].data == [{"path": "/a.txt"}, {"path": "/b.txt"}]
    assert outcomes[0].discovered_paths == ["/found/extra.txt"]


def test_a_batch_task_answers_once_per_path(tmp_path, box):
    """The orchestrator zips paths against results, so it needs one each.

    A guest task returns a single Result. Handing that straight over made
    ``zip`` stop after the first path: the rest were neither completed nor
    failed, so they stayed claimed and were never retried — a folder that
    silently stopped indexing, with nothing in the log.
    """
    module = adapt(_write(tmp_path, "task_extract.py", MIGRATED_TASK))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    paths = ["/a.txt", "/b.txt", "/c.txt", "/d.txt"]
    outcomes = instance.run(paths, SimpleNamespace(config={}))

    assert len(outcomes) == len(paths)
    # Rows ride on the first alone. ``_handle_success`` writes the whole of a
    # result's data for whichever path it is handling, so copying them onto
    # every outcome would write the same rows once per path in the batch.
    assert outcomes[0].data
    assert all(o.data == [] for o in outcomes[1:])


def test_a_task_may_report_each_path_separately(tmp_path, box):
    """per_path is how one bad file fails without taking the batch with it."""
    source = '''
"""A task that judges each file."""

from guest.bases import BaseTask


class Extract(BaseTask):
    """Fail the middle one."""

    name = "extract"
    writes = ["text_docs"]

    def run(self, sdk, paths):
        """One entry per path, in order."""
        return sdk.ok(per_path=[
            {"ok": True, "data": [{"path": paths[0]}],
             "also_contains": ["image"]},
            {"ok": False, "error": "unreadable"},
            {"ok": True, "data": [{"path": paths[2]}]},
        ])
'''
    module = adapt(_write(tmp_path, "task_judge.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    a, b, c = instance.run(["/a", "/b", "/c"], SimpleNamespace(config={}))
    assert a.success and a.data == [{"path": "/a"}]
    assert a.also_contains == ["image"]
    assert not b.success and b.error == "unreadable"
    assert c.success and c.data == [{"path": "/c"}]


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
    from plugins.native.service import BaseService

    assert isinstance(service, BaseService)
    # Named the way the native side names services.
    assert service.model_name == "counter"


PROMPTING_SERVICE = '''
"""A migrated service that contributes to the system prompt."""

from guest.bases import BaseService


class Advisor(BaseService):
    """Advise."""

    name = "advisor"
    exports = ["ping"]

    def start(self, sdk):
        """Nothing to set up."""
        self._where = "in the box"
        return True

    def ping(self, sdk):
        """Answer something."""
        return "pong"

    def agent_prompt(self, sdk):
        """Built from state only the residency has, so a literal cannot fake it."""
        return f"Advice from {self._where}."
'''


def test_a_resident_prompt_contribution_is_answered_in_the_box(tmp_path, box):
    """The resident half of the same doorway, and it is not a call like the rest.

    A service is not asked through ``_forward`` — it already owns a box — so the
    forwarding is separate code and would have failed separately and silently.
    The text is built from state established in ``start`` so that only a real
    call into the residency can produce it.
    """
    module = adapt(_write(tmp_path, "service_advisor.py", PROMPTING_SERVICE))
    service = module.build_services({})["advisor"]

    # Not loaded: contributes nothing rather than taking the prompt down. And
    # "nothing" must not become the answer forever — the cache is scoped to one
    # residency, not to the adapter, so loading clears it.
    assert service.agent_prompt(SimpleNamespace(config={})) == ""

    assert service.load() is True
    try:
        assert service.agent_prompt(
            SimpleNamespace(config={})) == "Advice from in the box."
    finally:
        service.unload()


def test_only_exported_methods_exist(tmp_path, box):
    """``exports`` is the public surface, and it is enforced by absence."""
    service = _service(tmp_path)
    assert callable(getattr(service, "bump", None))
    assert callable(getattr(service, "total", None))
    assert not hasattr(service, "internal")
    # Carried onto the adapter so handlers._service_call refuses unexported
    # methods when the caller is other sandboxed code rather than the kernel.
    assert service.exports == ["bump", "total"]


# ──────────────────────────────────────────────────────────────────────
# Several services in one file, sharing one box.
#
# The shape ``build_services`` has always supported, and it exists for a
# reason worth testing rather than merely allowing: two services put in one
# file share something expensive, so they must share the *process* too, and
# unloading one must not take the other's state down with it.
# ──────────────────────────────────────────────────────────────────────

TWO_SERVICES = '''
"""Two services that share a module-level resource."""

from guest.bases import BaseService

LOADS = []


class Alpha(BaseService):
    """First."""

    name = "alpha"
    exports = ["bump", "total", "loads"]

    def start(self, sdk):
        """Begin."""
        LOADS.append("alpha")
        self._n = 0
        return True

    def bump(self, sdk, by=1):
        """Add."""
        self._n += by
        return self._n

    def total(self, sdk):
        """Read."""
        return self._n

    def loads(self, sdk):
        """Who has started in this process."""
        return list(LOADS)


class Beta(BaseService):
    """Second."""

    name = "beta"
    exports = ["bump", "total", "loads"]

    def start(self, sdk):
        """Begin."""
        LOADS.append("beta")
        self._n = 100
        return True

    def bump(self, sdk, by=1):
        """Add."""
        self._n += by
        return self._n

    def total(self, sdk):
        """Read."""
        return self._n

    def loads(self, sdk):
        """Who has started in this process."""
        return list(LOADS)
'''


def _two_services(tmp_path, monkeypatch, isolated):
    """Build both adapters the way discovery would, in a chosen tree.

    Isolation is provenance, so the *tree* is how a test picks a runner —
    ``bundled`` is always in-process and ``workspace`` always a subprocess.
    A declaration would be ignored, which is the point of ``isolation.py``.
    """
    from tests.support import retarget_trees

    roots = retarget_trees(monkeypatch, tmp_path)
    tree = roots["workspace" if isolated else "bundled"]
    services = tree / "services"
    services.mkdir(parents=True, exist_ok=True)
    path = services / "service_pair.py"
    path.write_text(TWO_SERVICES, encoding="utf-8")

    module = adapt(path)
    built = module.build_services({})
    return built["alpha"], built["beta"]


@pytest.mark.parametrize("isolated", [False, True])
def test_two_services_in_one_file_both_register(tmp_path, box, isolated,
                                                monkeypatch):
    """``build_services`` answers with both, as the native version always did."""
    alpha, beta = _two_services(tmp_path, monkeypatch, isolated)
    assert alpha.name == "alpha" and beta.name == "beta"
    assert alpha.load() is True and beta.load() is True
    assert alpha.total() == 0
    assert beta.total() == 100


@pytest.mark.parametrize("isolated", [False, True])
def test_calls_route_to_the_right_occupant(tmp_path, box, isolated,
                                           monkeypatch):
    """Two occupants, one box: a call must not reach the neighbour.

    Both hold a method of the same name over different state, which is the
    case a target-less dispatch gets silently wrong.
    """
    alpha, beta = _two_services(tmp_path, monkeypatch, isolated)
    alpha.load()
    beta.load()

    assert alpha.bump(by=5) == 5
    assert beta.bump(by=5) == 105
    assert alpha.total() == 5          # untouched by beta's bump


@pytest.mark.parametrize("isolated", [False, True])
def test_the_file_is_imported_once_for_both(tmp_path, box, isolated,
                                            monkeypatch):
    """One module import, which is the whole reason to share a file.

    A module-level list both classes append to is the cheapest observable
    proof: two imports would give each occupant its own ``LOADS``.
    """
    alpha, beta = _two_services(tmp_path, monkeypatch, isolated)
    alpha.load()
    beta.load()
    assert alpha.loads() == ["alpha", "beta"] == beta.loads()


@pytest.mark.parametrize("isolated", [False, True])
def test_unloading_one_service_leaves_its_neighbour_running(
        tmp_path, box, isolated, monkeypatch):
    """The refcount. Naively, one ``unload`` closes the shared box.

    That failure has no symptom beyond the survivor's calls suddenly failing,
    and nothing about the survivor changed — which makes it exactly the kind
    of thing to pin.
    """
    alpha, beta = _two_services(tmp_path, monkeypatch, isolated)
    alpha.load()
    beta.load()
    beta.bump(by=7)

    alpha.unload()
    assert alpha.loaded is False
    assert beta.loaded is True
    assert beta.total() == 107          # its box, and its state, survived


def test_the_shared_box_takes_its_slowest_occupants_ceiling(tmp_path, box):
    """One box, one deadline — so it has to fit whoever needs longest.

    Reading the first class's ``timeout`` would silently starve a sibling
    that declared more, and the symptom would be a call that dies at a
    deadline the plugin never asked for.
    """
    source = TWO_SERVICES.replace('    name = "alpha"',
                                  '    name = "alpha"\n    timeout = 30')
    source = source.replace('    name = "beta"',
                            '    name = "beta"\n    timeout = 300')
    path = tmp_path / "service_pair.py"
    path.write_text(source, encoding="utf-8")
    _report, spec = box.inspect(path)
    assert spec.timeout == 300


@pytest.mark.parametrize("isolated", [False, True])
def test_the_box_closes_once_the_last_service_unloads(
        tmp_path, box, isolated, monkeypatch):
    """The other half of the refcount: nothing is left running."""
    alpha, beta = _two_services(tmp_path, monkeypatch, isolated)
    alpha.load()
    beta.load()
    alpha.unload()
    beta.unload()
    assert box.box("service_pair") is None


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
# The loader carries sandboxed code and nothing else.
# ──────────────────────────────────────────────────────────────────────

def test_a_sandboxed_tool_registers_through_the_loader(tmp_path, box):
    """The one way in: adapt, then register the adapter like any tool."""
    from agent.tool_registry import ToolRegistry
    from plugins.native.tool import BaseTool
    from plugins.plugin_discovery import _load_plugin_module

    path = _write(tmp_path, "tool_word_count.py", MIGRATED_TOOL)
    module = _load_plugin_module(path)
    assert module is not None

    registry = ToolRegistry(None, {}, {})
    for value in vars(module).values():
        if (isinstance(value, type) and issubclass(value, BaseTool)
                and value is not BaseTool):
            registry.register(value())

    assert registry.list_tools() == ["word_count"]
    assert registry.call("word_count", text="a b c").data == {"words": 3}


def test_a_native_plugin_does_not_load_at_all(tmp_path, box, caplog):
    """The dual-mode loader is gone, and this is what replaced it.

    A file written against the old contract used to be imported the ordinary
    way and registered beside migrated ones — that coexistence is what made
    the migration survivable one file at a time. Keeping it past the end of
    the migration would only mean unmediated code running in the kernel's own
    process, so it is refused now.

    Refused *and reported*: a plugin that vanishes with no line in the log is
    indistinguishable from one that was never installed, and the whole cost of
    this change lands on somebody wondering where their tool went.
    """
    from plugins.plugin_discovery import _load_plugin_module

    path = _write(tmp_path, "tool_shout.py", NATIVE_TOOL)
    with caplog.at_level("WARNING"):
        module = _load_plugin_module(path)
    assert module is None
    assert "tool_shout.py" in caplog.text
    assert "SDK" in caplog.text


def test_refusing_one_plugin_does_not_abort_the_rest(tmp_path, box):
    """Reported, never raised.

    Discovery loops read ``None`` as "skip this file" with no ``try`` around
    them, so a raise here would let one unmigrated plugin take every other
    plugin's discovery down with it.
    """
    from plugins.plugin_discovery import _load_plugin_module

    native = _write(tmp_path, "tool_shout.py", NATIVE_TOOL)
    migrated = _write(tmp_path, "tool_word_count.py", MIGRATED_TOOL)

    assert _load_plugin_module(native) is None
    assert _load_plugin_module(migrated) is not None


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


PROMPTING_TOOL = '''
"""A migrated tool that contributes to the system prompt."""

requests = ["paths.get"]

from guest.bases import BaseTool


class Advisor(BaseTool):
    """Advise."""

    name = "advisor"
    description = "x"

    def agent_prompt(self, sdk):
        """Say where something lives — the reason this is dynamic at all."""
        return f"## Scripts\\nThey go in {sdk.paths.get('scripts')}."

    def run(self, sdk):
        """Do nothing in particular."""
        return "ok"
'''


def test_a_prompt_contribution_is_bridged(tmp_path, box):
    """The guidance a migrated plugin writes has to reach the system prompt.

    ``agent/system_prompt._collect`` calls this on the adapter. Unforwarded, the
    native base answered with its empty static ``agent_prompt`` — so every
    migrated plugin's point-of-use guidance disappeared while the plugin went on
    working, and the only symptom was an agent that no longer knew things. The
    text is dynamic on purpose here: a static literal crosses as a declaration
    and would pass even with nothing bridged.
    """
    module = adapt(_write(tmp_path, "tool_advisor.py", PROMPTING_TOOL))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))
    try:
        text = instance.agent_prompt(SimpleNamespace(config={}, scope=None))
        assert text.startswith("## Scripts")
        assert "scripts" in text
        # Computed once: ``_collect`` runs every turn, and for an ephemeral
        # family every call is a fresh box.
        assert instance._prompt_text == text
    finally:
        unload_box("tool_advisor")


def test_a_plugin_that_contributes_nothing_keeps_the_base_default(tmp_path,
                                                                  box):
    """Only forward what the file defines — same rule as ``form``.

    ``agent_prompt`` is one name with two shapes: a method when the text is
    dynamic, a plain string otherwise. A file defining neither must be left
    holding the base's empty *string* and no forwarding method at all —
    otherwise every prompt collection would pay a box spawn to be told
    nothing. So the assertion is on the shape, then on what the real collector
    makes of it.
    """
    from agent.system_prompt import _collect

    module = adapt(_write(tmp_path, "tool_word_count.py", MIGRATED_TOOL))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    assert instance.agent_prompt == ""
    assert not callable(instance.agent_prompt)
    assert _collect([instance], SimpleNamespace(config={})) == ""


def test_the_old_prompt_spelling_contributes_nothing(tmp_path, box):
    """``agent_prompt_for`` was the old name for the dynamic half, and is gone.

    It was carried for as long as the store had plugins still writing it,
    because dropping guidance silently is the exact failure this doorway
    exists to prevent — caught in practice once, when two installed tools went
    from 3.6kB of prompt text to nothing with every test passing.

    Pinned as a *negative* rather than deleted, because the failure it guards
    against is silence either way: nothing about a plugin whose prompt never
    arrives looks wrong, so the rename has to be a thing the suite states.
    """
    from agent.system_prompt import _collect

    source = PROMPTING_TOOL.replace("def agent_prompt(self, sdk)",
                                    "def agent_prompt_for(self, sdk)")
    module = adapt(_write(tmp_path, "tool_advisor.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))
    try:
        assert _collect([instance], SimpleNamespace(config={}, scope=None)) == ""
    finally:
        unload_box("tool_advisor")


def test_a_static_declaration_contributes_without_entering_the_box(tmp_path,
                                                                   box):
    """The cheap half of the same doorway.

    A literal crosses as an ordinary declaration and is copied onto the
    adapter, so the collector reads it straight off the attribute. Nothing
    opens a box — which is the whole reason the static spelling still exists
    now that both spellings share one name.
    """
    from agent.system_prompt import _collect

    source = MIGRATED_TOOL.replace(
        "    max_calls = 5",
        '    max_calls = 5\n    agent_prompt = "## Words\\nCount them."')
    module = adapt(_write(tmp_path, "tool_word_count.py", source))
    instance = next(v() for v in vars(module).values() if isinstance(v, type))

    assert instance.agent_prompt == "## Words\nCount them."
    assert _collect([instance], SimpleNamespace(config={})) == "## Words\nCount them."
    # No residency was opened, and no per-instance cache was written: the
    # forwarding path was never entered.
    assert getattr(instance, "_prompt_text", None) is None


@pytest.mark.parametrize("filename, source, base_module, base_name", [
    ("tool_word_count.py", MIGRATED_TOOL, "plugins.native.tool", "BaseTool"),
    ("service_counter.py", MIGRATED_SERVICE.replace("ISOLATION", ""),
     "plugins.native.service", "BaseService"),
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


# ────────────────────────────────────────────────────────────────────
# run_event is bridged (was test_sandbox_event_tasks.py)
# ────────────────────────────────────────────────────────────────────

from sandbox.bridge import adapt, configure
from sandbox.validator import validate_file


_EVENT_TASK = '''\
"""A task that reacts to a channel."""

from guest.bases import BaseTask

requests = []


class SweepTask(BaseTask):
    """Sweep."""

    name = "sweep"
    description = "x"
    trigger = "event"
    trigger_channels = ["sweep_now"]

    def run_event(self, sdk, payload):
        """Answer with what arrived."""
        return sdk.ok({"seen": payload.get("mark")})
'''

_PATH_TASK = '''\
"""A task that reacts to files."""

from guest.bases import BaseTask

requests = []


class IndexTask(BaseTask):
    """Index."""

    name = "index"
    description = "x"

    def run(self, sdk, paths):
        """Answer with what arrived."""
        return sdk.ok({"count": len(paths)})
'''


@pytest.fixture(autouse=True)
def _sandbox():
    configure(Sandbox())


def _adapted(tmp_path, source, stem):
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    path = tasks / f"task_{stem}.py"
    path.write_text(source, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the file should have adapted as a task"
    return module


def _instance(module):
    name = next(n for n in dir(module) if n.startswith("Sandboxed"))
    return getattr(module, name)()


def test_an_event_task_reaches_its_guest(tmp_path):
    """The orchestrator's call signature, forwarded to the guest's."""
    task = _instance(_adapted(tmp_path, _EVENT_TASK, "sweep"))

    result = task.run_event("run-1", {"mark": 7}, SimpleNamespace())

    assert result.success
    assert result.data == {"seen": 7}


def test_an_event_task_answers_with_a_task_result(tmp_path):
    """``run_event`` is an entry point, so it gets the family's translation.

    Handing the orchestrator raw data instead would make a failed sweep
    indistinguishable from a successful one.
    """
    task = _instance(_adapted(tmp_path, _EVENT_TASK, "sweep"))
    result = task.run_event("run-1", {}, SimpleNamespace())

    assert hasattr(result, "success") and hasattr(result, "error")


def test_a_path_task_does_not_grow_the_doorway(tmp_path):
    """Carried only when the guest defines one, like ``form`` on a command.

    An adapter advertising ``run_event`` it cannot fulfil would answer the
    orchestrator by forwarding into nothing.
    """
    module = _adapted(tmp_path, _PATH_TASK, "index")
    task = _instance(module)

    assert "run_event" not in vars(type(task))
    # A path task answers with one outcome per path; the batch's data rides
    # on the first. See ``test_a_batch_task_answers_once_per_path``.
    assert task.run(["a", "b"], SimpleNamespace())[0].data == {"count": 2}


def test_the_channel_declaration_survives_being_read(tmp_path):
    """Declarations are AST-read, so a *name* reads as nothing at all.

    ``trigger_channels = [CHANNEL]`` is the natural way to write it and used to
    produce a task subscribed to no channel: it validated, loaded, registered,
    and never fired. The validator now refuses it at authoring time.
    """
    task = _instance(_adapted(tmp_path, _EVENT_TASK, "sweep"))
    assert task.trigger_channels == ["sweep_now"]


def test_a_channel_named_by_reference_is_refused(tmp_path):
    """The failure this rule exists for, pinned as a refusal."""
    source = _EVENT_TASK.replace(
        'trigger_channels = ["sweep_now"]',
        'CHANNEL = "sweep_now"\n    trigger_channels = [CHANNEL]')
    path = tmp_path / "task_indirect.py"
    path.write_text(source, encoding="utf-8")

    report = validate_file(path)

    assert not report.ok
    assert "trigger_channels" in report.render()


import time
from sandbox import Sandbox, provenance
from sandbox.console import Console
from sandbox.guest.requests import Request, Result
from sandbox.interpreter import Execution, Interpreter
from sandbox.policy import SAFE, UNSAFE, Chain, Decision, classify

# ──────────────────────────────────────────────────────────────────────
# A resident service must have something to answer Requests from.
# ──────────────────────────────────────────────────────────────────────

SERVICE = '''
"""A service that persists a setting it owns."""

from guest.bases import BaseService


class Keeper(BaseService):
    """Reads and writes its own config."""

    name = "keeper"
    exports = ["remember", "recall"]
    requests = ["config.read", "config.write"]

    def start(self, sdk):
        """Nothing to open."""
        return True

    def remember(self, sdk, value):
        """Persist through the service-owned setting."""
        sdk.config.write("keeper_note", value, scope="plugin")
        return True

    def recall(self, sdk):
        """Read it back."""
        return sdk.config.read("keeper_note")
'''


def _keeper(tmp_path, sandbox_):
    """Build and load the migrated service the way discovery would."""
    path = tmp_path / "service_keeper.py"
    path.write_text(SERVICE, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the service did not bridge"
    return module.build_services({})["keeper"]


def test_a_resident_service_can_reach_config(tmp_path, box):
    """A service is loaded before any session exists, so nothing handed it a
    context — and a handler with no context answers from nothing.

    ``config.read`` is the probe because it is classified SAFE and therefore
    actually reaches a handler. It is also where the damage was worst: it
    returned None for every key, which is indistinguishable from unset, so the
    timekeeper read back an empty job list and carried on.
    """
    store = {"keeper_note": "already on disk"}
    box.bind_context(lambda session_key=None: SimpleNamespace(
        config=store, db=None, services={}, runtime=None, user_id=1,
        session_key=session_key))

    service = _keeper(tmp_path, box)
    assert service.load() is True
    try:
        assert service.recall() == "already on disk"
    finally:
        service.unload()


def test_without_a_context_a_service_reads_nothing(tmp_path, box):
    """The regression, stated as the bug: no context, no answer.

    Note what this does *not* do — raise. That is the whole reason it survived
    a green suite for so long: an unwired service looks exactly like a
    correctly wired one whose setting happens to be unset.
    """
    service = _keeper(tmp_path, box)
    assert service.load() is True
    try:
        assert service.recall() is None
    finally:
        service.unload()


def test_a_service_write_outside_its_own_settings_is_refused(tmp_path, box):
    """Reaching the handler is not the same as being allowed to.

    A plugin persisting a setting the registry says it owns is safe; anything
    else is a config change and asks. This service is synthetic and owns
    nothing, so it is refused at the gate — which is the correct answer and
    confirms the context did not quietly widen anything.
    """
    from sandbox.bridge import ServiceCallFailed

    box.bind_context(lambda session_key=None: SimpleNamespace(
        config={}, db=None, services={}, runtime=None, user_id=1,
        session_key=session_key))
    service = _keeper(tmp_path, box)
    assert service.load() is True
    try:
        with pytest.raises(ServiceCallFailed) as caught:
            service.remember("not mine to write")
        assert "denied" in str(caught.value)
    finally:
        service.unload()
