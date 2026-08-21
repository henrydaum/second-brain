"""The plugin contracts, boxes, and the requirement that none of it is
mandatory: the sandbox runs arbitrary code just as happily as a plugin."""

from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.bases import (COMMAND, FRONTEND, SERVICE, TASK, TOOL, BaseCommand,
                         BaseFrontend, BasePlugin, BaseService, BaseTask,
                         BaseTool, entry_for)
from guest.box import (EPHEMERAL, IN_PROCESS, PERSISTENT, SUBPROCESS,
                       Membership, resolve, same_box)
from guest.loader import load_entry, unload_box
from sandbox import Interpreter, run_in_process
from sandbox.runner_subprocess import run_in_subprocess
from sandbox.validator import validate_file

FIXTURES = Path(__file__).parent / "fixtures"
TOOL_FIXTURE = FIXTURES / "tool_wordcount.py"
SCRIPT_FIXTURE = FIXTURES / "scratch_script.py"


@pytest.fixture
def interp():
    """An interpreter that refuses everything unsafe."""
    it = Interpreter()
    yield it
    it.shutdown()


@pytest.fixture(autouse=True)
def clean_boxes():
    """Boxes are module caches; leaking one across tests hides staleness."""
    yield
    for name in ("wordcount", "scratch_script", "solo"):
        unload_box(name)


# ──────────────────────────────────────────────────────────────────────
# One ancestor.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cls,family", [
    (BaseTool, TOOL), (BaseTask, TASK), (BaseService, SERVICE),
    (BaseCommand, COMMAND), (BaseFrontend, FRONTEND),
])
def test_every_family_descends_from_baseplugin(cls, family):
    """Five families, one contract."""
    assert issubclass(cls, BasePlugin)
    assert cls.family == family


@pytest.mark.parametrize("attr", [
    "name", "description", "dependencies_files", "dependencies_pip",
    "requires_services", "config_settings", "agent_prompt",
    "agent_prompt_refresh", "requests",
    "box", "lifetime", "timeout", "memory_mb",
])
def test_shared_declarations_live_on_the_ancestor(attr):
    """Declared once, not five times."""
    assert hasattr(BasePlugin, attr)


def test_isolation_is_not_a_declaration_a_plugin_can_make():
    """The base class must not even offer the attribute.

    A guest class asserting its own containment is the vulnerability this
    replaced; leaving the slot on the ancestor would let one drift back in
    and read as authoritative.
    """
    assert not hasattr(BasePlugin, "isolation")


def test_family_is_not_the_authors_to_set():
    """A plugin cannot rename what kind of thing it is."""
    class MyTool(BaseTool):
        """A tool."""
        name = "mine"

    assert MyTool.family == TOOL


def test_services_and_frontends_are_persistent_by_default():
    """Both hold state across calls; that is what they are for."""
    assert BaseService.lifetime == PERSISTENT
    assert BaseFrontend.lifetime == PERSISTENT
    assert BaseTool.lifetime == ""      # ephemeral unless it says otherwise


def test_declared_reports_intent():
    """The kernel reads declarations; it does not obey them."""
    class Big(BaseTool):
        """Asks for a lot."""
        name = "big"
        timeout = 99999
        requests = ["fs.read", "net.http"]

    declared = Big.declared()
    assert declared["timeout"] == 99999      # asked
    assert declared["requests"] == ["fs.read", "net.http"]
    assert declared["family"] == TOOL


# ──────────────────────────────────────────────────────────────────────
# Boxes.
# ──────────────────────────────────────────────────────────────────────

def test_saying_nothing_gets_you_your_own_box():
    """Isolation is the default; grouping is a deliberate act."""
    boxes = resolve([Membership("a"), Membership("b")])
    assert set(boxes) == {"a", "b"}
    assert boxes["a"].isolation == IN_PROCESS
    assert boxes["a"].lifetime == EPHEMERAL


def test_declaring_a_box_groups_files():
    """A plugin and its helpers are one unit."""
    boxes = resolve([Membership("tool_x", box="g"),
                     Membership("helper_y", box="g"),
                     Membership("other")])
    assert set(boxes) == {"g", "other"}
    assert boxes["g"].members == ("helper_y", "tool_x")


def test_the_tightest_isolation_wins():
    """Joining a box can only narrow what the joiner may do."""
    boxes = resolve([Membership("a", box="g", isolation=IN_PROCESS),
                     Membership("b", box="g", isolation=SUBPROCESS)])
    assert boxes["g"].isolation == SUBPROCESS
    assert boxes["g"].isolated


def test_the_tightest_limits_win():
    """A careless member cannot loosen the box by moving into it."""
    boxes = resolve([Membership("a", box="g", timeout=300, memory_mb=1024),
                     Membership("b", box="g", timeout=30, memory_mb=256)])
    assert boxes["g"].timeout == 30
    assert boxes["g"].memory_mb == 256


def test_one_persistent_member_makes_the_box_persistent():
    """A box cannot be half torn down."""
    boxes = resolve([Membership("svc", box="g", lifetime=PERSISTENT),
                     Membership("tool", box="g")])
    assert boxes["g"].persistent


def test_unset_limits_do_not_win_over_set_ones():
    """0 means 'no opinion', not 'zero seconds'."""
    boxes = resolve([Membership("a", box="g", timeout=0),
                     Membership("b", box="g", timeout=45)])
    assert boxes["g"].timeout == 45


def test_the_import_rule_and_the_isolation_rule_are_one_rule():
    """Same box, importable. Different box, only a Request will do."""
    a = Membership("tool_x", box="g")
    b = Membership("helper_y", box="g")
    c = Membership("elsewhere")
    assert same_box(a, b)
    assert not same_box(a, c)


def test_a_plugin_declares_its_membership():
    """The class knows which box it wants — but not how isolated it is.

    ``box`` is intent the kernel honours. Isolation arrives from the host,
    resolved from the file's tree, so a class writing ``isolation`` changes
    nothing: the membership it builds carries whatever the host put there,
    which here is nothing.
    """
    class Boxed(BaseTool):
        """In a shared box."""
        name = "boxed"
        box = "shared"
        isolation = SUBPROCESS      # ignored: not the plugin's call

    m = Boxed.membership("tool_boxed")
    assert m.box_name == "shared"
    assert not resolve([m])["shared"].isolated

    # And the host's answer is what lands, whatever the class said.
    from dataclasses import replace
    assert resolve([replace(m, isolation=SUBPROCESS)])["shared"].isolated


# ──────────────────────────────────────────────────────────────────────
# Running the contracts for real.
# ──────────────────────────────────────────────────────────────────────

def test_a_plugin_class_runs_on_both_runners(interp, tmp_path):
    """entry_for resolves a class to its run method; both runners agree."""
    target = tmp_path / "doc.txt"
    target.write_text("one two three\nfour five", encoding="utf-8")

    entry = load_entry(TOOL_FIXTURE, "WordCount", box_name="wordcount")
    in_proc = run_in_process(interp, entry, name="word_count",
                             kwargs={"path": str(target)}, timeout=30)
    sub = run_in_subprocess(interp, str(TOOL_FIXTURE), "WordCount",
                            name="word_count", box="wordcount",
                            kwargs={"path": str(target)}, timeout=30)

    assert in_proc.ok and sub.ok, (in_proc.error, sub.error)
    assert in_proc.data == sub.data == 5


def test_box_members_can_import_each_other(interp, tmp_path):
    """The tool's answer comes from its helper, so the import resolved."""
    target = tmp_path / "doc.txt"
    target.write_text("a b c", encoding="utf-8")
    result = run_in_subprocess(interp, str(TOOL_FIXTURE), "WordCount",
                               name="word_count", box="wordcount",
                               kwargs={"path": str(target)}, timeout=30)
    assert result.ok, result.error
    assert result.data == 3


def test_a_helper_is_not_a_plugin():
    """No base class, no contract check - just a module in a box."""
    report = validate_file(FIXTURES / "helper_words.py")
    assert report.ok, report.render()


# ──────────────────────────────────────────────────────────────────────
# The requirement: none of this is mandatory.
# ──────────────────────────────────────────────────────────────────────

def test_an_arbitrary_script_runs_with_no_base_class(interp, tmp_path):
    """Scratch code an agent writes needs no contract at all."""
    target = tmp_path / "notes.txt"
    target.write_text("alpha beta\ngamma\n", encoding="utf-8")

    entry = load_entry(SCRIPT_FIXTURE, "summarize", box_name="scratch_script")
    in_proc = run_in_process(interp, entry, name="scratch",
                             kwargs={"path": str(target)}, timeout=30)
    sub = run_in_subprocess(interp, str(SCRIPT_FIXTURE), "summarize",
                            name="scratch", box="scratch_script",
                            kwargs={"path": str(target)}, timeout=30)

    assert in_proc.ok and sub.ok, (in_proc.error, sub.error)
    assert in_proc.data == sub.data
    assert in_proc.data["lines"] == 2
    assert in_proc.data["words"] == 3


def test_a_script_that_asks_for_nothing_still_works(interp):
    """Pure computation makes no Requests and needs no permission."""
    entry = load_entry(SCRIPT_FIXTURE, "pure_math", box_name="scratch_script")
    result = run_in_process(interp, entry, name="math",
                            kwargs={"values": [2, 4, 6]}, timeout=30)
    assert result.ok
    assert result.data == 4


def test_an_arbitrary_script_passes_the_validator():
    """A script is held to the effect rules, not to the plugin contract."""
    report = validate_file(SCRIPT_FIXTURE)
    assert report.ok, report.render()


def test_entry_for_passes_plain_functions_through():
    """Only classes need resolving."""
    def plain(sdk):
        """A function."""
        return None

    assert entry_for(plain) is plain


# ──────────────────────────────────────────────────────────────────────
# Box teardown.
# ──────────────────────────────────────────────────────────────────────

def test_unloading_a_box_clears_its_members(tmp_path):
    """An ephemeral box that kept its modules would serve stale code."""
    import sys
    load_entry(SCRIPT_FIXTURE, "pure_math", box_name="solo")
    assert any(k.startswith("box_solo") for k in sys.modules)
    unload_box("solo")
    assert not any(k.startswith("box_solo") for k in sys.modules)


# ────────────────────────────────────────────────────────────────────
# FormStep crosses as a plain dict (was test_sandbox_forms.py)
# ────────────────────────────────────────────────────────────────────

from sandbox import Sandbox
from sandbox.bridge import adapt
from guest.forms import FormStep


def test_guest_form_step_is_plain_mapping_data():
    step = FormStep(
        "action",
        "Choose.",
        enum=["load", "unload"],
        enum_labels=["Load it", "Unload it"],
        columns=2,
    )

    assert isinstance(step, dict)
    assert step.to_dict() == {
        "name": "action",
        "prompt": "Choose.",
        "required": True,
        "type": "string",
        "enum": ["load", "unload"],
        "enum_labels": ["Load it", "Unload it"],
        "default": None,
        "prompt_when_missing": False,
        "columns": 2,
    }
    assert "validator" not in step


def test_guest_form_step_crosses_subprocess_and_rehydrates(tmp_path):
    plugin = tmp_path / "command_form.py"
    plugin.write_text(
        "from guest.bases import BaseCommand\n"
        "from guest.forms import FormStep\n\n"
        "class FormCommand(BaseCommand):\n"
        "    name = 'form'\n"
        "    isolation = 'subprocess'\n"
        "    def form(self, sdk, args):\n"
        "        return [FormStep('value', 'Enter it.', False, "
        "type='integer', default=3)]\n"
        "    def run(self, sdk, args):\n"
        "        return args.get('value')\n",
        encoding="utf-8",
    )

    report = validate_file(plugin)
    assert report.ok, report.render()

    sandbox = Sandbox()
    try:
        result = sandbox.run(
            plugin, "FormCommand", kwargs={"args": {}}, method="form")
    finally:
        sandbox.shutdown()
    assert result.ok, result.error
    assert result.data == [FormStep(
        "value", "Enter it.", False, type="integer", default=3)]

    module = adapt(plugin)
    command_cls = next(
        value for value in vars(module).values()
        if isinstance(value, type) and getattr(value, "_sandboxed", False)
    )
    [native] = command_cls().form({}, None)
    assert native.to_dict() == FormStep(
        "value", "Enter it.", False, type="integer", default=3)


@pytest.mark.parametrize(
    ("family", "base"),
    [
        ("tool", "BaseTool"),
        ("task", "BaseTask"),
        ("service", "BaseService"),
        ("frontend", "BaseFrontend"),
    ],
)
def test_validator_rejects_form_steps_outside_commands(
        tmp_path, family, base):
    plugin = tmp_path / f"{family}_bad_form.py"
    plugin.write_text(
        f"from guest.bases import {base}\n"
        "from guest.forms import FormStep\n\n"
        f"class Bad({base}):\n"
        "    name = 'bad_form'\n"
        "    def run(self, sdk, **kwargs):\n"
        "        return FormStep('value')\n",
        encoding="utf-8",
    )

    report = validate_file(plugin)

    assert not report.ok
    assert "FormStep is command-only" in report.render()


# ──────────────────────────────────────────────────────────────────────
# The two administrative moments.
# ──────────────────────────────────────────────────────────────────────

_HOOKED = (
    "from guest.bases import BaseService\n\n\n"
    "class Probe(BaseService):\n"
    "    name = 'probe'\n"
    "    description = 'probe'\n"
    "    requests = ['paths.get']\n\n"
    "    def start(self, sdk):\n"
    "        return True\n\n"
    "    def on_install(self, sdk):\n"
    "        return 'set up ' + str(bool(sdk.paths.get('workspace')))\n"
)


def test_every_plugin_inherits_both_hooks_as_no_ops():
    """Declared on ``BasePlugin`` so they are documented in one place.

    A plugin with nothing to arrange writes nothing and the kernel calls
    nothing — the base versions exist for the author reading the contract, not
    for the package manager, which finds a hook by AST and skips a file that
    only inherits one.
    """
    for base in (BaseTool, BaseTask, BaseService, BaseCommand, BaseFrontend):
        assert base().on_install(None) is None
        assert base().on_uninstall(None) is None
    assert issubclass(BaseTool, BasePlugin)


def test_lifecycle_entries_finds_a_definition_and_ignores_an_inheritance():
    """Inheriting the no-op must not count, or every file costs a box.

    All five families inherit both, so "does this class have an ``on_install``"
    answers yes for every plugin ever written. The question has to be "does
    this class *define* one", which is the same distinction ``_prompt_method``
    draws for ``agent_prompt`` and for the same reason.
    """
    from sandbox.bridge import LIFECYCLE_METHODS, lifecycle_entries

    assert LIFECYCLE_METHODS == ("on_install", "on_uninstall")
    assert lifecycle_entries(_HOOKED, "on_install") == [("Probe", "probe")]
    assert lifecycle_entries(_HOOKED, "on_uninstall") == []
    assert lifecycle_entries("class Probe:\n    pass\n", "on_install") == []
    assert lifecycle_entries("def on_install(sdk):\n    return 1\n",
                             "on_install") == [("on_install", "")]
    assert lifecycle_entries("class Broken(:\n", "on_install") == []


def test_lifecycle_entries_answers_with_the_registered_name():
    """The declared ``name``, not the class and not the file stem.

    That string becomes the run's chain link, and the chain link is what
    ``policy._owns_setting`` matches against the setting registry — which knows
    ``probe``, never ``Probe`` or ``service_probe``. Getting it wrong makes a
    plugin a stranger to its own settings and costs a dialog it should not see.
    """
    from sandbox.bridge import lifecycle_entries

    [(entry, name)] = lifecycle_entries(_HOOKED, "on_install")
    assert entry == "Probe" and name == "probe"
    # No declaration is answered honestly rather than guessed at; the caller
    # falls back to the box's own name.
    assert lifecycle_entries(
        _HOOKED.replace("    name = 'probe'\n", ""), "on_install"
    ) == [("Probe", "")]


def test_once_runs_one_method_of_an_otherwise_resident_plugin(tmp_path):
    """A service is persistent, and ``on_install`` is still a single call.

    The refusal exists to stop a service being *run* like a tool — adapted into
    a one-shot that sets up a transport and is never called again. Naming one
    method and discarding the box is a different act, so the caller says so
    rather than the plugin declaring its way out of it.
    """
    from sandbox.facade import BoxError, Sandbox

    plugin = tmp_path / "service_probe.py"
    plugin.write_text(_HOOKED, encoding="utf-8")
    assert validate_file(plugin).ok

    sandbox = Sandbox()
    try:
        with pytest.raises(BoxError, match="persistent lifetime"):
            sandbox.run(plugin, "Probe", method="on_install")

        result = sandbox.run(plugin, "Probe", method="on_install", once=True)
    finally:
        sandbox.shutdown()

    assert result.ok, result.error
    assert result.data == "set up True"


def test_an_install_hook_is_asked_about_rather_than_refused():
    """The whole safety argument, in one classification.

    A hook runs under the chain of the ``/packages`` command the person typed:
    two links deep, so it inherits neither ``typed_command`` nor the install's
    ``approved`` grant and a config write is unsafe — but rooted at the user,
    so it is *attended* and the dialog can actually be drawn. Both halves
    matter. A service writing the same setting from its own chain roots at
    ``service:`` and is refused outright with nobody to ask, which is how two
    earlier attempts at install-time seeding failed silently.

    SQL is the other half and is deliberately free: dropping a table a plugin
    created is what ``on_uninstall`` is mostly for, and a dialog per DROP would
    teach people to stop reading them.
    """
    from guest.requests import Request
    from sandbox.policy import Chain, classify

    hook = Chain(root="user:command", links=("packages", "memory_retrieve"),
                 approved=frozenset({"plugin.install"}))
    assert not hook.typed_command and hook.attended

    write = Request("config.write",
                    {"key": "sync_directories", "value": ["/memory"]})
    assert not classify(write, hook).safe
    assert "sync_directories" in classify(write, hook).reason
    assert classify(Request("db.define",
                            {"ddl": "DROP TABLE IF EXISTS memory_usage"}),
                    hook).safe

    service = Chain(root="service:memory_retrieve")
    assert not classify(write, service).safe and not service.attended
