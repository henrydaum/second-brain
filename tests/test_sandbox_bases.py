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
    "requires_services", "config_settings", "agent_prompt", "requests",
    "box", "isolation", "lifetime", "timeout", "memory_mb",
])
def test_shared_declarations_live_on_the_ancestor(attr):
    """Declared once, not five times."""
    assert hasattr(BasePlugin, attr)


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
    """The class knows which box it wants."""
    class Boxed(BaseTool):
        """In a shared box."""
        name = "boxed"
        box = "shared"
        isolation = SUBPROCESS

    m = Boxed.membership("tool_boxed")
    assert m.box_name == "shared"
    assert resolve([m])["shared"].isolated


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
