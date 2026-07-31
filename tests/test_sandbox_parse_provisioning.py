"""Parsers a plugin *declares* and the kernel loads into its box.

Image, audio, video and tabular parses produce live objects — a PIL image, a
numpy waveform, an open container, a DataFrame — and none of them can cross a
process boundary. For a long time that left sandboxed code with no route to a
heavy modality at all: ``sdk.parse.file`` refused them, and reaching the
kernel's registry directly is impossible from a box (the child runs with
``sandbox/`` as its cwd, so ``import parsing`` is a ``ModuleNotFoundError``).

The route is to move the parser rather than the result. A plugin declares
``parse_modalities``, the kernel resolves that against its live registry, and
the resolved *files* are imported into the plugin's own box before it runs.
``sdk.parse.file`` then calls one directly and the object never travels.

Two properties are the point, and both are tested here:

- **The kernel resolves, the box does not.** A modality is a fact about which
  parser packages are installed, which the box has no way to know. Naming a
  capability therefore cannot reach a file the kernel would not have offered.
- **Declaring tightens isolation.** The declaration is read before anything
  runs, so provisioning foreign code is visible to the isolation decision —
  unlike a relative import of a declared helper, which is not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox.guest import parsing as guest_parsing
from sandbox.guest.loader import install_parsers, unload_box
from sandbox.interpreter import Interpreter
from sandbox.isolation import IN_PROCESS, SUBPROCESS, required_isolation
from sandbox.runner_subprocess import run_in_subprocess
from sandbox.validator import validate_file
from tests.support import retarget_trees

# A parser whose output is deliberately unserializable, which is the whole
# case: if this crossed a wire the feature would not need to exist.
FAKE_PARSER = '''
"""A parser for a made-up format."""

from guest.parsing import ParseResult, register


class Canvas:
    """Stands in for a PIL image: live, and not JSON."""

    def __init__(self, size):
        self.size = size


def parse_fake_image(sdk, path, config=None):
    """Two 'images' out of one file."""
    return ParseResult(modality="image", output=[Canvas(4), Canvas(8)])


register([".fake"], "image", parse_fake_image)
'''

PROBE = '''
def run(sdk):
    """Parse locally and report what came back, without returning it."""
    images = sdk.parse.file("thing.fake", "image")
    return sdk.ok({"count": len(images),
                   "type": type(images[0]).__name__,
                   "size": images[0].size})
'''


@pytest.fixture(autouse=True)
def _clean_box_registry():
    """Each test gets an empty local table; the module-level one is global."""
    guest_parsing._LOCAL.clear()
    guest_parsing.drain_registrations()
    yield
    guest_parsing._LOCAL.clear()
    guest_parsing.drain_registrations()


def _write(directory: Path, name: str, source: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(source, encoding="utf-8")
    return path


# ──────────────────────────────────────────────────────────────────────
# The box side: an imported parser becomes a route.
# ──────────────────────────────────────────────────────────────────────

def test_installing_a_parser_gives_the_box_a_route(tmp_path):
    """register() fires on import; adopt_registrations collects it."""
    parser = _write(tmp_path / "parsers", "parse_fake.py", FAKE_PARSER)

    gained = install_parsers([str(parser)], box_name="probe_box")
    try:
        assert gained == 1
        assert guest_parsing.local_parser(".fake", "image") is not None
        assert guest_parsing.local_modalities() == ["image"]
    finally:
        unload_box("probe_box")


def test_a_route_is_keyed_by_extension_and_modality(tmp_path):
    """Same file, different modality, is a different route — as in the kernel."""
    parser = _write(tmp_path / "parsers", "parse_fake.py", FAKE_PARSER)

    install_parsers([str(parser)], box_name="probe_box")
    try:
        assert guest_parsing.local_parser(".fake", "tabular") is None
        assert guest_parsing.local_parser(".other", "image") is None
        # Extensions normalize the way the kernel's registry normalizes them.
        assert guest_parsing.local_parser("fake", "image") is not None
        assert guest_parsing.local_parser(".FAKE", "image") is not None
    finally:
        unload_box("probe_box")


def test_one_broken_parser_does_not_sink_the_others(tmp_path):
    """Discovery tolerates a bad parser; provisioning has to agree."""
    broken = _write(tmp_path / "parsers", "parse_broken.py",
                    "raise RuntimeError('this parser is broken')\n")
    good = _write(tmp_path / "parsers", "parse_fake.py", FAKE_PARSER)

    gained = install_parsers([str(broken), str(good)], box_name="probe_box")
    try:
        assert gained == 1
        assert guest_parsing.local_parser(".fake", "image") is not None
    finally:
        unload_box("probe_box")


def test_a_partial_declaration_is_not_inherited(tmp_path):
    """A parser that registers then raises must not leave its routes behind.

    ``_DECLARED`` is module-global, so anything a failed import left in it
    would be adopted by whichever parser loaded next — attributing one file's
    routes to another.
    """
    half = _write(tmp_path / "parsers", "parse_half.py", FAKE_PARSER.replace(
        'register([".fake"], "image", parse_fake_image)',
        'register([".half"], "image", parse_fake_image)\n'
        'raise RuntimeError("fell over after registering")'))
    good = _write(tmp_path / "parsers", "parse_fake.py", FAKE_PARSER)

    install_parsers([str(half), str(good)], box_name="probe_box")
    try:
        assert guest_parsing.local_parser(".half", "image") is None
        assert guest_parsing.local_parser(".fake", "image") is not None
    finally:
        unload_box("probe_box")


# ──────────────────────────────────────────────────────────────────────
# End to end, in a real child process.
# ──────────────────────────────────────────────────────────────────────

def test_a_live_object_is_usable_inside_the_box_that_parsed_it(tmp_path):
    """The point of the whole mechanism, proved across a process boundary."""
    parser = _write(tmp_path / "parsers", "parse_fake.py", FAKE_PARSER)
    probe = _write(tmp_path, "probe.py", PROBE)

    result = run_in_subprocess(Interpreter(), str(probe), "run", name="probe",
                               parsers=[str(parser)])

    assert result.ok, result.error
    assert result.data == {"count": 2, "type": "Canvas", "size": 4}


def test_an_undeclared_modality_is_refused_with_the_fix(tmp_path):
    """No provisioning means the Request path, which refuses — actionably.

    The old message told the caller its result could not cross and left it
    there. There is something to do about it now, so the message says what.
    """
    probe = _write(tmp_path, "probe.py", PROBE)

    result = run_in_subprocess(Interpreter(), str(probe), "run", name="probe")

    assert not result.ok
    assert "parse_modalities = ['image']" in result.error


def test_a_crossable_modality_still_goes_to_the_kernel(tmp_path):
    """Declaring nothing must not break the cheap path.

    Text crosses fine, so it stays a Request and the parser's dependencies
    stay out of the caller's box. Only an unprovisioned route falls through.
    """
    probe = _write(tmp_path, "probe.py", '''
def run(sdk):
    """No parser provisioned, so this must be a Request."""
    return sdk.ok(sdk.parse.modality(".png"))
''')

    result = run_in_subprocess(Interpreter(), str(probe), "run", name="probe")

    assert result.ok, result.error
    assert result.data == "image"


# ──────────────────────────────────────────────────────────────────────
# Declaring provisioning tightens isolation.
# ──────────────────────────────────────────────────────────────────────

PURE_TASK = '''
"""Pure stdlib as far as its own AST is concerned."""

from guest.bases import BaseTask


class Index(BaseTask):
    """Does nothing interesting."""

    name = "index"

    def run(self, sdk, paths):
        """Nothing."""
        return sdk.ok()
'''


def test_a_declared_modality_forces_a_subprocess(monkeypatch, tmp_path):
    """Provisioned parsers are foreign by construction.

    Without this the entry file's AST is the only evidence, and it is clean:
    an installed task declaring ``parse_modalities`` would resolve IN_PROCESS
    and then have PyMuPDF loaded into the kernel's own process.
    """
    roots = retarget_trees(monkeypatch, tmp_path)
    source = _write(roots["installed"] / "tasks", "task_index.py",
                    PURE_TASK.replace('name = "index"',
                                      'name = "index"\n    parse_modalities'
                                      ' = ["image"]'))

    report = validate_file(source)
    assert not report.unmediated, "the entry file itself imports nothing foreign"
    assert required_isolation(source, report) == SUBPROCESS


def _task_declaring(roots, declared: str, imports: str = "") -> Path:
    """An installed task whose own source imports nothing foreign."""
    return _write(
        roots["installed"] / "tasks", "task_index.py",
        PURE_TASK.replace(
            '"""Pure stdlib as far as its own AST is concerned."""',
            '"""Pure stdlib as far as its own AST is concerned."""\n\n'
            f"dependencies_files = [{declared!r}]\n{imports}"))


def test_an_imported_helper_carries_its_imports_into_the_decision(
        monkeypatch, tmp_path):
    """The same hole, reached the other way.

    ``from . import parse_heavy`` reads as an ordinary sibling, so the task's
    own AST shows nothing foreign while the helper behind it imports a C
    library. Before this, an installed task like that resolved IN_PROCESS and
    ran PyMuPDF inside the kernel.
    """
    roots = retarget_trees(monkeypatch, tmp_path)
    _write(roots["installed"] / "parsers", "parse_heavy.py",
           "def parse(sdk, path, config=None):\n"
           "    import fitz\n"
           "    return None\n")
    source = _task_declaring(roots, "parsers/parse_heavy.py",
                             "from . import parse_heavy")

    report = validate_file(source)
    assert not report.unmediated, "the entry file's own imports are clean"
    assert required_isolation(source, report) == SUBPROCESS


def test_a_declared_helper_nobody_imports_is_packaging_only(
        monkeypatch, tmp_path):
    """``dependencies_files`` does two jobs, and only one loads code.

    A tool declares the *service* it calls over the wire so the package
    manager installs both — but it never imports it, so the service's torch
    never enters this box. Counting that would subprocess most of the store
    for a packaging relationship.
    """
    roots = retarget_trees(monkeypatch, tmp_path)
    _write(roots["installed"] / "services", "service_heavy.py",
           "def start(sdk):\n    import torch\n    return True\n")
    source = _task_declaring(roots, "services/service_heavy.py")

    assert required_isolation(source, validate_file(source)) == IN_PROCESS


def test_the_chain_of_imports_is_followed(monkeypatch, tmp_path):
    """A helper may reach another declared helper of its own."""
    roots = retarget_trees(monkeypatch, tmp_path)
    _write(roots["installed"] / "tasks", "middle.py",
           "from . import deep\n")
    _write(roots["installed"] / "tasks", "deep.py",
           "def go():\n    import fitz\n    return None\n")
    source = _write(
        roots["installed"] / "tasks", "task_index.py",
        PURE_TASK.replace(
            '"""Pure stdlib as far as its own AST is concerned."""',
            '"""Pure stdlib as far as its own AST is concerned."""\n\n'
            "dependencies_files = ['tasks/middle.py', 'tasks/deep.py']\n"
            "from . import middle"))

    assert required_isolation(source, validate_file(source)) == SUBPROCESS


def test_a_pure_helper_leaves_the_plugin_in_process(monkeypatch, tmp_path):
    """Tightening only where it is earned.

    An installed plugin that is pure computation over the SDK runs in-process,
    and importing a stdlib-only helper must not cost it that.
    """
    roots = retarget_trees(monkeypatch, tmp_path)
    _write(roots["installed"] / "tasks", "shared_rows.py",
           "def page(rows):\n    return list(rows)\n")
    source = _task_declaring(roots, "tasks/shared_rows.py",
                             "from .shared_rows import page")

    assert required_isolation(source, validate_file(source)) == IN_PROCESS


def test_an_imported_helper_that_is_missing_fails_closed(monkeypatch,
                                                         tmp_path):
    """A helper the kernel cannot find is one it cannot vouch for."""
    roots = retarget_trees(monkeypatch, tmp_path)
    source = _task_declaring(roots, "parsers/not_there.py",
                             "from . import not_there")

    assert required_isolation(source, validate_file(source)) == SUBPROCESS


# ──────────────────────────────────────────────────────────────────────
# Host-side resolution.
# ──────────────────────────────────────────────────────────────────────

def test_the_kernel_answers_which_files_provide_a_modality(monkeypatch,
                                                           tmp_path):
    """sources_for is what turns a declaration into something a box can load."""
    import parsing

    roots = retarget_trees(monkeypatch, tmp_path)
    _write(roots["installed"] / "parsers", "parse_fake.py", FAKE_PARSER)
    parsing.discover()

    found = parsing.sources_for(["image"])
    assert [Path(p).name for p in found] == ["parse_fake.py"]
    # A modality nothing provides is silence, not an error: the plugin finds
    # out when it parses one and is told there is no route for that extension.
    assert parsing.sources_for(["video"]) == []
    parsing.discover()      # leave the real registry as we found it
