"""The validation script: does it catch carelessness, and does it teach?"""

import textwrap

from sandbox.validator import (ERROR, NOTE, WARNING, validate, validate_file)

GOOD_TOOL = '''
"""A well-behaved tool."""

dependencies_files = []
dependencies_pip = []

import json
import re

from plugins.BaseTool import BaseTool


class ReadNotes(BaseTool):
    """Read a notes file."""
    name = "read_notes"
    max_calls = 3

    def run(self, sdk, path):
        """Read and shorten."""
        r = sdk.fs.read(path)
        if not r:
            return sdk.fail(r.error)
        return sdk.ok(sdk.text.truncate(r.data, 500))
'''


def _validate(source, filename="tool_read_notes.py", known_names=()):
    """Validate a dedented snippet."""
    return validate(textwrap.dedent(source), filename=filename,
                    known_names=known_names)


def _messages(report):
    """All finding text, joined for substring assertions."""
    return " | ".join(f.message + " " + f.fix for f in report.findings)


# ──────────────────────────────────────────────────────────────────────
# The happy path.
# ──────────────────────────────────────────────────────────────────────

def test_a_conforming_tool_passes_clean():
    """No findings at all - the baseline the error cases are measured against."""
    report = _validate(GOOD_TOOL)
    assert report.ok
    assert not report.findings
    assert report.render().endswith("conforms.")


def test_relative_imports_are_left_alone():
    """A sibling plugin file is validated on its own, not through this one."""
    report = _validate(GOOD_TOOL.replace(
        "import json", "from .helpers import shared"))
    assert report.ok


def test_the_fixture_plugin_conforms():
    """The plugin both runners already execute must pass its own linter."""
    report = validate_file("tests/fixtures/sandbox_plugin.py")
    assert report.ok, report.render()


def test_polling_is_only_available_to_resident_plugins():
    source = GOOD_TOOL.replace(
        "max_calls = 3",
        "max_calls = 3\n    poll_interval = 1.0",
    ).replace(
        "def run(self, sdk, path):",
        "def poll(self, sdk):\n"
        "        return False\n\n"
        "    def run(self, sdk, path):",
    )
    report = _validate(source)
    assert not report.ok
    assert "only valid for resident services and frontends" in _messages(report)


def test_enabled_resident_polling_requires_a_poll_method():
    source = GOOD_TOOL.replace(
        "from plugins.BaseTool import BaseTool",
        "from guest.bases import BaseService",
    ).replace(
        "class ReadNotes(BaseTool):",
        "class ReadNotes(BaseService):",
    ).replace(
        "max_calls = 3",
        "poll_interval = 1.0",
    )
    report = _validate(source, filename="service_read_notes.py")
    assert not report.ok
    assert "defines no poll(self, sdk)" in _messages(report)


# ──────────────────────────────────────────────────────────────────────
# Reaching for the environment.
# ──────────────────────────────────────────────────────────────────────

def test_open_is_refused_and_names_the_request():
    """The error message is the point, not the refusal."""
    report = _validate(GOOD_TOOL.replace(
        "r = sdk.fs.read(path)", "r = open(path).read()"))
    assert not report.ok
    assert "sdk.fs.read" in _messages(report)


def test_effect_modules_are_refused():
    """Direct environment access is an error, whichever door it uses."""
    for module, expected in (("os", "sdk.fs"), ("subprocess", "sdk.proc.run"),
                             ("socket", "sdk.net.http"),
                             ("shutil", "sdk.fs.move")):
        report = _validate(GOOD_TOOL.replace("import json", f"import {module}"))
        assert not report.ok, module
        assert expected in _messages(report), module


def test_urllib_parse_is_pure_but_urllib_request_is_not():
    """Precision matters: one is string munging, the other is egress."""
    assert _validate(GOOD_TOOL.replace(
        "import json", "import urllib.parse")).ok
    assert not _validate(GOOD_TOOL.replace(
        "import json", "import urllib.request")).ok


def test_effect_methods_are_caught():
    """Path is fine for building paths, not for touching them."""
    report = _validate(GOOD_TOOL.replace(
        "r = sdk.fs.read(path)", "r = path.read_text()"))
    assert not report.ok
    assert "sdk.fs.read" in _messages(report)


def test_sdk_calls_are_never_mistaken_for_effects():
    """sdk.fs.read is the sanctioned path and must not trip the heuristic."""
    report = _validate(GOOD_TOOL.replace(
        "r = sdk.fs.read(path)", "r = sdk.fs.read_text(path)"))
    assert report.ok, report.render()


def test_eval_and_exec_are_refused():
    """Dynamic execution defeats the check entirely."""
    for call in ("eval('1')", "exec('x=1')", "__import__('os')"):
        report = _validate(GOOD_TOOL.replace("r = sdk.fs.read(path)",
                                             f"r = {call}"))
        assert not report.ok, call


# ──────────────────────────────────────────────────────────────────────
# Foreign libraries: run with a disclaimer, per the contract.
# ──────────────────────────────────────────────────────────────────────

def test_a_foreign_library_warns_rather_than_refusing():
    """It cannot be validated, so it is disclaimed - not blocked."""
    report = _validate(GOOD_TOOL.replace("import json", "import numpy"))
    assert report.ok
    assert report.disclaimed
    assert "subprocess" in _messages(report)
    assert len(report.of(WARNING)) == 1


# ──────────────────────────────────────────────────────────────────────
# The plugin contract.
# ──────────────────────────────────────────────────────────────────────

def test_missing_base_class_is_an_error():
    """A tool_*.py file must actually declare a tool."""
    report = _validate(GOOD_TOOL.replace("(BaseTool)", "(object)"))
    assert not report.ok
    assert "BaseTool" in _messages(report)


def test_two_plugin_classes_in_one_file_is_an_error():
    """One file, one plugin - discovery assumes it."""
    report = _validate(GOOD_TOOL + "\n\nclass Other(BaseTool):\n"
                                   '    """Second."""\n    name = "other"\n')
    assert not report.ok
    assert "exactly one" in _messages(report)


def test_a_missing_name_is_an_error():
    """Provenance chains are made of names."""
    report = _validate(GOOD_TOOL.replace('    name = "read_notes"\n', ""))
    assert not report.ok
    assert "name" in _messages(report)


def test_a_computed_name_is_an_error():
    """The name is read without importing, so it has to be a literal."""
    report = _validate(GOOD_TOOL.replace(
        'name = "read_notes"', 'name = "read_" + "notes"'))
    assert not report.ok
    assert "literal" in _messages(report)


def test_name_collisions_are_caught_at_authoring_time():
    """Caught while the agent can still fix it, not at load."""
    report = _validate(GOOD_TOOL, known_names={"read_notes", "grep"})
    assert not report.ok
    assert "already registered" in _messages(report)
    assert _validate(GOOD_TOOL, known_names={"grep"}).ok


def test_non_literal_dependencies_are_an_error():
    """The package manager reads these by AST, never by importing."""
    report = _validate(GOOD_TOOL.replace(
        "dependencies_pip = []", "dependencies_pip = []").replace(
        '    name = "read_notes"',
        '    name = "read_notes"\n    dependencies_pip = list(SOMETHING)'))
    assert not report.ok
    assert "literal list" in _messages(report)


def test_helper_files_skip_the_contract_check():
    """A parse_*.py helper is not a plugin and declares no class."""
    report = validate("import json\n\ndef parse(path):\n    return None\n",
                      filename="parse_text.py")
    assert report.ok


# ──────────────────────────────────────────────────────────────────────
# Declare freely, kernel clamps.
# ──────────────────────────────────────────────────────────────────────

def test_an_over_ceiling_declaration_is_advisory_not_fatal():
    """The plugin may ask; it just does not get it."""
    report = _validate(GOOD_TOOL.replace("max_calls = 3", "max_calls = 9999"))
    assert report.ok
    assert not report.disclaimed
    assert len(report.of(NOTE)) == 1
    assert "clamped" in _messages(report)


# ──────────────────────────────────────────────────────────────────────
# Mechanics.
# ──────────────────────────────────────────────────────────────────────

def test_syntax_errors_are_reported_not_raised():
    """A broken file is a finding, never an exception."""
    report = _validate("def broken(:\n")
    assert not report.ok
    assert "does not parse" in _messages(report)


def test_validating_never_imports_the_file(tmp_path):
    """Checking a file must not run it."""
    marker = tmp_path / "ran.txt"
    plugin = tmp_path / "tool_evil.py"
    plugin.write_text(
        f"open({str(marker)!r}, 'w').write('x')\n", encoding="utf-8")
    validate_file(plugin)
    assert not marker.exists()


def test_the_report_carries_the_bytes_it_checked(tmp_path):
    """Validate a path then re-open it, and the code that ran was never
    checked. The caller executes what the report carries."""
    plugin = tmp_path / "tool_x.py"
    plugin.write_text(GOOD_TOOL, encoding="utf-8")
    report = validate_file(plugin)
    plugin.write_text("import os\n", encoding="utf-8")
    assert "read_notes" in report.source
    assert "import os" not in report.source


def test_errors_sort_before_advisories():
    """The thing that blocks loading is read first."""
    report = _validate(GOOD_TOOL
                       .replace("max_calls = 3", "max_calls = 9999")
                       .replace("import json", "import os"))
    assert report.findings[0].level == ERROR
    rendered = report.render()
    assert rendered.index("Will not load") < rendered.index("Advisory")


# ──────────────────────────────────────────────────────────────────────
# Discovery is by filename, so the filename is part of the contract.
# ──────────────────────────────────────────────────────────────────────

def test_a_plugin_must_carry_its_family_prefix():
    """Discovery finds plugins by filename; a mismatch never loads."""
    report = _validate(GOOD_TOOL, filename="wordcount.py")
    assert not report.ok
    assert "tool_<name>.py" in _messages(report)


def test_the_prefix_must_match_the_declared_family():
    """A BaseTool in service_*.py is a tool nobody will ever find."""
    report = _validate(GOOD_TOOL, filename="service_read_notes.py")
    assert not report.ok
    assert "must be named tool_" in _messages(report)


def test_a_bare_family_name_is_not_enough():
    """tool.py declares a family and no plugin."""
    report = _validate(GOOD_TOOL, filename="tool_.py")
    assert not report.ok


def test_every_family_prefix_is_enforced():
    """All five, not just tools."""
    for family, base in (("task", "BaseTask"), ("service", "BaseService"),
                         ("command", "BaseCommand"),
                         ("frontend", "BaseFrontend")):
        source = GOOD_TOOL.replace("BaseTool", base)
        assert _validate(source, filename=f"{family}_x.py").ok, family
        assert not _validate(source, filename=f"wrong_{family}.py").ok, family


# ──────────────────────────────────────────────────────────────────────
# What counts as pure.
# ──────────────────────────────────────────────────────────────────────

def test_pure_stdlib_passes_clean():
    """Computation-only modules are not effects and must not be flagged."""
    for module in ("time", "email", "csv", "ast", "struct", "mimetypes",
                   "secrets", "traceback", "contextlib"):
        report = _validate(GOOD_TOOL.replace("import json", f"import {module}"))
        assert report.ok and not report.disclaimed, f"{module}: {report.render()}"


def test_vouched_third_party_packages_pass_clean():
    """Pure computation from pip is still pure computation."""
    for module in ("croniter", "cron_descriptor"):
        report = _validate(GOOD_TOOL.replace("import json", f"import {module}"))
        assert report.ok and not report.disclaimed, f"{module}: {report.render()}"


def test_logging_points_at_the_sdk_rather_than_being_allowed():
    """A subprocessed plugin's log lines must reach the kernel's sink, and
    a child process cannot see the host's logging configuration."""
    report = _validate(GOOD_TOOL.replace("import json", "import logging"))
    assert not report.ok
    assert "sdk.log" in _messages(report)


def test_io_is_pure_but_its_one_impure_name_is_not():
    """``BytesIO`` is how you hand bytes to a decoder without giving it a path.

    Banning the module punished that and taught nothing, so the module is pure
    and ``io.open`` is caught as an attribute instead — the same shape as
    reaching past a Database object into its connection.
    """
    assert _validate(GOOD_TOOL.replace("import json", "import io")).ok

    reaching = _validate(GOOD_TOOL.replace(
        "import json", "import io").replace(
        "r = sdk.fs.read(path)", "r = io.open(path).read()"))
    assert not reaching.ok
    assert "sdk.fs.read" in _messages(reaching)


def test_xml_stays_out_because_its_parser_takes_a_filename():
    """``ElementTree.parse`` opens a path, so the module is not pure."""
    xml = _validate(GOOD_TOOL.replace("import json",
                                      "import xml.etree.ElementTree"))
    assert xml.disclaimed


def test_unmediated_stdlib_is_disclaimed_rather_than_refused():
    """sqlite3, zipfile and tarfile open a file the plugin names.

    That cannot be mediated, but it is legitimate — a tabular parser reading a
    user's ``.db`` read-only, or a container parser opening an archive. They
    take the foreign-library disclaimer instead of being refused outright,
    because refusing them would lose the whole parser.
    """
    for module in ("sqlite3", "zipfile", "tarfile"):
        report = _validate(GOOD_TOOL.replace("import json", f"import {module}"))
        assert report.ok, module
        assert report.disclaimed, module
        assert "subprocess" in _messages(report), module


def test_reaching_around_the_kernel_database_is_still_an_error():
    """Allowing the sqlite3 import must not allow the misuse it guarded."""
    report = _validate(GOOD_TOOL.replace(
        "r = sdk.fs.read(path)", "r = context.db.conn.execute(path)"))
    assert not report.ok
    assert "sdk.db" in _messages(report)


def test_a_misspelled_request_declaration_is_an_error():
    """``requests`` is the grant an approval spends, so typos cost something.

    Unlike a bus channel, a plugin cannot invent a Request type, so this is a
    closed vocabulary and a name outside it grants nothing.
    """
    report = _validate(GOOD_TOOL.replace(
        "    max_calls = 3",
        '    max_calls = 3\n    requests = ["path.get", "totally.bogus"]'))
    assert not report.ok
    messages = _messages(report)
    assert "'path.get' is not a Request type" in messages
    assert "paths.get" in messages          # and it suggests the right one
    assert "'totally.bogus' is not a Request type" in messages


def test_a_correct_request_declaration_passes():
    report = _validate(GOOD_TOOL.replace(
        "    max_calls = 3",
        '    max_calls = 3\n    requests = ["fs.read", "proc.run"]'))
    assert report.ok, report.render()


def test_kernel_modules_are_not_called_foreign_libraries():
    """They are the boundary, not an unvalidatable dependency."""
    for module in ("paths", "runtime.context", "plugins.helpers.plugin_paths"):
        report = _validate(GOOD_TOOL.replace("import json", f"import {module}"))
        assert not report.ok, module
        assert "kernel side" in _messages(report), module
