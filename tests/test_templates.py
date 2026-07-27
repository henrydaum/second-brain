"""The templates teach the contract, so the contract checks the templates.

``agent/system_prompt_static.md`` tells the agent every turn that templates are
the source of truth for authoring each family. That makes them load-bearing
documentation: a template still teaching ``run(self, context)`` propagates the
old contract into every plugin written from it, and nothing would notice.

The examples inside them are therefore real, uncommented code, run through the
same validator a plugin faces. That is the whole point — commented-out examples
are what rotted last time, because nothing could check them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sandbox.bridge import imports_sdk
from sandbox.validator import ERROR, validate_file

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"

# Migrated to the SDK: these must validate and must not mention the old
# contract anywhere.
SANDBOXED = ["tool_template.py", "task_template.py", "command_template.py",
             "service_template.py", "script_template.py", "hook_template.py",
             "frontend_template.py"]

# Deliberately still native, each for a stated reason carried in a banner at
# the top of the file. Listed explicitly so that adding a template forces a
# decision about which group it belongs to, rather than silently defaulting
# into the unchecked one. Empty since frontends were bridged — kept because
# the next family to be added may well arrive before its contract does.
NATIVE = {}

# The validator rules a template is allowed to break, and only these. Both are
# rules about DISCOVERY — one class per file, and the family prefix in the
# filename — which cannot apply to files that are never discovered. Showing
# several variants side by side is worth more than obeying them here, and
# hook_template.py has to hold services because that is where hooks live.
# Every other finding is a real failure.
DISCOVERY_ONLY = (
    "plugin classes; a plugin file must declare exactly one",
    "discovery finds plugins by filename",
)


def _templates() -> list:
    """Every template on disk, so a new one cannot be added unnoticed."""
    return sorted(p.name for p in TEMPLATES.glob("*_template.py"))


def test_every_template_is_accounted_for():
    """A new template must be classified as sandboxed or deliberately native."""
    assert set(_templates()) == set(SANDBOXED) | set(NATIVE)


@pytest.mark.parametrize("filename", SANDBOXED)
def test_sandboxed_template_validates(filename):
    """The examples must be code that would actually load."""
    report = validate_file(TEMPLATES / filename)
    real = [f for f in report.of(ERROR)
            if not any(rule in f.message for rule in DISCOVERY_ONLY)]
    assert not real, f"{filename} would not load:\n" + "\n".join(
        f.render() for f in real)


@pytest.mark.parametrize("filename", SANDBOXED)
def test_sandboxed_template_uses_the_sdk(filename):
    """It must import guest.bases, not the native bases."""
    source = (TEMPLATES / filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    if filename != "script_template.py":       # a script needs no base class
        assert imports_sdk(tree), f"{filename} does not import the SDK contract"

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                ("plugins.", "runtime.", "state_machine.", "pipeline.")):
            pytest.fail(f"{filename} imports kernel module {node.module}")


@pytest.mark.parametrize("filename", SANDBOXED)
def test_sandboxed_template_drops_the_old_contract(filename):
    """None of the pre-sandbox vocabulary may survive, in code or in prose.

    Prose counts: the agent reads the docstring as instruction, so a stale
    sentence teaches the old contract just as effectively as stale code.
    """
    source = (TEMPLATES / filename).read_text(encoding="utf-8")
    for term in ("ToolResult", "TaskResult", "build_services", "context.db",
                 "context.services", "context.config", "context.call_tool",
                 "import logging"):
        assert term not in source, f"{filename} still mentions {term}"


@pytest.mark.parametrize("filename", sorted(NATIVE))
def test_native_template_says_so(filename):
    """A template still on the old contract must carry its banner.

    Without one, an agent reading it beside the migrated templates has no way
    to tell that this family works differently.
    """
    source = (TEMPLATES / filename).read_text(encoding="utf-8")
    assert "STILL THE NATIVE CONTRACT" in source


@pytest.mark.parametrize("filename", sorted(NATIVE))
def test_native_template_imports_nothing(filename):
    """Documentation must not be able to break the app by being imported."""
    tree = ast.parse((TEMPLATES / filename).read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import,
                                                           ast.ImportFrom))]
    assert not imports, f"{filename} executes imports at module level"
