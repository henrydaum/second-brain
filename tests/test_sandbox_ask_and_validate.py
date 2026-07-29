"""Two Requests an authoring agent leans on: ``ui.ask`` and ``plugin.validate``.

They are tested together because they arrived together, for the same reason:
nothing sandboxed had exercised either path. ``ui.ask`` had been broken for
every question with options since it was written — ``choices=`` went straight
through to a ``request_input`` parameter that does not exist — and a green
suite said nothing, because no test asked a question.
"""

from types import SimpleNamespace

import pytest

from sandbox.handlers.kernel import _plugin_validate, _ui_ask

CLEAN = '''
"""A tool that conforms."""

from guest.bases import BaseTool


class Counter(BaseTool):
    name = "counter_probe"
    description = "Count the words in a file."
    parameters = {"type": "object", "properties": {}}
    requests = ["fs.read"]

    def run(self, sdk, path):
        return len(sdk.fs.read(path).split())
'''

BANNED_IMPORT = '''
"""A tool reaching past the boundary."""

import os

from guest.bases import BaseTool


class Sneaky(BaseTool):
    name = "sneaky_probe"
    description = "Reaches the filesystem directly."
    parameters = {"type": "object", "properties": {}}

    def run(self, sdk):
        return os.listdir(".")
'''

KERNEL_IMPORT = '''
"""A tool importing the kernel, which no subprocess box can resolve."""

from guest.bases import BaseTool
from runtime.context import SecondBrainContext


class Reacher(BaseTool):
    name = "reacher_probe"
    description = "Imports a kernel module."
    parameters = {"type": "object", "properties": {}}

    def run(self, sdk):
        return str(SecondBrainContext)
'''

FOREIGN_IMPORT = '''
"""A tool using a library the validator cannot see inside."""

dependencies_pip = ["numpy"]

import numpy

from guest.bases import BaseTool


class Cruncher(BaseTool):
    name = "cruncher_probe"
    description = "Uses a foreign library."
    parameters = {"type": "object", "properties": {}}

    def run(self, sdk):
        return int(numpy.zeros(3).sum())
'''


def _tool(tmp_path, source, stem="tool_probe"):
    """Write a source file where resolve_plugin_path will accept it."""
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    path = tools / f"{stem}.py"
    path.write_text(source, encoding="utf-8")
    return path


# ── plugin.validate ───────────────────────────────────────────────────

def test_a_conforming_file_validates_clean(tmp_path):
    result = _plugin_validate(None, {"path": str(_tool(tmp_path, CLEAN))})
    assert result.ok
    assert result.data["ok"] is True
    assert result.data["disclaimed"] is False
    assert result.data["findings"] == []
    assert result.data["unmediated"] == []


def test_a_banned_import_will_not_load_and_says_where(tmp_path):
    """The whole value of the tool: a line number and a fix, not 'it failed'."""
    result = _plugin_validate(None, {"path": str(_tool(tmp_path, BANNED_IMPORT))})
    assert result.data["ok"] is False
    errors = [f for f in result.data["findings"] if f["level"] == "error"]
    assert errors
    assert all(f["line"] > 0 for f in errors)
    assert any("os" in f["message"] for f in errors)
    assert any(f["fix"] for f in errors)


def test_a_kernel_import_will_not_load(tmp_path):
    """The case a subprocess box most needs caught: the child cannot see it."""
    result = _plugin_validate(None, {"path": str(_tool(tmp_path, KERNEL_IMPORT))})
    assert result.data["ok"] is False
    assert any("runtime" in f["message"] for f in result.data["findings"])


def test_a_foreign_library_loads_with_a_disclaimer_and_names_itself(tmp_path):
    """``unmediated`` is structured on purpose — isolation is computed from it."""
    result = _plugin_validate(None, {"path": str(_tool(tmp_path, FOREIGN_IMPORT))})
    assert result.data["ok"] is True
    assert result.data["disclaimed"] is True
    assert "numpy" in result.data["unmediated"]


def test_findings_carry_the_four_fields_a_reader_needs(tmp_path):
    result = _plugin_validate(None, {"path": str(_tool(tmp_path, BANNED_IMPORT))})
    for finding in result.data["findings"]:
        assert set(finding) == {"level", "line", "message", "fix"}
        assert finding["level"] in {"error", "warning", "note"}


def test_declarations_cross_the_wire_as_plain_data(tmp_path):
    """An AST literal_eval yields tuples and sets, which JSON does not carry."""
    import json

    result = _plugin_validate(None, {"path": str(_tool(tmp_path, CLEAN))})
    json.dumps(result.data)  # would raise on a tuple key or a set value
    assert result.data["declarations"]["name"] == "counter_probe"


def test_validating_a_registered_plugin_is_not_a_duplicate_of_itself(tmp_path):
    """The most common call: re-check the file you just edited.

    The duplicate-name check exists to stop a *new* plugin shadowing an
    existing one. Reporting a file as a duplicate of itself would fire on
    every edit and train the reader to ignore the check.
    """
    path = _tool(tmp_path, CLEAN)
    registered = SimpleNamespace(_source_path=str(path))
    ctx = SimpleNamespace(tool_registry=SimpleNamespace(
        tools={"counter_probe": registered}))

    assert _plugin_validate(ctx, {"path": str(path)}).data["ok"] is True


def test_a_genuinely_duplicate_name_is_still_reported(tmp_path):
    """The other half — the check must still do its job."""
    path = _tool(tmp_path, CLEAN)
    other = SimpleNamespace(_source_path=str(tmp_path / "tools" / "tool_other.py"))
    ctx = SimpleNamespace(tool_registry=SimpleNamespace(
        tools={"counter_probe": other}))

    result = _plugin_validate(ctx, {"path": str(path)})
    assert result.data["ok"] is False
    assert any("already registered" in f["message"]
               for f in result.data["findings"])


def test_a_missing_file_fails_by_name(tmp_path):
    result = _plugin_validate(None, {"path": str(tmp_path / "tools" / "tool_gone.py")})
    assert not result.ok
    assert "no such file" in result.error


def test_a_path_outside_the_plugin_roots_is_refused():
    """A linter is not a general-purpose file reader."""
    result = _plugin_validate(None, {"path": "C:/Windows/System32/drivers/etc/hosts"})
    assert not result.ok


def test_an_empty_path_is_refused():
    assert not _plugin_validate(None, {"path": ""}).ok


def test_validate_is_read_only_and_always_safe():
    """It changes nothing, so putting a dialog in the authoring loop would
    only teach the agent to stop checking its work."""
    from sandbox import policy
    from sandbox.guest import requests as R

    assert R.PLUGIN_VALIDATE in R.READ_ONLY
    assert R.PLUGIN_VALIDATE in policy.ALWAYS_SAFE
    assert policy.classify(
        R.Request(R.PLUGIN_VALIDATE, {"path": "x"}),
        policy.Chain(root="user").push("some_plugin")).safe


# ── ui.ask ────────────────────────────────────────────────────────────

class _Answer:
    """The state machine's request object, as far as the handler uses it."""

    def __init__(self, value=None, cancelled=False, answered=True):
        self.value = value
        self.metadata = {"cancelled": cancelled}
        self._answered = answered

    def wait(self, timeout=None):
        return self._answered


def _asker(answer):
    """A ctx whose request_user_input records how it was called."""
    calls = []

    def request_user_input(title, prompt, **kwargs):
        calls.append({"title": title, "prompt": prompt, **kwargs})
        return answer

    return SimpleNamespace(request_user_input=request_user_input), calls


def test_choices_reach_the_state_machine_as_enum():
    """The bug this test exists for.

    ``request_input`` takes ``enum``; the guest says ``choices``. Passing the
    guest's spelling straight through raised a TypeError that surfaced as
    'could not ask', so every multiple-choice question was unanswerable.
    """
    ctx, calls = _asker(_Answer(value="red"))
    result = _ui_ask(ctx, {"prompt": "Pick one", "choices": ["red", "blue"]})

    assert result.ok
    assert result.data == "red"
    assert calls[0]["enum"] == ["red", "blue"]
    assert "choices" not in calls[0]


def test_the_default_answer_type_is_a_form_step_type():
    """'text' matched no display branch, so questions rendered with no help."""
    ctx, calls = _asker(_Answer(value="hi"))
    _ui_ask(ctx, {"prompt": "Say something"})
    assert calls[0]["type"] == "string"


def test_required_and_default_are_forwarded():
    """Dropped on the floor before — an optional question could not be skipped."""
    ctx, calls = _asker(_Answer(value=7))
    _ui_ask(ctx, {"prompt": "How many?", "type": "integer",
                  "required": False, "default": 3})
    assert calls[0]["required"] is False
    assert calls[0]["default"] == 3


def test_the_prompt_carries_the_form_steps_assistance():
    """A sandboxed question should read like a native one."""
    ctx, calls = _asker(_Answer(value=1))
    _ui_ask(ctx, {"prompt": "How many?", "type": "integer"})
    assert "How many?" in calls[0]["prompt"]
    assert "whole number" in calls[0]["prompt"]


def test_a_cancelled_question_is_a_denial_not_a_failure():
    """Cancellation and breakage are different events, and callers treat them
    differently — ``except sdk.Denied`` is the whole point of the distinction."""
    ctx, _calls = _asker(_Answer(cancelled=True))
    result = _ui_ask(ctx, {"prompt": "Proceed?", "type": "boolean"})
    assert result.denied


def test_an_unanswered_question_fails_and_is_retryable():
    ctx, _calls = _asker(_Answer(answered=False))
    result = _ui_ask(ctx, {"prompt": "Still there?"})
    assert not result.ok
    assert not result.denied
    assert result.retryable


def test_ui_ask_needs_somebody_to_ask():
    assert not _ui_ask(SimpleNamespace(), {"prompt": "Anyone?"}).ok
