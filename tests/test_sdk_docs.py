"""The SDK documentation, checked against the SDK.

``SDK.md`` is written to be handed to an agent as the whole context for
writing sandbox code. If an example in it does not run, the agent writes code
that does not run — so the examples are executed here rather than trusted.
"""

import re
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.validator import validate

DOC = Path(__file__).resolve().parents[1] / "SDK.md"


@pytest.fixture
def box():
    """A sandbox with nothing unsafe allowed."""
    made = Sandbox()
    yield made
    made.shutdown()


def _blocks(marker: str) -> list:
    """Fenced python blocks from the document."""
    text = DOC.read_text(encoding="utf-8")
    return [b for b in re.findall(r"```python\n(.*?)```", text, re.S)
            if marker in b]


def test_the_document_exists_and_is_python():
    """Every fenced python block must at least parse."""
    import ast

    blocks = re.findall(r"```python\n(.*?)```",
                        DOC.read_text(encoding="utf-8"), re.S)
    assert len(blocks) > 10
    for block in blocks:
        # Reference listings are call signatures, not runnable statements;
        # they still have to be syntactically valid Python.
        ast.parse(block)


# ──────────────────────────────────────────────────────────────────────
# The two worked examples must actually run.
# ──────────────────────────────────────────────────────────────────────

def test_the_script_example_runs(box, tmp_path):
    """The 'a script' example, executed as written."""
    source = next(b for b in _blocks("def summarize"))
    script = tmp_path / "summarize_doc.py"
    script.write_text(source, encoding="utf-8")

    target = tmp_path / "notes.txt"
    target.write_text("alpha beta\ngamma\n", encoding="utf-8")

    try:
        result = box.run(script, "summarize", kwargs={"path": str(target)})
    finally:
        unload_box("summarize_doc")

    assert result.ok, result.error
    assert result.data == {"lines": 2, "words": 3}


def test_the_plugin_example_runs_and_conforms(box, tmp_path):
    """The 'a plugin' example, validated and executed as written."""
    source = next(b for b in _blocks("class WordCount"))
    plugin = tmp_path / "tool_word_count.py"
    plugin.write_text(source, encoding="utf-8")

    report = validate(source, filename="tool_word_count.py")
    assert report.ok, report.render()

    target = tmp_path / "doc.txt"
    target.write_text("one two three four", encoding="utf-8")

    try:
        result = box.run(plugin, "WordCount", kwargs={"path": str(target)})
    finally:
        unload_box("tool_word_count")

    assert result.ok, result.error
    assert result.data == 4


def test_the_helper_and_box_example_runs(box, tmp_path):
    """Two files, one box, a relative import - exactly as documented."""
    helper = next(b for b in _blocks("def count_words"))
    plugin_bit = next(b for b in _blocks("from .helper_words"))

    (tmp_path / "helper_words.py").write_text(helper, encoding="utf-8")
    plugin = tmp_path / "tool_wordcount.py"
    plugin.write_text(
        '"""Count words."""\n\n'
        "from guest.bases import BaseTool\n\n"
        + plugin_bit.split("\n", 1)[1] + "\n\n"
        "class WordCount(BaseTool):\n"
        '    """Count."""\n'
        '    name = "wc_doc"\n'
        '    box = "wordcount"\n\n'
        "    def run(self, sdk, path):\n"
        '        """Count."""\n'
        "        return count_words(sdk.fs.read(path))\n",
        encoding="utf-8")

    target = tmp_path / "d.txt"
    target.write_text("a b c", encoding="utf-8")
    try:
        result = box.run(plugin, "WordCount", kwargs={"path": str(target)})
    finally:
        unload_box("wordcount")

    assert result.ok, result.error
    assert result.data == 3


# ──────────────────────────────────────────────────────────────────────
# The rejection table has to be true in both columns.
# ──────────────────────────────────────────────────────────────────────

REJECTED = [
    "open(p).read()",
    "import os",
    "import pathlib",
    "import subprocess",
    "import requests",
    "import logging",
    "import paths",
    "eval('1')",
]

ACCEPTED = [
    "sdk.fs.read(p)",
    "sdk.fs.write(p, s)",
    "sdk.proc.run(['ls'])",
    "sdk.net.http('https://x.test')",
    "sdk.db.query('select 1')",
    "sdk.log('hi')",
]


@pytest.mark.parametrize("snippet", REJECTED)
def test_the_left_column_really_is_rejected(snippet):
    """Everything the document says not to write must actually be caught."""
    source = (f"{snippet}\n\n\ndef go(sdk):\n    return 1\n"
              if snippet.startswith("import")
              else f"def go(sdk, p):\n    return {snippet}\n")
    report = validate(source, filename="scratch.py")
    assert not report.ok, f"{snippet} was allowed"


# Stdlib that opens a file the plugin names. SDK.md says these are disclaimed
# rather than refused, so "not an error" and "not silently fine" both matter.
DISCLAIMED = ["import sqlite3", "import zipfile", "import tarfile"]


@pytest.mark.parametrize("snippet", DISCLAIMED)
def test_unmediated_stdlib_loads_with_a_disclaimer(snippet):
    """The document promises a disclaimer, not a refusal and not silence."""
    source = f"{snippet}\n\n\ndef go(sdk):\n    return 1\n"
    report = validate(source, filename="scratch.py")
    assert report.ok, f"{snippet} was refused"
    assert report.disclaimed, f"{snippet} passed without a disclaimer"


@pytest.mark.parametrize("snippet", ACCEPTED)
def test_the_right_column_really_is_accepted(snippet):
    """And everything it recommends must pass."""
    report = validate(f"def go(sdk, p='', s=''):\n    return {snippet}\n",
                      filename="scratch.py")
    assert report.ok, f"{snippet}: {report.render()}"


def test_the_free_stdlib_list_is_accurate():
    """The document promises these need no Request."""
    promised = ("json", "re", "math", "datetime", "time", "collections",
                "itertools", "hashlib", "base64", "csv", "email", "textwrap",
                "statistics", "dataclasses", "typing", "croniter",
                "cron_descriptor")
    for module in promised:
        report = validate(f"import {module}\n\n\ndef go(sdk):\n    return 1\n",
                          filename="scratch.py")
        assert report.ok and not report.disclaimed, \
            f"{module} is not actually free: {report.render()}"


def test_every_documented_namespace_exists():
    """A namespace named in the reference but missing from the SDK would
    send an agent down a path that cannot work."""
    from sandbox.guest.sdk import SDK

    text = DOC.read_text(encoding="utf-8")
    named = set(re.findall(r"\bsdk\.([a-z_]+)\.", text))
    sdk = SDK(None)
    for namespace in named:
        assert hasattr(sdk, namespace), f"SDK.md documents missing sdk.{namespace}"


def test_every_documented_method_exists():
    """Same for the methods on them."""
    from sandbox.guest.sdk import SDK

    text = DOC.read_text(encoding="utf-8")
    sdk = SDK(None)
    for namespace, method in set(re.findall(r"\bsdk\.([a-z_]+)\.([a-z_]+)\(",
                                            text)):
        target = getattr(sdk, namespace, None)
        assert target is not None, f"no sdk.{namespace}"
        assert hasattr(target, method), \
            f"SDK.md documents missing sdk.{namespace}.{method}()"
