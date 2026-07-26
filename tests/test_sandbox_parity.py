"""The parity harness, and the enlarged Result it compares.

The harness answers one narrow question — given the same arguments, does the
migrated plugin return the same thing? — by running the working tree against
the version git still has.
"""

import subprocess
from pathlib import Path

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Result, Sandbox
from sandbox.parity import compare, compare_many, previous_source

REPO = Path(__file__).resolve().parents[1]

NATIVE_TOOL = '''
"""The pre-migration version."""

from plugins.BaseTool import BaseTool, ToolResult


class WordCount(BaseTool):
    """Count words."""
    name = "word_count"

    def run(self, context, text=""):
        """Count and summarize."""
        words = len(text.split())
        return ToolResult(data={"words": words},
                          llm_summary=f"{words} words")
'''

MIGRATED_TOOL = '''
"""The migrated version."""

from guest.bases import BaseTool


class WordCount(BaseTool):
    """Count words."""
    name = "word_count"

    def run(self, sdk, text=""):
        """Count and summarize."""
        words = len(text.split())
        return sdk.ok({"words": words}, llm_summary=f"{words} words")
'''


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo standing in for the real one.

    The harness reads the previous version out of git, so a test of it needs
    a git history to read.
    """
    monkeypatch.setattr("sandbox.parity.REPO", tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path,
                   check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path,
                   check=True)
    return tmp_path


def _commit(repo: Path, relative: str, source: str, message="c"):
    """Write a file and commit it."""
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)
    return path


@pytest.fixture(autouse=True)
def clean_boxes():
    """Boxes are module caches; a leak between tests hides staleness."""
    yield
    for name in ("tool_word_count", "tool_word_count_previous",
                 "parity_tool_word_count", "tool_drifted",
                 "tool_drifted_previous", "parity_tool_drifted"):
        unload_box(name)


# ──────────────────────────────────────────────────────────────────────
# The enlarged Result.
# ──────────────────────────────────────────────────────────────────────

def test_result_carries_what_a_tool_produces():
    """llm_summary and attachment_paths had real engineering behind them."""
    result = Result(data=1, llm_summary="a summary",
                    attachment_paths=["/tmp/a.png"])
    assert result.llm_summary == "a summary"
    assert result.attachment_paths == ["/tmp/a.png"]


def test_result_carries_what_a_task_produces():
    """also_contains drives extraction; discovered_paths drives registration."""
    result = Result(also_contains=["inner.pdf"], discovered_paths=["/x/y.txt"])
    assert result.also_contains == ["inner.pdf"]
    assert result.discovered_paths == ["/x/y.txt"]


def test_the_new_fields_survive_the_wire():
    """A subprocess result that dropped them would lose behaviour silently."""
    original = Result(data={"a": 1}, llm_summary="s", attachment_paths=["p"],
                      also_contains=["c"], discovered_paths=["d"])
    rebuilt = Result.from_dict(original.to_dict())
    assert rebuilt == original


def test_sdk_ok_sets_them():
    """The author needs a way to produce them without building a Result."""
    from sandbox.guest.sdk import SDK

    sdk = SDK(None)
    result = sdk.ok({"x": 1}, llm_summary="short", attachments=["/a.png"])
    assert result.llm_summary == "short"
    assert result.attachment_paths == ["/a.png"]
    assert result.ok


def test_defaults_are_not_shared_between_results():
    """A mutable default on a frozen dataclass is still a shared-state bug."""
    first, second = Result(), Result()
    first.attachment_paths.append("x")
    assert second.attachment_paths == []


# ──────────────────────────────────────────────────────────────────────
# Reading the previous version out of git.
# ──────────────────────────────────────────────────────────────────────

def test_the_previous_version_comes_from_git(repo):
    """No duplicate files: the two versions are the tree and the history."""
    path = _commit(repo, "plugins/tools/tool_word_count.py", NATIVE_TOOL)
    path.write_text(MIGRATED_TOOL, encoding="utf-8")

    previous = previous_source(path)
    assert "ToolResult" in previous
    assert "sdk.ok" not in previous
    assert "sdk.ok" in path.read_text(encoding="utf-8")


def test_an_uncommitted_file_has_no_previous_version(repo):
    """Nothing to compare against is a clear answer, not a crash."""
    path = repo / "plugins" / "tools" / "tool_new.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MIGRATED_TOOL, encoding="utf-8")
    assert previous_source(path) is None


# ──────────────────────────────────────────────────────────────────────
# Comparing.
# ──────────────────────────────────────────────────────────────────────

def test_a_faithful_migration_matches(repo):
    """The case the harness exists to confirm."""
    path = _commit(repo, "tool_word_count.py", NATIVE_TOOL)
    path.write_text(MIGRATED_TOOL, encoding="utf-8")

    verdict = compare(path, "WordCount", payload={"text": "one two three"})
    assert verdict.matched, verdict.render()
    assert verdict.native["data"] == {"words": 3}
    assert verdict.sandboxed["data"] == {"words": 3}
    assert "identical" in verdict.render()


def test_llm_summary_is_compared(repo):
    """The field that would otherwise be lost silently."""
    drifted = MIGRATED_TOOL.replace('llm_summary=f"{words} words"',
                                    'llm_summary="different"')
    path = _commit(repo, "tool_word_count.py", NATIVE_TOOL)
    path.write_text(drifted, encoding="utf-8")

    verdict = compare(path, "WordCount", payload={"text": "one two"})
    assert not verdict.matched
    assert [d[0] for d in verdict.differences] == ["llm_summary"]
    assert "different" in verdict.render()


def test_a_changed_answer_is_caught(repo):
    """The whole point: same arguments, different answer."""
    drifted = MIGRATED_TOOL.replace("len(text.split())", "len(text)")
    path = _commit(repo, "tool_word_count.py", NATIVE_TOOL)
    path.write_text(drifted, encoding="utf-8")

    verdict = compare(path, "WordCount", payload={"text": "one two three"})
    assert not verdict.matched
    fields = [d[0] for d in verdict.differences]
    assert "data" in fields


def test_a_broken_migration_reports_rather_than_raises(repo):
    """A migration that crashes is a verdict, not an exception."""
    broken = MIGRATED_TOOL.replace("len(text.split())", "len(nope)")
    path = _commit(repo, "tool_word_count.py", NATIVE_TOOL)
    path.write_text(broken, encoding="utf-8")

    verdict = compare(path, "WordCount", payload={"text": "a b"})
    assert not verdict.matched
    assert verdict.error or verdict.differences


def test_a_missing_previous_version_is_reported(repo):
    """Comparing something never committed says so plainly."""
    path = repo / "tool_word_count.py"
    path.write_text(MIGRATED_TOOL, encoding="utf-8")
    verdict = compare(path, "WordCount", payload={"text": "x"})
    assert not verdict.matched
    assert "HEAD" in verdict.error


def test_absence_is_not_disagreement(repo):
    """Old result types do not carry every field the new one does.

    A ToolResult predating a field cannot disagree about it, so a field only
    one side reports is skipped rather than counted as drift.
    """
    thin_native = '''
"""No summary."""

from plugins.BaseTool import BaseTool, ToolResult


class WordCount(BaseTool):
    """Count words."""
    name = "word_count"

    def run(self, context, text=""):
        """Count."""
        return ToolResult(data={"words": len(text.split())})
'''
    thin_migrated = '''
"""No summary."""

from guest.bases import BaseTool


class WordCount(BaseTool):
    """Count words."""
    name = "word_count"

    def run(self, sdk, text=""):
        """Count."""
        return sdk.ok({"words": len(text.split())})
'''
    path = _commit(repo, "tool_word_count.py", thin_native)
    path.write_text(thin_migrated, encoding="utf-8")

    verdict = compare(path, "WordCount", payload={"text": "one two"})
    assert verdict.matched, verdict.render()


# ──────────────────────────────────────────────────────────────────────
# One context, two paths.
# ──────────────────────────────────────────────────────────────────────

def test_both_versions_see_the_same_context(repo, tmp_path):
    """A difference must mean the plugin differs, not the world does."""
    native = '''
"""Reads config the old way."""

from plugins.BaseTool import BaseTool, ToolResult


class ReadSetting(BaseTool):
    """Read a setting."""
    name = "read_setting"

    def run(self, context, key=""):
        """Straight off the context."""
        return ToolResult(data=context.config.get(key))
'''
    migrated = '''
"""Reads config through a Request."""

from guest.bases import BaseTool


class ReadSetting(BaseTool):
    """Read a setting."""
    name = "read_setting"

    def run(self, sdk, key=""):
        """Through the gate."""
        return sdk.config.read(key)
'''
    path = _commit(repo, "tool_read_setting.py", native)
    path.write_text(migrated, encoding="utf-8")

    context = type("Ctx", (), {"config": {"model": "opus"}})()
    verdict = compare(path, "ReadSetting", payload={"key": "model"},
                      context=context)
    unload_box("tool_read_setting")
    unload_box("parity_tool_read_setting")

    assert verdict.matched, verdict.render()
    assert verdict.sandboxed["data"] == "opus"


# ──────────────────────────────────────────────────────────────────────
# Running a suite.
# ──────────────────────────────────────────────────────────────────────

def test_many_comparisons_share_one_sandbox(repo):
    """A whole-suite run should cost one interpreter, not twenty."""
    path = _commit(repo, "tool_word_count.py", NATIVE_TOOL)
    path.write_text(MIGRATED_TOOL, encoding="utf-8")

    verdicts = compare_many([
        {"path": path, "entry": "WordCount", "payload": {"text": "a b"}},
        {"path": path, "entry": "WordCount", "payload": {"text": "a b c d"}},
    ])
    assert all(v.matched for v in verdicts), \
        "\n".join(v.render() for v in verdicts)
    assert verdicts[0].sandboxed["data"] == {"words": 2}
    assert verdicts[1].sandboxed["data"] == {"words": 4}


# ──────────────────────────────────────────────────────────────────────
# Planning a migration.
# ──────────────────────────────────────────────────────────────────────

def test_a_plan_names_the_request_for_each_effect(tmp_path):
    """The plan is the checklist; a line with no fix is not useful."""
    from sandbox.migrate import plan

    path = tmp_path / "tool_reader.py"
    path.write_text('''
"""Native."""

import pathlib

from plugins.BaseTool import BaseTool, ToolResult


class Reader(BaseTool):
    """Read a file."""
    name = "reader"

    def run(self, context, path=""):
        """Read it."""
        return ToolResult(data=pathlib.Path(path).read_text())
''', encoding="utf-8")

    found = plan(path)
    assert found.family == "tool"
    assert found.entry == "Reader"
    assert "sdk.fs.read" in found.requests
    rendered = found.render()
    assert "run(self, sdk, **kwargs)" in rendered
    assert "read_text" in rendered


def test_a_plan_sees_effects_reached_through_context(tmp_path):
    """getattr(context, 'db') is the common idiom and must not be invisible."""
    from sandbox.migrate import plan

    path = tmp_path / "tool_notes.py"
    path.write_text('''
"""Native."""

from plugins.BaseTool import BaseTool, ToolResult


class Notes(BaseTool):
    """Store notes."""
    name = "notes"

    def run(self, context, text=""):
        """Write a row."""
        db = getattr(context, "db", None)
        db.conn.execute("INSERT INTO notes VALUES (?)", (text,))
        return ToolResult(data=True)
''', encoding="utf-8")

    rendered = plan(path).render()
    assert "context.db" in rendered
    assert "sdk.db" in rendered
    assert ".conn" in rendered


def test_an_already_migrated_plugin_says_so(tmp_path):
    """Planning is for work not yet done."""
    from sandbox.migrate import plan

    path = tmp_path / "tool_done.py"
    path.write_text('''
"""Migrated."""

from guest.bases import BaseTool


class Done(BaseTool):
    """Nothing to do."""
    name = "done"

    def run(self, sdk):
        """Return."""
        return sdk.ok(1)
''', encoding="utf-8")

    found = plan(path)
    assert found.already_migrated
    assert "already written against the SDK" in found.render()


def test_plan_tree_orders_the_easy_ones_first(tmp_path):
    """The first migrations should prove the path, not test it."""
    from sandbox.migrate import plan_tree

    (tmp_path / "tool_easy.py").write_text('''
"""Easy."""

from plugins.BaseTool import BaseTool, ToolResult


class Easy(BaseTool):
    """Nothing to convert."""
    name = "easy"

    def run(self, context):
        """Return."""
        return ToolResult(data=1)
''', encoding="utf-8")

    (tmp_path / "tool_hard.py").write_text('''
"""Hard."""

import os
import subprocess

from plugins.BaseTool import BaseTool, ToolResult


class Hard(BaseTool):
    """Plenty to convert."""
    name = "hard"

    def run(self, context):
        """Do things."""
        return ToolResult(data=open(os.getcwd()).read())
''', encoding="utf-8")

    plans = plan_tree(tmp_path)
    assert [Path(p.path).stem for p in plans] == ["tool_easy", "tool_hard"]
