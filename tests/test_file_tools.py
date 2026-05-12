"""Regression tests for file tools."""

from pathlib import Path
from types import SimpleNamespace

from plugins.tools.tool_edit_file import EditFile, PLUGIN_EDIT_REMINDER, ROOT_WARNING
from plugins.tools.tool_read_file import ReadFile
from plugins.tools.tool_run_command import RunCommand


def _ctx(approve=None):
    """Internal helper to handle ctx."""
    return SimpleNamespace(approve_command=approve, services={}, config={})


def test_edit_file_crud_and_read_modes():
    """Verify edit file crud and read modes."""
    path = Path(".codex_file_tools_test.txt")
    tool = EditFile()
    ctx = _ctx(lambda *_: True)
    try:
        assert tool.run(ctx, operation="create", path=str(path), content="hello\nworld\n", justification="test create").success
        assert "1: hello" in ReadFile().run(_ctx(), path=str(path)).llm_summary
        assert ReadFile().run(_ctx(), path=str(path), line_numbers=False).llm_summary == "hello\nworld"
        assert tool.run(ctx, operation="replace", path=str(path), old_text="world", new_text="brain", justification="test replace").success
        assert tool.run(ctx, operation="append", path=str(path), content="!", justification="test append").success
        assert path.read_text(encoding="utf-8") == "hello\nbrain\n!"
        assert tool.run(ctx, operation="delete", path=str(path), justification="test delete").success
        assert not path.exists()
    finally:
        path.unlink(missing_ok=True)


def test_edit_file_requires_approval_and_keeps_dialog_concise():
    """Verify edit file requires approval and keeps dialog concise."""
    path = Path(".codex_file_tools_test.txt")
    approvals = []
    try:
        result = EditFile().run(
            _ctx(lambda c, j: approvals.append((c, j)) or False),
            operation="create",
            path=str(path),
            content="secret full file text",
            justification="create a smoke-test file",
        )
        assert not result.success
        assert not path.exists()
        assert approvals and approvals[0][0].startswith("edit_file create ")
        assert "create a smoke-test file" in approvals[0][1]
        assert ROOT_WARNING in approvals[0][1]
        assert "secret full file text" not in approvals[0][1]
    finally:
        path.unlink(missing_ok=True)


def test_edit_file_data_dir_edit_has_no_root_warning(monkeypatch):
    """Verify edit file data dir edit has no root warning."""
    import plugins.tools.tool_edit_file as edit_mod

    data_dir = Path(".codex_file_tools_data").resolve()
    path = data_dir / "sandbox_tools" / "tool_demo.py"
    approvals = []
    try:
        monkeypatch.setattr(edit_mod, "DATA_DIR", data_dir)
        monkeypatch.setattr(edit_mod, "ROOTS", tuple(p.resolve() for p in (edit_mod.ROOT_DIR, data_dir)))
        result = EditFile().run(
            _ctx(lambda c, j: approvals.append((c, j)) or False),
            operation="create",
            path=str(path),
            content="x",
            justification="create a sandbox tool",
        )

        assert not result.success
        assert approvals
        assert ROOT_WARNING not in approvals[0][1]
    finally:
        path.unlink(missing_ok=True)


def test_edit_file_plugin_edit_reminds_to_test(monkeypatch):
    """Verify edit file plugin edit reminds to test."""
    import plugins.tools.tool_edit_file as edit_mod

    plugin_dir = Path(".codex_file_tools_plugins").resolve()
    path = plugin_dir / "tool_demo.py"
    try:
        plugin_dir.mkdir(exist_ok=True)
        monkeypatch.setattr(edit_mod, "iter_plugin_dirs", lambda: [("tool", plugin_dir)])

        result = EditFile().run(
            _ctx(lambda *_: True),
            operation="create",
            path=str(path),
            content="x",
            justification="create a sandbox tool",
        )

        assert result.success
        assert PLUGIN_EDIT_REMINDER.strip() in result.llm_summary
    finally:
        path.unlink(missing_ok=True)
        plugin_dir.rmdir()


def test_edit_file_non_plugin_edit_has_no_plugin_reminder(monkeypatch):
    """Verify edit file non plugin edit has no plugin reminder."""
    import plugins.tools.tool_edit_file as edit_mod

    path = Path(".codex_file_tools_test.txt")
    try:
        monkeypatch.setattr(edit_mod, "iter_plugin_dirs", lambda: [])

        result = EditFile().run(
            _ctx(lambda *_: True),
            operation="create",
            path=str(path),
            content="x",
            justification="create a normal file",
        )

        assert result.success
        assert PLUGIN_EDIT_REMINDER.strip() not in result.llm_summary
    finally:
        path.unlink(missing_ok=True)


def test_run_command_allows_arbitrary_command_after_approval():
    """Verify run command allows arbitrary command after approval."""
    approvals = []
    result = RunCommand().run(_ctx(lambda c, j: approvals.append((c, j)) or True), command="echo SB_OK", justification="smoke test")

    assert result.success
    assert approvals and approvals[0][0] == "echo SB_OK"
    assert "SB_OK" in result.llm_summary


def test_run_command_denied_approval_does_not_run():
    """Verify run command denied approval does not run."""
    result = RunCommand().run(_ctx(lambda *_: False), command="echo NOPE", justification="smoke test")

    assert not result.success
    assert "denied" in result.error.lower()
