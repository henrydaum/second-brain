from pathlib import Path
from types import SimpleNamespace

from plugins.tools.tool_edit_file import EditFile, ROOT_WARNING
from plugins.tools.tool_read_file import ReadFile
from plugins.tools.tool_run_command import RunCommand


def _ctx(approve=None):
    return SimpleNamespace(approve_command=approve, services={}, config={})


def test_edit_file_crud_and_read_modes():
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


def test_run_command_allows_arbitrary_command_after_approval():
    approvals = []
    result = RunCommand().run(_ctx(lambda c, j: approvals.append((c, j)) or True), command="echo SB_OK", justification="smoke test")

    assert result.success
    assert approvals and approvals[0][0] == "echo SB_OK"
    assert "SB_OK" in result.llm_summary


def test_run_command_denied_approval_does_not_run():
    result = RunCommand().run(_ctx(lambda *_: False), command="echo NOPE", justification="smoke test")

    assert not result.success
    assert "denied" in result.error.lower()
