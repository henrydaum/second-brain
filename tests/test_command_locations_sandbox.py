"""SDK and parity coverage for the sandboxed ``/locations`` command."""

from pathlib import Path
from types import SimpleNamespace

from sandbox.facade import Sandbox
from sandbox.handlers.fs_net import _fs_list


def _run_locations(root, kind="root"):
    context = SimpleNamespace(root_dir=root)
    sandbox = Sandbox(context=context)
    try:
        return sandbox.run(
            "plugins/commands/command_locations.py",
            "LocationsCommand",
            kwargs={"args": {"kind": kind}},
        )
    finally:
        sandbox.shutdown()


def test_fs_list_details_preserves_default_and_reports_entry_types(tmp_path):
    (tmp_path / "folder").mkdir()
    (tmp_path / "file.txt").write_text("text", encoding="utf-8")

    plain = _fs_list(None, {"path": str(tmp_path)})
    detailed = _fs_list(None, {"path": str(tmp_path), "details": True})

    assert plain.ok
    assert plain.data == sorted([
        str(tmp_path / "file.txt"),
        str(tmp_path / "folder"),
    ])
    assert detailed.ok
    assert detailed.data == [
        {
            "path": str(tmp_path / "file.txt"),
            "name": "file.txt",
            "is_dir": False,
        },
        {
            "path": str(tmp_path / "folder"),
            "name": "folder",
            "is_dir": True,
        },
    ]


def test_locations_command_preserves_tree_format_and_order(tmp_path):
    (tmp_path / "Zoo").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "Beta.txt").write_text("text", encoding="utf-8")

    result = _run_locations(tmp_path)

    assert result.ok, result.error
    assert result.data.startswith(
        f"**Project root**\n`{tmp_path}`\n```\n"
        "alpha/\nZoo/\nBeta.txt\n```")
    assert "\n\n**Data directory**\n`" in result.data


def test_locations_command_marks_a_missing_plugins_directory(tmp_path):
    result = _run_locations(tmp_path, "plugins")

    assert result.ok, result.error
    assert (
        f"**Project root**\n`{tmp_path / 'plugins'}`\n```\n(missing)\n```"
        in result.data
    )
