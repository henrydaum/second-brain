"""Tests for the package-store publishing utility."""

from __future__ import annotations

import json

import pytest

from scripts import package_publisher


def test_write_package_copies_file_and_dir_updates_index_and_validates(tmp_path):
    tool = tmp_path / "tool_echo.py"
    helper_dir = tmp_path / "helpers"
    helper = helper_dir / "echo_format.py"
    tool.write_text("print('tool')\n", encoding="utf-8")
    helper_dir.mkdir()
    helper.write_text("def fmt(value): return value\n", encoding="utf-8")

    manifest = package_publisher.write_package(
        tmp_path / "store",
        package_id="echo-tool",
        name="Echo Tool",
        description="Example package.",
        file_specs=[f"{tool}=tools/tool_echo.py", f"{helper_dir}=tools/helpers"],
        requires=[],
        tags=["example", "tool"],
        entrypoints=[],
        update=False,
    )

    assert manifest["files"] == ["tools/tool_echo.py", "tools/helpers/echo_format.py"]
    assert (tmp_path / "store" / "packages" / "echo-tool" / "files" / "tools" / "helpers" / "echo_format.py").exists()
    index = json.loads((tmp_path / "store" / "packages" / "index.json").read_text(encoding="utf-8"))
    assert index["packages"][0]["id"] == "echo-tool"
    package_publisher.validate_store(tmp_path / "store")


def test_write_package_refuses_existing_package_without_update(tmp_path):
    source = tmp_path / "tool_echo.py"
    source.write_text("print('tool')\n", encoding="utf-8")
    kwargs = dict(
        store_root=tmp_path / "store",
        package_id="echo-tool",
        name="Echo Tool",
        description="Example package.",
        file_specs=[f"{source}=tools/tool_echo.py"],
        requires=[],
        tags=[],
        entrypoints=[],
    )

    package_publisher.write_package(**kwargs, update=False)

    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.write_package(**kwargs, update=False)


def test_validate_store_rejects_manifest_file_mismatch(tmp_path):
    package_dir = tmp_path / "store" / "packages" / "echo-tool"
    (package_dir / "files" / "tools").mkdir(parents=True)
    (tmp_path / "store" / "packages" / "index.json").write_text(
        json.dumps({"packages": [{"id": "echo-tool", "name": "Echo", "description": "", "tags": []}]}),
        encoding="utf-8",
    )
    (package_dir / "manifest.json").write_text(
        json.dumps({"id": "echo-tool", "requires": [], "files": ["tools/tool_echo.py"]}),
        encoding="utf-8",
    )

    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.validate_store(tmp_path / "store")
