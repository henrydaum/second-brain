"""Tests for the tree-store publishing utility."""

from __future__ import annotations

import pytest

from scripts import package_publisher
from plugins.commands.helpers import package_manager


def test_write_package_copies_file_and_dir_and_validates(tmp_path):
    tool = tmp_path / "tool_echo.py"
    helper_dir = tmp_path / "helpers"
    helper = helper_dir / "echo_format.py"
    tool.write_text("print('tool')\n", encoding="utf-8")
    helper_dir.mkdir()
    helper.write_text("def fmt(value): return value\n", encoding="utf-8")

    written = package_publisher.write_package(
        tmp_path / "store",
        package_id="tool_echo",
        file_specs=[f"{tool}=tools/tool_echo.py", f"{helper_dir}=tools/helpers"],
        requires=["tools/helpers/echo_format.py"],
        pip=["echo-lib"],
        update=False,
    )

    assert written == ["tools/tool_echo.py", "tools/helpers/echo_format.py"]
    assert (tmp_path / "store" / "tools" / "helpers" / "echo_format.py").exists()
    meta = package_manager.read_dependency_meta("tools/tool_echo.py", (tmp_path / "store" / "tools" / "tool_echo.py").read_text())
    assert meta.dependencies_files == ("tools/helpers/echo_format.py",)
    assert meta.dependencies_pip == ("echo-lib",)
    package_publisher.validate_store(tmp_path / "store")


def test_write_package_refuses_existing_package_without_update(tmp_path):
    source = tmp_path / "tool_echo.py"
    source.write_text("print('tool')\n", encoding="utf-8")
    kwargs = dict(
        store_root=tmp_path / "store",
        package_id="tool_echo",
        file_specs=[f"{source}=tools/tool_echo.py"],
        requires=[],
        pip=None,
    )

    package_publisher.write_package(**kwargs, update=False)
    source.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.write_package(**kwargs, update=False)


def test_validate_store_rejects_missing_dependency(tmp_path):
    path = tmp_path / "store" / "tools" / "tool_echo.py"
    path.parent.mkdir(parents=True)
    path.write_text("dependencies_files = ['tools/helpers/missing.py']\n", encoding="utf-8")

    with pytest.raises(package_publisher.StorePublishError):
        package_publisher.validate_store(tmp_path / "store")


def test_dependency_metadata_is_written_after_future_import(tmp_path):
    source = tmp_path / "tool_future.py"
    source.write_text('"""Doc."""\n\nfrom __future__ import annotations\n\nVALUE = 1\n', encoding="utf-8")

    package_publisher.write_package(
        tmp_path / "store",
        package_id="tool_future",
        file_specs=[f"{source}=tools/tool_future.py"],
        requires=[],
        pip=["future-lib"],
        update=False,
    )

    text = (tmp_path / "store" / "tools" / "tool_future.py").read_text(encoding="utf-8")
    assert text.index("from __future__ import annotations") < text.index("dependencies_pip")
    compile(text, "tool_future.py", "exec")
