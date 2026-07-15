"""Tests for the Claude Code store bundle manifest."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_REL = "bundles/bundle_claude_code.json"


def _store_ref() -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{_REL}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return ref
    return None


@pytest.fixture(scope="module")
def bundle():
    ref = _store_ref()
    if ref is None:
        pytest.skip("claude_code bundle not present on a local store ref")
    text = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{ref}:{_REL}"],
        stdout=subprocess.PIPE, text=True, encoding="utf-8", check=True).stdout
    return ref, json.loads(text)


def test_manifest_shape(bundle):
    _, manifest = bundle
    assert manifest["name"] == "Claude Code"
    assert manifest["description"]
    assert isinstance(manifest["files"], list) and manifest["files"]


def test_files_match_family_naming(bundle):
    _, manifest = bundle
    for rel in manifest["files"]:
        assert re.fullmatch(r"tools/tool_[a-z0-9_]+\.py", rel), rel


def test_files_exist_on_ref(bundle):
    ref, manifest = bundle
    for rel in manifest["files"]:
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        assert proc.returncode == 0, f"{rel} missing on {ref}"


def test_new_tools_included(bundle):
    _, manifest = bundle
    files = set(manifest["files"])
    for tool in ("tool_grep", "tool_glob", "tool_todo", "tool_spawn_agent"):
        assert f"tools/{tool}.py" in files
