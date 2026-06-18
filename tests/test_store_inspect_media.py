"""Contract test for the store inspect_media tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.hooks import HookRegistry


def _load_tool():
    path = Path(__file__).resolve().parents[2] / "sb-store" / "tools" / "tool_inspect_media.py"
    if not path.exists():
        pytest.skip("sb-store worktree not present")
    spec = importlib.util.spec_from_file_location("store_tool_inspect_media", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InspectMedia()


def test_inspect_media_stages_attachment_for_current_session(tmp_path):
    class Parser:
        def get_modality(self, ext):
            return "image" if ext == ".gif" else "binary"

        def parse(self, path, modality, config=None):
            return None

    tool = _load_tool()
    media = tmp_path / "clip.gif"
    media.write_bytes(b"gif")
    class Runtime:
        def __init__(self):
            self.hooks = HookRegistry()
            self.sessions = {"s": SimpleNamespace(key="s")}

        def add_turn_attachment(self, session_key, attachment):
            return self.hooks.stage_attachment(self.sessions.get(session_key), attachment)

    runtime = Runtime()
    context = SimpleNamespace(runtime=runtime, session_key="s", services={"parser": Parser()})

    result = tool.run(context, path=str(media))
    staged = runtime.hooks.drain_attachments(runtime.sessions["s"])

    assert result.success
    assert staged[0].file_name == "clip.gif"
    assert staged[0].modality == "image"
