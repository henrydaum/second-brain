"""Contract tests for the store LiteLLM attachment adapter."""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import pytest

from attachments.attachment import Attachment, AttachmentBundle


def _load_litellm_service():
    path = Path(__file__).resolve().parents[2] / "sb-store" / "services" / "service_litellm.py"
    if not path.exists():
        pytest.skip("sb-store worktree not present")
    spec = importlib.util.spec_from_file_location("store_service_litellm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LiteLLMService


def test_litellm_injects_image_audio_and_video_blocks(tmp_path, monkeypatch):
    LiteLLMService = _load_litellm_service()
    llm = LiteLLMService("openai/test")
    image = tmp_path / "image.png"
    audio = tmp_path / "clip.wav"
    video = tmp_path / "movie.mp4"
    image.write_bytes(b"fake-image")
    audio.write_bytes(b"audio-bytes")
    video.write_bytes(b"video-bytes")
    monkeypatch.setattr(llm, "_image_data_url", lambda _path: "data:image/jpeg;base64,abc")

    messages = llm._inject_attachments(
        [{"role": "user", "content": "inspect"}],
        AttachmentBundle([
            Attachment(str(image), ".png", "image.png", "image"),
            Attachment(str(audio), ".wav", "clip.wav", "audio"),
            Attachment(str(video), ".mp4", "movie.mp4", "video"),
        ]),
    )

    content = messages[0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "input_audio", "video_url"]
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,abc"
    assert content[2]["input_audio"] == {"data": base64.b64encode(b"audio-bytes").decode("utf-8"), "format": "wav"}
    assert content[3]["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_litellm_attachment_mismatch_falls_back_to_parsed_text(tmp_path):
    LiteLLMService = _load_litellm_service()
    llm = LiteLLMService("openai/test")
    gif = tmp_path / "clip.gif"
    gif.write_bytes(b"gif-bytes")

    messages = llm._inject_attachments(
        [{"role": "user", "content": "inspect"}],
        AttachmentBundle([Attachment(str(gif), ".gif", "clip.gif", "video", parsed_text="animated frames summary")]),
    )

    assert len(messages[0]["content"]) == 1
    assert messages[0]["content"][0]["type"] == "text"
    assert "clip.gif" in messages[0]["content"][0]["text"]
    assert "Parsed contents:\nanimated frames summary" in messages[0]["content"][0]["text"]
