"""Regression tests for dream memory."""

import time
from types import SimpleNamespace

from config.config_data import DEFAULT_SCHEDULED_JOBS
from plugins.tasks import task_dream_memory as dream


class FakeDb:
    """Test double for fake DB."""
    def __init__(self):
        """Initialize the fake DB."""
        self.cid = 1

    def list_conversations(self, limit=50):
        """List conversations."""
        return [{"id": self.cid, "title": "Prefs", "kind": "user", "category": None, "updated_at": time.time()}]

    def get_conversation_messages(self, conversation_id):
        """Get conversation messages."""
        return [{"role": "user", "content": "Please remember that I like surgical changes."}]


class FakeLlm:
    """Test double for fake LLM."""
    loaded = True

    def __init__(self, *contents):
        """Initialize the fake LLM."""
        self.contents = list(contents)
        self.kwargs = []

    def invoke(self, messages, **kwargs):
        """Handle invoke."""
        self.kwargs.append(kwargs)
        return SimpleNamespace(content=self.contents.pop(0), error=None)


def _patch_paths(monkeypatch, tmp_path):
    """Internal helper to handle patch paths."""
    monkeypatch.setattr(dream, "MEMORY_PATH", tmp_path / "memory.md")
    monkeypatch.setattr(dream, "STATE_PATH", tmp_path / "memory_dream_state.json")
    monkeypatch.setattr(dream, "REPORT_PATH", tmp_path / "memory_dream_report.md")
    monkeypatch.setattr(dream, "BACKUP_PATH", tmp_path / "memory.md.bak")


def test_dream_memory_rewrites_memory_and_writes_report(monkeypatch, tmp_path):
    """Verify dream memory rewrites memory and writes report."""
    _patch_paths(monkeypatch, tmp_path)
    dream.MEMORY_PATH.write_text("# User\n\n- Old\n", encoding="utf-8")
    llm = FakeLlm('{"memory_md":"# User\\n\\n- Likes surgical changes.\\n\\n# Projects\\n\\n# Operating Lessons\\n\\n# Do Not Do\\n","changes":["added preference"],"skipped":["none"]}')

    result = dream.DreamMemory().run_event("run", {}, SimpleNamespace(db=FakeDb(), services={"llm": llm}, config={}))

    assert result.success
    assert "Likes surgical changes" in dream.MEMORY_PATH.read_text(encoding="utf-8")
    assert dream.BACKUP_PATH.exists()
    assert "added preference" in dream.REPORT_PATH.read_text(encoding="utf-8")
    assert llm.kwargs[0] == {"response_format": {"type": "json_object"}}


def test_dream_memory_accepts_fenced_json():
    """Verify dream memory accepts fenced JSON."""
    assert dream._extract_json('```json\n{"memory_md":"# User\\n","changes":[],"skipped":[]}\n```')["memory_md"] == "# User\n"


def test_dream_memory_invalid_json_preserves_memory(monkeypatch, tmp_path):
    """Verify dream memory invalid JSON preserves memory."""
    _patch_paths(monkeypatch, tmp_path)
    dream.MEMORY_PATH.write_text("# User\n\n- Keep me\n", encoding="utf-8")
    llm = FakeLlm("nope", "still nope", "bad")

    result = dream.DreamMemory().run_event("run", {}, SimpleNamespace(db=FakeDb(), services={"llm": llm}, config={}))

    assert not result.success
    assert dream.MEMORY_PATH.read_text(encoding="utf-8") == "# User\n\n- Keep me\n"
    assert not dream.BACKUP_PATH.exists()
    assert "Invalid dream JSON" in dream.REPORT_PATH.read_text(encoding="utf-8")


def test_default_scheduled_jobs_include_titles_and_dream_memory():
    """Verify default scheduled jobs include titles and dream memory."""
    assert DEFAULT_SCHEDULED_JOBS["update_titles"]["cron"] == "*/30 * * * *"
    assert DEFAULT_SCHEDULED_JOBS["dream_memory"]["cron"] == "0 4 * * *"
