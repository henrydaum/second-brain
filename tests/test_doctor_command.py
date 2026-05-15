"""Regression tests for `/doctor`."""

from types import SimpleNamespace

from plugins.commands import command_doctor
from plugins.commands.command_doctor import DoctorCommand


class FakeDb:
    """Test double for doctor DB calls."""
    def get_system_stats(self):
        """Return task and file stats."""
        return {"files": {"text": 2}, "tasks": {"extract_text": {"FAILED": 1, "PENDING": 2}}}

    def get_run_stats(self):
        """Return event-run stats."""
        return {"spawn_subagent": {"PROCESSING": 1}}


class FakeTimekeeper:
    """Test double for timekeeper."""
    loaded = True

    def list_jobs(self):
        """Return jobs."""
        return {"daily": {"enabled": True}, "old": {"enabled": False}}


def test_doctor_reports_runtime_findings_and_recent_log_errors(tmp_path, monkeypatch):
    """Verify doctor summarizes health and log problems."""
    monkeypatch.setattr(command_doctor, "DATA_DIR", tmp_path)
    (tmp_path / "app.log").write_text(
        "01:00PM | Main         | INFO  | ok\n"
        "01:01PM | Discovery    | WARNING | Plugin registration failed: demo\n"
        "01:02PM | Main         | ERROR | Auto-load failed for 'llm': boom\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(
        config={"sync_directories": [str(tmp_path / "missing")], "autoload_services": ["llm", "ghost"], "enabled_frontends": ["repl"]},
        services={"llm": SimpleNamespace(active=SimpleNamespace(model_name="gpt-5", loaded=True, last_prompt_tokens=1500, last_cached_prompt_tokens=1024)), "timekeeper": FakeTimekeeper()},
        orchestrator=SimpleNamespace(tasks={"extract_text": object(), "spawn_subagent": object()}, paused={"extract_text"}),
        tool_registry=SimpleNamespace(tools={"read_file": object()}),
        db=FakeDb(),
        runtime=SimpleNamespace(sessions={"chat": SimpleNamespace(plan_mode=True, full_permissions_this_turn=True)}),
        session_key="chat",
    )

    out = DoctorCommand().run({}, context)

    assert "Doctor:" in out
    assert "autoload service not discovered: ghost" in out
    assert "autoload service not loaded: llm" in out
    assert "extract_text: 2 pending, 0 running, 1 failed" in out
    assert "Plan mode: on, full permissions this turn" in out
    assert "spawn_subagent: 0 pending, 1 running, 0 failed" in out
    assert "2 job(s), 1 disabled" in out
    assert "Prompt cache: 1024/1500 input tokens cached on last call (68%)." in out
    assert "Plugin registration failed: demo" in out
    assert "Auto-load failed for 'llm': boom" in out
