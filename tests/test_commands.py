"""Tests for the kernel slash commands.

Only the REPL/introspection commands ship in the kernel. These exercise the
two with non-trivial logic: ``/llm`` (profile management form + handler) and
``/doctor`` (runtime health summary). Both stub their context dependencies.
"""

from types import SimpleNamespace

from plugins.commands import command_doctor
from plugins.commands.command_doctor import DoctorCommand
from plugins.commands.command_llm import LlmCommand


# ── /llm ─────────────────────────────────────────────────────────────

def test_llm_command_can_set_default(monkeypatch):
    saved = []
    monkeypatch.setattr("plugins.commands.command_llm._save", lambda config: saved.append(dict(config)))
    context = SimpleNamespace(config={"llm_profiles": {"a": {}, "b": {}}, "default_llm_profile": "a"}, services={})

    steps = LlmCommand().form({"model_name": "b"}, context)
    result = LlmCommand().run({"model_name": "b", "action": "set_default"}, context)

    assert steps[0].prompt == "Select an LLM profile, or add a new one.\nDefault: a"
    assert steps[1].enum == ["edit", "set_default", "remove"]
    assert steps[1].enum_labels == ["Edit", "Set default", "Remove"]
    assert result == "Default LLM profile set to: b"
    assert context.config["default_llm_profile"] == "b"
    assert saved[-1]["default_llm_profile"] == "b"


# ── /doctor ──────────────────────────────────────────────────────────

class _DoctorDb:
    def get_system_stats(self):
        return {"files": {"text": 2}, "tasks": {"extract_text": {"FAILED": 1, "PENDING": 2}}}

    def get_run_stats(self):
        return {"spawn_subagent": {"PROCESSING": 1}}


class _Timekeeper:
    loaded = True

    def list_jobs(self):
        return {"daily": {"enabled": True}, "old": {"enabled": False}}


def test_doctor_reports_runtime_findings_and_recent_log_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(command_doctor, "DATA_DIR", tmp_path)
    (tmp_path / "app.log").write_text(
        "01:00PM | Main         | INFO  | ok\n"
        "01:01PM | Discovery    | WARNING | Plugin registration failed: demo\n"
        "01:02PM | Main         | ERROR | Auto-load failed for 'llm': boom\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(
        config={"sync_directories": [str(tmp_path / "missing")], "autoload_services": ["llm", "ghost"], "enabled_frontends": ["repl"]},
        services={"llm": SimpleNamespace(active=SimpleNamespace(model_name="gpt-5", loaded=True)), "timekeeper": _Timekeeper()},
        orchestrator=SimpleNamespace(tasks={"extract_text": object(), "spawn_subagent": object()}, paused={"extract_text"}),
        tool_registry=SimpleNamespace(tools={"read_file": object()}),
        db=_DoctorDb(),
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
    assert "Model: gpt-5 (loaded)" in out
    assert "Plugin registration failed: demo" in out
    assert "Auto-load failed for 'llm': boom" in out
