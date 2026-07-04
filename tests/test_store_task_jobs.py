"""Store-ref tests: ask_user_question enum hardening, update_titles
retitled event, and default Timekeeper job declarations.

Materializes the store modules via ``git show`` (same pattern as the other
``test_store_*`` suites) — they only use absolute kernel imports, so they
load as plain modules from a temp directory.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _store_source(rel: str) -> str | None:
    for ref in ("store", "origin/store"):
        proc = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{ref}:{rel}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", check=False)
        if proc.returncode == 0:
            return proc.stdout
    return None


def _load_module(rel: str, tmp_path_factory):
    src = _store_source(rel)
    if src is None:
        pytest.skip(f"{rel} not present on a local store ref")
    stem = Path(rel).stem
    path = tmp_path_factory.mktemp("store_mod") / f"{stem}.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"store_test_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ask_mod(tmp_path_factory):
    return _load_module("tools/tool_ask_user_question.py", tmp_path_factory)


@pytest.fixture(scope="module")
def titles_mod(tmp_path_factory):
    return _load_module("tasks/task_update_titles.py", tmp_path_factory)


@pytest.fixture(scope="module")
def dream_mod(tmp_path_factory):
    return _load_module("tasks/task_dream_memory.py", tmp_path_factory)


# ── ask_user_question enum hardening ──────────────────────────────────

def test_clean_enum_passes_good_options(ask_mod):
    assert ask_mod._clean_enum(["GPS", "IP lookup"]) == (["GPS", "IP lookup"], None)
    assert ask_mod._clean_enum(None) == (None, None)


def test_clean_enum_extracts_option_objects(ask_mod):
    enum, err = ask_mod._clean_enum([{"label": "GPS", "value": "gps"}, {"label": "IP"}])
    assert err is None
    assert enum == ["gps", "IP"]


def test_clean_enum_drops_empty_and_fails_on_total_loss(ask_mod):
    enum, err = ask_mod._clean_enum(["GPS", "", "  "])
    assert (enum, err) == (["GPS"], None)
    enum, err = ask_mod._clean_enum(["", "", ""])
    assert enum is None and "no usable choices" in err
    enum, err = ask_mod._clean_enum("not-a-list")
    assert enum is None and err


def test_run_rejects_all_empty_enum_before_asking(ask_mod):
    # The failure must go back to the agent — the user is never prompted.
    def _never_ask(*_a, **_k):
        raise AssertionError("request_user_input must not be called")

    tool = ask_mod.AskUserQuestion()
    result = tool.run(SimpleNamespace(request_user_input=_never_ask),
                      question="How should the service find your location?",
                      enum=["", "", ""])
    assert not result.success
    assert "no usable choices" in (result.error or "")


# ── update_titles: retitled bus event + default job ───────────────────

def test_update_titles_emits_retitled(titles_mod):
    from events.event_bus import bus
    from events.event_channels import CONVERSATION_CHANGED

    writes, seen = [], []
    db = SimpleNamespace(
        get_conversation_messages=lambda cid: [{"role": "user", "content": "plan my virginia holiday"}],
        update_conversation_title=lambda cid, title: writes.append((cid, title)),
        update_conversation_title_check_count=lambda cid, count: None,
    )
    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="Virginia Holiday", error=None))
    unsub = bus.subscribe(CONVERSATION_CHANGED, seen.append)
    try:
        titles_mod.UpdateTitles()._process_conversation(db, llm, 7, 5)
    finally:
        unsub()
    assert writes == [(7, "Virginia Holiday")]
    assert {"action": "retitled", "conversation_id": 7} in [
        {k: p.get(k) for k in ("action", "conversation_id")} for p in seen]


def test_update_titles_declares_default_job(titles_mod):
    job = titles_mod.UpdateTitles.default_jobs["update_titles"]
    assert job["cron"] == "*/15 * * * *"
    assert job["channel"] == titles_mod.UPDATE_TITLES


def test_dream_memory_declares_default_job(dream_mod):
    job = dream_mod.DreamMemory.default_jobs["dream_memory"]
    assert job["cron"] == "0 4 * * *"
    assert job["channel"] == dream_mod.DREAM_MEMORY
