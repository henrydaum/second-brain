"""Deterministic races at turn admission, approval and completion boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

import state_machine  # noqa: F401 -- initialize the runtime import cycle
from runtime import runtime_config
from tests.support import plain_runtime, FakeLLM
from pipeline.database import Database


@pytest.fixture
def runtime(tmp_path):
    rt = plain_runtime(Database(str(tmp_path / "concurrency.db")), services={"llm": FakeLLM()})
    rt.load_conversation("shared", rt.create_conversation("Shared"))
    return rt


@pytest.mark.parametrize("boundary", ["start_turn", "finish_turn"])
def test_turn_ownership_covers_hooks(runtime, monkeypatch, boundary):
    entered, release = Event(), Event()
    session = runtime.get_session("shared")
    original = getattr(runtime.hooks, boundary)

    def block(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.hooks, boundary, block)
    with ThreadPoolExecutor(max_workers=1) as pool:
        turn = pool.submit(runtime.handle_action, "shared", "send_text", "first")
        try:
            assert entered.wait(5)
            assert session.in_flight
            assert session.to_marker()["busy"]
            assert not runtime.close_session("shared")
            with pytest.raises(RuntimeError):
                runtime.reset_conversation("shared")
            original_user = session.user_id
            with pytest.raises(RuntimeError):
                runtime.set_session_user("shared", 99)
            assert session.user_id == original_user
            cid = session.conversation_id
            assert not runtime.delete_conversation("other", cid)
            with pytest.raises(RuntimeError):
                runtime.clear_conversation("other", cid)
            assert runtime.db.get_conversation(cid) is not None
            queued = runtime.handle_action("shared", "send_text", "second")
            assert queued.data.get("queued") is True
            # Do not block the follow-up turn at this boundary again.
            monkeypatch.setattr(runtime.hooks, boundary, original)
        finally:
            release.set()
        assert turn.result(timeout=5).ok
    assert not session.in_flight
    assert session.cs.turn_priority == "user"
    assert [m["content"] for m in session.history if m["role"] == "user"] == ["first", "second"]


def test_cancel_before_driver_starts_is_not_cleared(runtime, monkeypatch):
    entered, release = Event(), Event()
    session = runtime.get_session("shared")
    original = runtime.hooks.start_turn

    def block(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.hooks, "start_turn", block)
    with ThreadPoolExecutor(max_workers=1) as pool:
        turn = pool.submit(runtime.handle_action, "shared", "send_text", "first")
        try:
            assert entered.wait(5)
            assert runtime.handle_action("shared", "cancel").data["cancelled"]
        finally:
            release.set()
        assert turn.result(timeout=5).messages == []
    assert not session.in_flight
    assert not session.cancel_event.is_set()


def test_cancel_during_approval_stops_turn_and_wakes_request(runtime, monkeypatch):
    entered = Event()
    requests = []
    session = runtime.get_session("shared")

    class Loop:
        def drive(self, *args):
            req = runtime.request_input("shared", "Permission", "Proceed?")
            requests.append(req)
            entered.set()
            assert req.wait(5)
            assert session.cancel_event.is_set()
            assert req.metadata["cancelled"]
            return None, [], []

    monkeypatch.setattr(runtime_config, "build_loop", lambda *_: Loop())
    with ThreadPoolExecutor(max_workers=1) as pool:
        turn = pool.submit(runtime.handle_action, "shared", "send_text", "first")
        try:
            assert entered.wait(5)
            assert runtime.handle_action("shared", "cancel").data["cancelled"]
            assert turn.result(timeout=5).messages == []
        finally:
            for req in requests:
                req.resolve(None)
    assert not session.in_flight
    assert not runtime._approval_requests


def test_two_answers_to_same_approval_only_settle_once(runtime):
    req = runtime.request_input("shared", "Permission", "Proceed?")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda value: runtime.answer_request("shared", req.id, value), [True, False]))
    assert sum(result.ok for result in results) == 1
    assert req.is_resolved
    assert not runtime._approval_requests


def test_clearing_from_another_session_refreshes_the_live_owner(runtime):
    session = runtime.get_session("shared")
    cid = session.conversation_id
    runtime.inject_user_message("shared", "old message")
    assert runtime.clear_conversation("other", cid)
    assert runtime.get_session("shared").history == []
    assert runtime.get_session("shared").conversation_id == cid


def test_stale_cancel_cannot_dismiss_a_newer_approval(runtime):
    first = runtime.request_input("shared", "First", "Proceed?")
    assert runtime.answer_request("shared", first.id, True).ok
    second = runtime.request_input("shared", "Second", "Proceed again?")
    out = runtime.handle_action("shared", "cancel", {"request_id": first.id})
    assert not out.ok
    assert not second.is_resolved
    assert runtime.answer_request("shared", second.id, False).ok


def test_queued_background_turn_does_not_rewrite_active_history(runtime, monkeypatch):
    session = runtime.get_session("shared")
    session.driver_token = object()
    writes = []
    monkeypatch.setattr(runtime.db, "replace_conversation_messages", lambda *args: writes.append(args))
    try:
        out = runtime.iterate_agent_turn("shared", "queued")
        assert out.data["queued"]
        assert writes == []
    finally:
        session.driver_token = None


def test_approval_created_after_cancel_is_already_denied(runtime):
    session = runtime.get_session("shared")
    session.driver_token = object()
    session.cancel_event.set()
    req = runtime.request_input("shared", "Too late", "Proceed?")
    assert req.is_resolved
    assert not req.approved
    assert req.metadata["cancelled"]
    assert not runtime._approval_requests


def test_stale_session_cannot_persist_over_its_replacement(runtime):
    from runtime.persistence import persist_marker
    session = runtime.get_session("shared")
    cid = session.conversation_id
    assert runtime.close_session("shared")
    runtime.load_conversation("shared", cid)
    before = runtime.db.get_conversation_messages(cid)
    session.busy = True
    persist_marker(runtime, session)
    assert runtime.db.get_conversation_messages(cid) == before


def test_legacy_orphaned_agent_priority_recovers_to_user(runtime):
    from state_machine.serialization import save_state_marker
    from state_machine.conversation_phases import BASE_PHASE
    cid = runtime.get_session("shared").conversation_id
    runtime.close_session("shared")
    save_state_marker(runtime.db, cid, {"busy": False, "phase": BASE_PHASE,
                                      "turn_priority": "agent", "cache": {"phases": []}})
    session = runtime.load_conversation("shared", cid)
    assert session.cs.turn_priority == "user"
    assert session.restore_notices


def test_failed_start_hook_releases_turn_ownership(runtime, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("start failed")

    monkeypatch.setattr(runtime.hooks, "start_turn", fail)
    with pytest.raises(RuntimeError, match="start failed"):
        runtime.handle_action("shared", "send_text", "hello")
    session = runtime.get_session("shared")
    assert not session.in_flight
    assert session.turn_id is None
    assert session.cs.turn_priority == "user"
    assert runtime.close_session("shared")


def test_lifecycle_mutation_does_not_wait_on_dispatch_lock(runtime):
    entered, release = Event(), Event()
    session = runtime.get_session("shared")

    def hold():
        with session.lock:
            entered.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        held = pool.submit(hold)
        try:
            assert entered.wait(5)
            assert not runtime.close_session("shared")
            with pytest.raises(RuntimeError):
                runtime.clear_conversation("other", session.conversation_id)
        finally:
            release.set()
        held.result(timeout=5)


def test_stale_action_cannot_write_after_session_replacement(runtime, monkeypatch):
    stale = runtime.get_session("shared")
    runtime.reset_conversation("shared")
    monkeypatch.setattr(runtime, "get_session", lambda _: stale)
    result = runtime.handle_action("shared", "send_text", "too late")
    assert not result.ok
    assert result.error["code"] == "session_changed"
