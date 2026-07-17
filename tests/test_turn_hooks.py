"""Tests for the turn-starter hook: the pre-turn seam for memory recall /
skill retrieval plugins.

Starters run once per logical turn from ``handle_action``'s drive loop, just
before ``_drive_agent_turn``: a restart re-drive (``session.restart_turn``)
skips them, a closing-race follow-up turn runs them again, and a raising
starter never blocks the turn.
"""

import json
from types import SimpleNamespace

# Import the state_machine package before runtime modules to settle the
# package-init circular import (state_machine/__init__ pulls in the runtime).
import state_machine  # noqa: F401

from events.event_bus import bus
from events.event_channels import SESSION_TURN_COMPLETED
from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime


def _response(content="", tool_calls=None):
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        has_tool_calls=bool(tool_calls),
        is_error=False,
        prompt_tokens=0,
    )


class _FakeLLM:
    context_size = 0
    is_llm_backend = True
    model_name = "fake"
    loaded = True

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.calls = []

    def chat_with_tools(self, messages, tools, attachments=None):
        self.calls.append(list(messages))
        return self._responses.pop(0) if self._responses else _response(content="done.")


def _runtime(tmp_path, responses=None):
    db = Database(str(tmp_path / "hooks.db"))
    cid = db.create_conversation("x")
    llm = _FakeLLM(responses)
    rt = ConversationRuntime(db=db, services={"llm": llm}, config={})
    session = rt.load_conversation("s", cid)
    return rt, session, llm


def test_starter_runs_before_drive_and_sees_latest_user_text(tmp_path):
    rt, session, llm = _runtime(tmp_path)
    seen = []

    def starter(sess):
        seen.append((len(llm.calls), [m["content"] for m in sess.history if m["role"] == "user"]))

    rt.hooks.add_turn_starter(starter)
    out = rt.handle_action("s", "send_text", "remember the milk")

    assert out.ok
    calls_at_start, user_texts = seen[0]
    assert calls_at_start == 0  # ran before any LLM call
    assert "remember the milk" in user_texts


def test_starter_prompt_extra_reaches_the_model(tmp_path):
    rt, session, llm = _runtime(tmp_path)

    def starter(sess):
        sess.system_prompt_extras["memory"] = "MEMOMARK-7731"

    rt.hooks.add_turn_starter(starter)
    rt.handle_action("s", "send_text", "hello")

    assert llm.calls, "the drive never reached the model"
    assert "MEMOMARK-7731" in json.dumps(llm.calls[0])


def test_raising_starter_never_blocks_the_turn(tmp_path):
    rt, session, llm = _runtime(tmp_path, [_response(content="fine.")])

    def boom(sess):
        raise RuntimeError("memory service down")

    rt.hooks.add_turn_starter(boom)
    out = rt.handle_action("s", "send_text", "hello")

    assert out.ok
    assert "fine." in out.messages


def test_starter_skipped_on_restart_redrive(tmp_path):
    rt, session, _ = _runtime(tmp_path)
    starts, drives = [], []

    def fake_drive(sess, out, allow_restart=True):
        drives.append(len(drives) + 1)
        if len(drives) == 1:
            sess.restart_turn = True  # what the escalate tool does
        out.messages.append(f"reply {len(drives)}")
        sess.cs.set_priority("user")
        sess.busy = False
        return out

    rt._drive_agent_turn = fake_drive
    rt.hooks.add_turn_starter(lambda sess: starts.append(1))
    rt.handle_action("s", "send_text", "hello")

    assert drives == [1, 2]  # the logical turn was two drives...
    assert len(starts) == 1  # ...but the starter ran once


def test_starter_runs_again_for_closing_race_follow_up_turn(tmp_path):
    rt, session, _ = _runtime(tmp_path)
    starts, drives = [], []

    def fake_drive(sess, out, allow_restart=True):
        drives.append(len(drives) + 1)
        if len(drives) == 1:
            # Simulate the race: a message lands after the loop's final drain.
            sess.pending_user_messages.append("leftover")
        out.messages.append(f"reply {len(drives)}")
        sess.cs.set_priority("user")
        sess.busy = False
        return out

    rt._drive_agent_turn = fake_drive
    rt.hooks.add_turn_starter(lambda sess: starts.append(1))
    rt.handle_action("s", "send_text", "first message")

    assert drives == [1, 2]
    assert len(starts) == 2  # the follow-up is a fresh logical turn


def test_remove_unregisters_a_starter(tmp_path):
    rt, session, _ = _runtime(tmp_path)
    starts = []
    starter = lambda sess: starts.append(1)  # noqa: E731

    rt.hooks.add_turn_starter(starter)
    rt.handle_action("s", "send_text", "one")
    rt.hooks.remove(starter)
    rt.handle_action("s", "send_text", "two")

    assert len(starts) == 1


def test_turn_completed_carries_user_id(tmp_path):
    rt, session, _ = _runtime(tmp_path, [_response(content="Hi.")])
    seen = []
    unsub = bus.subscribe(SESSION_TURN_COMPLETED, lambda p: seen.append(p))
    try:
        rt.handle_action("s", "send_text", "hello")
    finally:
        unsub()

    assert len(seen) == 1
    assert seen[0]["user_id"] == rt.session_user_id("s")
    assert seen[0]["ok"] is True
