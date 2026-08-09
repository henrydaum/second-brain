"""A question that stops waiting says so.

``APPROVAL_REQUESTED`` had no counterpart. A frontend learned a question
existed by being handed one to render, and there was no event at all for the
other end of its life -- so a surface that draws a dialog could not be told to
take it down. That is not a hypothetical gap: another frontend can answer the
same question, and the approver denies by name after its 300s timeout. Neither
is something the frontend holding the dialog did, and without an event neither
is something it could find out except by asking on a timer.

What these pin is that the announcement means *the frame is gone*, on every
route out, rather than "somebody called resolve".
"""

import pytest

import state_machine  # noqa: F401  (break the runtime import cycle)

from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime
from state_machine.conversation_phases import BASE_PHASE, PHASE_APPROVING_REQUEST


@pytest.fixture
def make_runtime(tmp_path):
    """A runtime whose events land in a list instead of on the bus.

    Torn down rather than left to the garbage collector. A session parked in
    ``approving_request`` holds a live ``threading.Event`` that nothing is ever
    going to set, and a runtime left holding one outlives this module — which
    showed up as an unrelated sandbox test failing later in the same run, the
    way leaked state always does.
    """
    made: list[ConversationRuntime] = []

    def build():
        seen: list[tuple[str, object]] = []
        db = Database(str(tmp_path / f"settle{len(made)}.db"))
        cid = db.create_conversation("Notes", user_id=1)
        runtime = ConversationRuntime(
            db=db, services={}, config={},
            emit_event=lambda channel, payload: seen.append((channel, payload)))
        made.append(runtime)
        session = runtime.get_session("repl")
        session.conversation_id = cid
        return runtime, session, seen

    yield build

    for runtime in made:
        for request in list(runtime._approval_requests.values()):
            if not request.is_resolved:
                request.resolve(None)
        runtime._approval_requests.clear()
        for key in list(runtime.sessions):
            try:
                runtime.close_session(key)
            except Exception:
                pass


def _settled(seen):
    return [payload for channel, payload in seen if channel == "approval_settled"]


def _ask(runtime, **kwargs):
    return runtime.request_input("repl", "Run a shell command",
                                 "rm -rf /tmp/x", **kwargs)


def test_answering_settles_it(make_runtime):
    runtime, session, seen = make_runtime()
    request = _ask(runtime)
    assert session.cs.phase == PHASE_APPROVING_REQUEST

    runtime.handle_action("repl", "answer_approval",
                          {"value": True, "request_id": request.id})

    assert session.cs.phase == BASE_PHASE
    assert _settled(seen) == [{"session_key": "repl",
                               "request_id": request.id,
                               "reason": "answered"}]


def test_cancelling_settles_it_and_says_which(make_runtime):
    """``reason`` says how the question ended, not what the answer was -- the
    answer went to whoever was blocked on it, and is not a bystander's."""
    runtime, session, seen = make_runtime()
    request = _ask(runtime)

    runtime.handle_action("repl", "cancel", {})

    assert _settled(seen) == [{"session_key": "repl",
                               "request_id": request.id,
                               "reason": "cancelled"}]
    assert request.metadata.get("cancelled")


def test_the_timeout_path_settles_it_too(make_runtime):
    """The approver denies by name after 300s through ``answer_request``. That
    is the route no frontend can see coming, and the one this exists for."""
    runtime, _session, seen = make_runtime()
    request = _ask(runtime, type="string", enum=["allow", "deny"])

    runtime.answer_request("repl", request.id, "deny")

    assert [payload["reason"] for payload in _settled(seen)] == ["answered"]
    assert request.value == "deny"


def test_a_refused_answer_settles_nothing(make_runtime):
    """Still in the phase means the question is still waiting. Announcing here
    would take a live dialog off the screen."""
    runtime, session, seen = make_runtime()
    _ask(runtime, type="string", enum=["allow", "deny"])

    runtime.handle_action("repl", "answer_approval", {"value": "maybe"})

    assert session.cs.phase == PHASE_APPROVING_REQUEST
    assert _settled(seen) == []


def test_it_settles_even_with_no_live_request_to_fulfil(make_runtime):
    """A gated command's approval is rebuilt from the phase frame rather than
    held in ``_approval_requests``, so there is no object to resolve -- the
    ``pop`` finds nothing and the old code did nothing at all. The id is what a
    frontend is holding, so the id is what it has to hear about."""
    from runtime import runtime_approvals
    from state_machine.errors import ActionResult

    runtime, _session, seen = make_runtime()
    assert runtime._approval_requests == {}

    runtime_approvals.resolve_answered_request(
        runtime, "repl", "approve_callable",
        ActionResult(True, "answer_approval", data={"approved": False}))

    assert _settled(seen) == [{"session_key": "repl",
                               "request_id": "approve_callable",
                               "reason": "answered"}]
