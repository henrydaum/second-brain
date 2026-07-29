"""Turn priority must survive a programmatic approval.

``runtime.request_input`` is the dialog behind every sandbox Request
approval. It records the pre-dialog ``previous_priority`` on the frame;
``AnswerApproval`` has to put it back. It didn't — it fell through to
``other_id``, so answering "y" to a Request raised mid-``/llm`` handed
priority to the *agent* and every later user action was refused with
"agent cannot call_command".
"""

import state_machine  # noqa: F401

from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime


def _session(tmp_path):
    db = Database(str(tmp_path / "approval.db"))
    cid = db.create_conversation("x")
    rt = ConversationRuntime(db=db, services={}, config={})
    return rt, rt.load_conversation("s", cid)


def test_answering_a_request_restores_the_user_as_priority(tmp_path):
    rt, session = _session(tmp_path)
    session.cs.set_priority("user")

    req = rt.request_input("s", "Change settings", "config.write", type="boolean")
    assert session.cs.turn_priority == "user"

    out = rt.handle_action("s", "answer_approval", {"value": True, "request_id": req.id})

    assert out.ok
    assert session.cs.turn_priority == "user"


def test_a_request_raised_during_an_agent_turn_hands_back_to_the_agent(tmp_path):
    rt, session = _session(tmp_path)
    session.cs.set_priority("agent")

    req = rt.request_input("s", "Run a command", "proc.run", type="boolean")
    assert session.cs.turn_priority == "user"

    rt.handle_action("s", "answer_approval", {"value": True, "request_id": req.id})

    assert session.cs.turn_priority == "agent"


def test_cancelling_a_request_also_restores_priority(tmp_path):
    rt, session = _session(tmp_path)
    session.cs.set_priority("user")

    rt.request_input("s", "Change settings", "config.write", type="boolean")
    rt.handle_action("s", "cancel", None)

    assert session.cs.turn_priority == "user"
