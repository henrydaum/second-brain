from plugins.native.frontend import BaseFrontend
from events.event_bus import bus
from pipeline.database import Database
from tests.support import plain_runtime
from state_machine.conversation import CallableSpec, FormStep
from state_machine.conversation_phases import BASE_PHASE, PHASE_APPROVING_REQUEST
from state_machine.serialization import latest_state, save_state_marker


def _db(tmp_path):
    return Database(str(tmp_path / "restart.db"))


def test_stale_busy_marker_recovers_to_user_without_replay(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    db.save_message(cid, "user", "do a thing")
    save_state_marker(db, cid, {"busy": True, "turn_priority": "agent", "phase": BASE_PHASE, "cache": {"phases": []}})

    rt = plain_runtime(db)
    session = rt.load_conversation("s", cid)

    marker = latest_state(db.get_conversation_messages(cid))
    assert session.cs.turn_priority == "user"
    assert session.cs.phase == BASE_PHASE
    assert session.restore_notices
    assert marker["busy"] is False and marker["turn_priority"] == "user"
    assert [m["role"] for m in session.history] == ["user"]


def test_restored_command_form_can_be_reprompted(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    spec = CallableSpec("setup", lambda *_: "done", form=[FormStep("name", "Enter name.")])
    rt = plain_runtime(db, commands={"setup": spec})
    rt.load_conversation("s", cid)
    assert rt.handle_action("s", "call_command", {"name": "setup", "args": {}}).ok

    # Restart: restore re-emits FORM_REQUESTED on the bus, the bound frontend
    # re-prompts the current field — no explicit "render the restored prompt" call.
    frontend = _PromptFrontend()
    rt2 = plain_runtime(db, commands={"setup": spec},
                              emit_event=lambda c, p: bus.emit(c, p))
    frontend.bind(rt2, {})
    try:
        rt2.load_conversation("s", cid)
    finally:
        frontend.unbind()

    assert frontend.forms[-1]["field"]["name"] == "name"


def test_replayable_approval_survives_restart_and_runs(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    ran = []
    spec = CallableSpec("restart", lambda _cs, _actor, args: ran.append(args) or "ok", require_approval=True, approval_actor_id="user")
    rt = plain_runtime(db, commands={"restart": spec})
    rt.load_conversation("s", cid)
    assert rt.handle_action("s", "call_command", {"name": "restart", "args": {}}).ok

    events = []
    rt2 = plain_runtime(db, commands={"restart": spec}, emit_event=lambda c, p: events.append((c, p)))
    rt2.load_conversation("s", cid)
    req = events[-1][1]
    out = rt2.answer_request("s", req.id, True)

    assert out.ok
    assert ran == [{}]
    assert not rt2._approval_requests


def test_process_local_input_request_expires_on_restart(tmp_path):
    db = _db(tmp_path)
    cid = db.create_conversation("x")
    save_state_marker(db, cid, {
        "turn_priority": "user",
        "phase": PHASE_APPROVING_REQUEST,
        "cache": {"phases": [{
            "phase": PHASE_APPROVING_REQUEST,
            "action_type": "answer_approval",
            "actor_id": "user",
            "name": "Need input",
            "data": {"request_id": "r1", "type": "string", "title": "Need input", "prompt": "Value?"},
            "steps": [],
            "step_index": 0,
            "previous_phase": BASE_PHASE,
        }]},
    })

    rt = plain_runtime(db, emit_event=lambda *_: None)
    session = rt.load_conversation("s", cid)

    assert session.cs.phase == BASE_PHASE
    assert session.cs.cache["phases"] == []
    assert session.restore_notices
    assert rt._approval_requests == {}


class _PromptFrontend(BaseFrontend):
    name = "test"

    def __init__(self):
        super().__init__()
        self.forms = []

    def start(self): pass
    def stop(self): pass
    def session_key(self, _ctx=None): return "s"
    def render_messages(self, *_): pass
    def render_attachments(self, *_): pass
    def render_form_field(self, _key, form): self.forms.append(form)
    def render_approval_request(self, *_): pass
    def render_buttons(self, *_): pass
    def render_error(self, *_): pass


# ────────────────────────────────────────────────────────────────────
# Approval priority across a restart (was test_approval_priority.py)
# ────────────────────────────────────────────────────────────────────

import state_machine  # noqa: F401


def _session(tmp_path):
    db = Database(str(tmp_path / "approval.db"))
    cid = db.create_conversation("x")
    rt = plain_runtime(db)
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


def test_option_labels_survive_a_restart(tmp_path):
    """The frame is what a restored dialog is rebuilt from.

    Losing the labels here would leave a person restarting mid-approval with
    buttons reading "always:api.brave.com" — the site most likely to be missed,
    because everything works right up until the process stops.
    """
    rt, session = _session(tmp_path)
    rt.request_input("s", "Reach the network?", "net.http", type="string",
                     enum=["allow", "always:brave.com", "deny"],
                     enum_labels=["Allow once", "Always allow brave.com",
                                  "Deny"])

    frame = session.cs.frame
    assert frame.data["enum_labels"] == ["Allow once", "Always allow brave.com",
                                         "Deny"]

    rebuilt = []
    rt.emit_event = lambda name, req: rebuilt.append(req)
    from runtime.persistence import restore_pending_requests
    restore_pending_requests(rt, session)

    assert rebuilt[-1].enum_labels == ["Allow once", "Always allow brave.com",
                                       "Deny"]


def test_a_typed_option_label_answers_the_dialog(tmp_path):
    """A person answers with the words they were shown, not the value.

    ``AnswerApproval._coerce`` builds the ``FormStep`` that resolves this; if
    it stops passing ``enum_labels`` a multi-choice approval becomes answerable
    only by typing an internal value nothing renders.
    """
    rt, session = _session(tmp_path)
    req = rt.request_input("s", "Reach the network?", "net.http", type="string",
                           enum=["allow", "always:brave.com", "deny"],
                           enum_labels=["Allow once", "Always allow brave.com",
                                        "Deny"])

    rt.handle_action("s", "answer_approval",
                     {"value": "always allow brave.com",
                      "request_id": req.id})

    assert req.value == "always:brave.com"


def test_cancelling_a_request_also_restores_priority(tmp_path):
    rt, session = _session(tmp_path)
    session.cs.set_priority("user")

    rt.request_input("s", "Change settings", "config.write", type="boolean")
    rt.handle_action("s", "cancel", None)

    assert session.cs.turn_priority == "user"
