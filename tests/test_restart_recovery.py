from plugins.BaseFrontend import BaseFrontend
from events.event_bus import bus
from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime
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

    rt = ConversationRuntime(db=db, services={}, config={})
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
    rt = ConversationRuntime(db=db, services={}, config={}, commands={"setup": spec})
    rt.load_conversation("s", cid)
    assert rt.handle_action("s", "call_command", {"name": "setup", "args": {}}).ok

    # Restart: restore re-emits FORM_REQUESTED on the bus, the bound frontend
    # re-prompts the current field — no explicit "render the restored prompt" call.
    frontend = _PromptFrontend()
    rt2 = ConversationRuntime(db=db, services={}, config={}, commands={"setup": spec},
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
    rt = ConversationRuntime(db=db, services={}, config={}, commands={"restart": spec})
    rt.load_conversation("s", cid)
    assert rt.handle_action("s", "call_command", {"name": "restart", "args": {}}).ok

    events = []
    rt2 = ConversationRuntime(db=db, services={}, config={}, commands={"restart": spec}, emit_event=lambda c, p: events.append((c, p)))
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

    rt = ConversationRuntime(db=db, services={}, config={}, emit_event=lambda *_: None)
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
