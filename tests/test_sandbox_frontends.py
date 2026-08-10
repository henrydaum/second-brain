"""The frontend Requests, and the desk a token reaches.

A frontend is the one family that acts *for a person*: it says "someone typed
this" and the kernel believes it. So the security question is not what the
Request asks for but **which frontend is asking**, and the answer is
structural — a token parked when the box opened, resolving to that frontend's
own adapter and nothing else.

These tests are about that scoping. Rendering and the poll loop belong to the
bridge and are tested with it.
"""

import threading
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.bridge import get_sandbox
from sandbox.frontends import KINDS, desk, park, project_approval, unpark

# A frontend that does nothing but exercise sdk.frontend. It is opened as a
# plain box rather than through the bridge, so what is under test is the
# Request path alone.
TALKER = '''
"""A migrated frontend."""

from guest.bases import BaseFrontend


class Talker(BaseFrontend):
    """Submits whatever it is told to."""

    name = "talker"
    ISOLATION

    def start(self, sdk):
        """Nothing to open."""
        return True

    def poll(self, sdk):
        """Nothing arrives on its own here."""
        return False

    def say(self, sdk, text="hi"):
        """Carry a line in, as a real frontend would from its transport."""
        return sdk.frontend.submit_text("s1", text)

    def approve(self, sdk, value=True):
        """Answer whatever approval is pending."""
        return sdk.frontend.resolve("s1", value)

    def whoami(self, sdk, external_id=None):
        """Bind the session to a user."""
        return sdk.frontend.bind("s1", external_id)

    def token(self, sdk):
        """The token this box is holding. For the tests only."""
        return sdk._frontend_token
'''


class FakeAdapter:
    """Stands in for the native adapter a token resolves to."""

    def __init__(self):
        self.calls = []

    def submit_text(self, session_key, text):
        """Record and succeed."""
        self.calls.append(("submit_text", session_key, text))
        return SimpleNamespace(ok=True)

    def bind_session(self, session_key):
        """The default-user path."""
        self.calls.append(("bind_session", session_key))
        return 1

    def identify(self, session_key, external_id, config, user_type="user"):
        """The per-user upgrade path."""
        self.calls.append(("identify", session_key, external_id, user_type))
        return 42

    def resolve_next_approval(self, session_key, value):
        """Answer the session's next pending request."""
        self.calls.append(("resolve_next", session_key, value))
        return True

    def resolve_approval(self, session_key, request_id, value):
        """Answer one pending request by id."""
        self.calls.append(("resolve", session_key, request_id, value))
        return True

    #: Which approvals this stand-in claims are still waiting. The handler
    #: settles existence synchronously before it detaches the answering, so a
    #: double that always says "nothing pending" would make resolve a no-op.
    pending = True

    def is_approval_pending(self, session_key, request_id=None):
        """Whether there is still something to answer."""
        self.calls.append(("is_pending", session_key, request_id))
        return self.pending


@pytest.fixture
def box():
    """A sandbox torn down even if a test fails."""
    made = Sandbox()
    yield made
    made.shutdown()


@pytest.fixture
def adapter():
    """A parked adapter whose desk is always cleared."""
    made = FakeAdapter()
    made.token = park(made)
    yield made
    unpark(made.token)


def _open(box, tmp_path, isolation=""):
    """Open a frontend box directly, without the bridge."""
    source = TALKER.replace(
        "ISOLATION", f'isolation = "{isolation}"' if isolation else "")
    path = tmp_path / "frontend_talker.py"
    path.write_text(source, encoding="utf-8")
    return box.open(path, "Talker", name="frontend_talker")


# ──────────────────────────────────────────────────────────────────────
# The desk.
# ──────────────────────────────────────────────────────────────────────

def test_a_token_resolves_to_its_own_adapter():
    """The whole mechanism in one line."""
    first, second = FakeAdapter(), FakeAdapter()
    one, two = park(first), park(second)
    try:
        assert desk(one) is first
        assert desk(two) is second
    finally:
        unpark(one)
        unpark(two)


def test_clearing_a_desk_revokes_the_token():
    """A token that outlived its frontend must reach nothing."""
    adapter = FakeAdapter()
    token = park(adapter)
    unpark(token)

    assert desk(token) is None


def test_no_token_reaches_nothing():
    """What every non-frontend holds."""
    assert desk("") is None
    assert desk("made-up") is None


# ──────────────────────────────────────────────────────────────────────
# The Requests, from inside a real box.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("isolation", ["", "subprocess"])
def test_a_frontend_request_without_a_token_is_refused(box, tmp_path,
                                                       isolation):
    """Nothing bound this box, so it speaks for nobody.

    This is the case that matters: a tool or a script that imported the SDK
    reaches the same namespace and must get nowhere with it.
    """
    opened = _open(box, tmp_path, isolation)
    try:
        result = opened.call("say", text="hello")
    finally:
        box.close("frontend_talker")
        unload_box("frontend_talker")

    assert not result.ok
    assert "sdk.frontend" in result.error


@pytest.mark.parametrize("isolation", ["", "subprocess"])
def test_binding_a_box_lets_it_submit(box, tmp_path, adapter, isolation):
    """The token survives ``__bind__`` and is still there on a later call.

    Both runners, because a subprocess holds its own SDK instance across the
    boundary — if the token did not persist there, every frontend would work
    in-process and fail the moment it was isolated.
    """
    opened = _open(box, tmp_path, isolation)
    try:
        assert opened.call("__bind__", token=adapter.token).ok
        assert opened.call("token").data == adapter.token   # it stuck
        result = opened.call("say", text="hello")
    finally:
        box.close("frontend_talker")
        unload_box("frontend_talker")

    assert result.ok, result.error
    assert adapter.calls == [("submit_text", "s1", "hello")]


def test_a_revoked_token_stops_working(box, tmp_path, adapter):
    """Stopping a frontend takes its authority with it, mid-life."""
    opened = _open(box, tmp_path)
    try:
        assert opened.call("__bind__", token=adapter.token).ok
        assert opened.call("say", text="before").ok

        unpark(adapter.token)                 # as ``stop()`` does
        result = opened.call("say", text="after")
    finally:
        box.close("frontend_talker")
        unload_box("frontend_talker")

    assert not result.ok
    assert adapter.calls == [("submit_text", "s1", "before")]


def test_one_frontend_cannot_reach_another(box, tmp_path):
    """Two frontends, two desks. Holding one's token reaches one adapter."""
    mine, theirs = FakeAdapter(), FakeAdapter()
    my_token, their_token = park(mine), park(theirs)

    opened = _open(box, tmp_path)
    try:
        opened.call("__bind__", token=my_token)
        opened.call("say", text="mine")
    finally:
        box.close("frontend_talker")
        unload_box("frontend_talker")
        unpark(my_token)
        unpark(their_token)

    assert mine.calls == [("submit_text", "s1", "mine")]
    assert theirs.calls == []


def test_bind_picks_its_path_from_the_arguments(box, tmp_path, adapter):
    """Which native call runs is decided by the kernel, not by the plugin.

    A frontend naming no identity gets its declared default user; naming one
    gets that identity's own user. A plugin cannot pick ``identify`` for a
    session it did not authenticate, because it never names the method.
    """
    opened = _open(box, tmp_path)
    try:
        assert opened.call("__bind__", token=adapter.token).ok
        default = opened.call("whoami")
        upgraded = opened.call("whoami", external_id="u-77")
    finally:
        box.close("frontend_talker")
        unload_box("frontend_talker")

    assert default.data == 1
    assert upgraded.data == 42
    assert adapter.calls == [("bind_session", "s1"),
                             ("identify", "s1", "u-77", "user")]


def test_resolving_without_an_id_answers_the_next_one(box, tmp_path, adapter):
    """What a transport showing one message at a time wants."""
    opened = _open(box, tmp_path)
    try:
        assert opened.call("__bind__", token=adapter.token).ok
        result = opened.call("approve", value=True)
    finally:
        box.close("frontend_talker")
        unload_box("frontend_talker")

    assert result.data is True
    # Existence is settled first and synchronously, then the answering is
    # detached — so the verdict stays truthful without holding the box lock
    # across a turn. See ``_frontend_resolve``.
    assert adapter.calls == [("is_pending", "s1", None),
                             ("resolve_next", "s1", True)]


# ──────────────────────────────────────────────────────────────────────
# Projection.
# ──────────────────────────────────────────────────────────────────────

def test_an_approval_crosses_as_a_question_not_a_decision():
    """The box gets what it needs to ask, and nothing it could act on.

    The live Event the state machine waits on, and the action being
    authorized, must not cross — holding the id is enough to *answer*, and
    only to answer.
    """
    from state_machine.approval import StateMachineApprovalRequest

    request = StateMachineApprovalRequest(
        title="Delete everything?", body="Really?",
        pending_action={"tool": "rm", "args": {"path": "/"}},
        type="boolean", default=False)

    projected = project_approval(request)

    assert projected["title"] == "Delete everything?"
    assert projected["id"] == request.id
    assert projected["type"] == "boolean"
    assert "pending_action" not in projected
    assert "_event" not in projected
    # Whatever it carries has to survive a round trip as plain data.
    import json
    json.loads(json.dumps(projected))


def test_an_approvals_options_cross_with_their_labels():
    """Values answer, labels read — and they pair by index.

    A frontend that got ``enum`` without ``enum_labels`` would render an
    approval's internal option values ("always:api.brave.com") as button text,
    which is the whole reason this field exists.
    """
    import json

    from state_machine.approval import StateMachineApprovalRequest

    request = StateMachineApprovalRequest(
        title="Reach the network?", body="",
        type="string", enum=["allow", "always:brave.com", "deny"],
        enum_labels=["Allow once", "Always allow brave.com", "Deny"])

    projected = project_approval(request)

    assert projected["enum"] == ["allow", "always:brave.com", "deny"]
    assert projected["enum_labels"] == ["Allow once", "Always allow brave.com",
                                        "Deny"]
    assert len(projected["enum"]) == len(projected["enum_labels"])
    json.loads(json.dumps(projected))


def test_the_render_kinds_are_the_documented_ones():
    """The guest documents these and the adapter emits them; a typo on either
    side would silently show a person nothing."""
    assert set(KINDS) == {"messages", "attachments", "form_field", "approval",
                          "approval_settled", "buttons", "error", "typing",
                          "tool_status", "stream_delta", "notification"}


# ──────────────────────────────────────────────────────────────────────
# The bridge: a migrated frontend as the kernel sees it.
# ──────────────────────────────────────────────────────────────────────

MIGRATED_FRONTEND = '''
"""A migrated frontend."""

from guest.bases import BaseFrontend


class Web(BaseFrontend):
    """A surface reachable over a socket, near enough."""

    name = "web"
    description = "A test frontend."
    poll_interval = 0.01
    user_binding = "per_user"
    default_user_id = 3
    capabilities = {"supports_buttons": True, "supports_typing": True,
                    "invented_capability": True}

    def start(self, sdk):
        """Open the transport."""
        self._shown = []
        self._polls = 0
        self._stopped = False
        return True

    def poll(self, sdk):
        """One line arrives, then nothing ever again."""
        self._polls += 1
        if self._polls == 1:
            sdk.frontend.submit_text("web:1", "hello")
            return True
        return False

    def stop(self, sdk):
        """Close it."""
        self._stopped = True

    def render(self, sdk, session_key, kind, payload):
        """Record what we were asked to show."""
        self._shown.append({"session_key": session_key, "kind": kind,
                            "payload": payload})

    def session_key(self, sdk, ctx):
        """Name a session from whatever the transport gave us."""
        return "web:" + str((ctx or {}).get("room", "?"))

    def shown(self, sdk):
        """What was rendered. For the tests."""
        return list(self._shown)

    def stopped(self, sdk):
        """Whether the guest saw its stop(). For the tests."""
        return self._stopped
'''


@pytest.fixture
def frontend(tmp_path, box):
    """A migrated frontend as the manager would build it, always stopped."""
    from sandbox.bridge import adapt

    path = tmp_path / "frontend_web.py"
    path.write_text(MIGRATED_FRONTEND, encoding="utf-8")
    module = adapt(path)
    assert module is not None, "the frontend did not adapt"

    made = module.SandboxedWeb()
    yield made
    made.stop()
    unload_box("frontend_web")


def _drain(made, recorded, timeout=3.0):
    """Run the adapter's loop on its own thread until the guest submits."""
    import time

    thread = threading.Thread(target=made.start, daemon=True)
    thread.start()
    deadline = time.time() + timeout
    while time.time() < deadline and not recorded:
        time.sleep(0.01)
    return thread


def test_a_migrated_frontend_looks_native(frontend):
    """Everything downstream must keep seeing an ordinary frontend."""
    from plugins.native.frontend import BaseFrontend, FrontendCapabilities

    assert isinstance(frontend, BaseFrontend)
    assert frontend.name == "web"
    assert frontend.user_binding == "per_user"
    assert frontend.default_user_id == 3
    # Declared as a plain dict — a box cannot hold a dataclass — and rebuilt.
    assert isinstance(frontend.capabilities, FrontendCapabilities)
    assert frontend.capabilities.supports_buttons is True
    assert frontend.capabilities.supports_attachments_in is False
    # A capability this kernel never heard of is dropped, not fatal.
    assert not hasattr(frontend.capabilities, "invented_capability")


def test_the_kernel_drives_the_loop(frontend):
    """The inversion, end to end: the guest never loops, and still receives.

    ``start`` blocks on the manager's daemon thread the way a native frontend
    does, so nothing downstream can tell the difference.
    """
    # The adapter parks *itself*, so what the guest's token reaches is this
    # very object — which is the wiring under test. Recording on the instance
    # intercepts at the same place the native base would have carried on into
    # the state machine.
    recorded = []
    frontend.submit_text = lambda key, text: recorded.append((key, text))

    thread = _drain(frontend, recorded)
    try:
        assert recorded == [("web:1", "hello")]
    finally:
        frontend.stop()
        thread.join(timeout=2.0)


def test_stopping_ends_the_loop_and_the_box(frontend):
    """The loop must not outlive the frontend, and the guest must hear stop."""
    import time

    thread = threading.Thread(target=frontend.start, daemon=True)
    thread.start()
    time.sleep(0.1)                       # let it get into the loop

    frontend.stop()
    thread.join(timeout=2.0)

    assert not thread.is_alive(), "the poll loop outlived stop()"
    assert frontend._sandbox_box is None
    assert frontend._token == ""          # authority revoked with the box


def test_every_render_method_reaches_the_box(frontend):
    """Nine native doorways, one wire call, and the kinds must line up."""
    frontend._sandbox_box = get_sandbox().open(
        frontend._source_path, "Web", name="frontend_web")

    frontend.render_messages("s1", ["hello"])
    frontend.render_typing("s1", True)
    frontend.render_error("s1", {"message": "no"})
    frontend.render_tool_status("s1", {"tool": "search"})

    shown = frontend._sandbox_box.call("shown").data
    assert [entry["kind"] for entry in shown] == [
        "messages", "typing", "error", "tool_status"]
    assert shown[0]["payload"] == ["hello"]
    assert shown[0]["session_key"] == "s1"


def test_an_approval_render_crosses_as_a_question(frontend):
    """The native doorway takes a live object; the box gets a projection."""
    from state_machine.approval import StateMachineApprovalRequest

    frontend._sandbox_box = get_sandbox().open(
        frontend._source_path, "Web", name="frontend_web")

    frontend.render_approval_request("s1", StateMachineApprovalRequest(
        title="Delete?", body="Sure?", pending_action={"tool": "rm"}))

    shown, = frontend._sandbox_box.call("shown").data
    assert shown["kind"] == "approval"
    assert shown["payload"]["title"] == "Delete?"
    assert "pending_action" not in shown["payload"]


def test_rendering_to_a_closed_box_is_survivable(frontend):
    """A frontend that cannot show something must not break the turn.

    The kernel renders from the drive thread, so an exception here would
    surface as a failed agent turn rather than as a blank screen.
    """
    assert frontend._sandbox_box is None
    frontend.render_messages("s1", ["nobody is listening"])   # must not raise


def test_session_key_is_answered_by_the_box(frontend):
    """A transport context is the frontend's own; only its data crosses."""
    frontend._sandbox_box = get_sandbox().open(
        frontend._source_path, "Web", name="frontend_web")

    assert frontend.session_key({"room": 7}) == "web:7"


def test_session_key_falls_back_rather_than_losing_a_message(frontend):
    """With no box there is no answer, and dropping the message is worse."""
    assert frontend._sandbox_box is None
    assert frontend.session_key({"room": 7}) == "default"


# ──────────────────────────────────────────────────────────────────────
# What a frontend is allowed to render to.
# ──────────────────────────────────────────────────────────────────────

def test_a_frontend_renders_only_to_its_own_sessions(frontend):
    """The native default is *every* session the runtime knows about.

    That works natively because each frontend overrides it — the REPL answers
    ``["default"]``. A sandboxed frontend cannot override a native method, so
    inheriting that default would render a Telegram conversation to a
    terminal. Ownership is read from the tag ``_tag_session`` already writes.
    """
    frontend.runtime = SimpleNamespace(sessions={
        "web:1": SimpleNamespace(frontend_name="web"),
        "tg:9": SimpleNamespace(frontend_name="telegram"),
        "fresh": SimpleNamespace(frontend_name=None),
    })

    keys = frontend._live_session_keys()

    assert "web:1" in keys
    assert "tg:9" not in keys, "rendered another frontend's session"
    # Nobody has claimed this one yet, and it may be about to arrive here —
    # dropping it would lose a conversation's first message.
    assert "fresh" in keys


def test_live_sessions_is_empty_before_binding(frontend):
    """No runtime, no sessions. Never an exception on the render path."""
    assert frontend._live_session_keys() == []


def test_pending_approval_is_asked_not_remembered(box, tmp_path):
    """A frontend must be able to find out an approval went away.

    It learns one exists by being handed it to render, but not when it is
    answered elsewhere or times out — and acting on a stale record swallows
    the next thing a person types as a yes or no.
    """
    class Approvals:
        """An adapter with one pending approval, then none."""

        def __init__(self):
            self._pending = True
            self._pending_approval_order = {"s1": ["approve_abc"]}

        def has_pending_approval(self, session_key):
            """Whether anything is waiting."""
            return self._pending

    adapter = Approvals()
    token = park(adapter)
    source = TALKER.replace("ISOLATION", "").replace(
        '    def token(self, sdk):',
        '    def pending(self, sdk):\n'
        '        """Ask what is waiting."""\n'
        '        return sdk.frontend.pending_input("s1")\n\n'
        '    def token(self, sdk):')
    path = tmp_path / "frontend_talker.py"
    path.write_text(source, encoding="utf-8")
    opened = box.open(path, "Talker", name="frontend_talker")
    try:
        assert opened.call("__bind__", token=token).ok
        assert opened.call("pending").data == "approve_abc"

        adapter._pending = False              # answered somewhere else
        assert opened.call("pending").data is None
    finally:
        unpark(token)
        box.close("frontend_talker")
        unload_box("frontend_talker")


# ──────────────────────────────────────────────────────────────────────
# ``frontend.pending {details: true}`` — getting back to a question nobody
# was connected for. A render is an event and events are not re-sent, so a
# frontend that reconnected has no other route to one, and an id alone only
# buys the ability to answer something nobody can read.
# ──────────────────────────────────────────────────────────────────────


class _Waiting:
    """An adapter blocked on whatever it is handed.

    ``registered`` is what separates the two ways a session can be blocked: a
    question this frontend was handed and remembers, or one only the phase
    stack knows about — which is what a restart, or a frontend loaded after its
    session was restored, actually leaves behind.
    """

    def __init__(self, approval=None, step=None, registered=True):
        from state_machine.conversation import PhaseFrame
        from state_machine.conversation_phases import (
            PHASE_APPROVING_REQUEST, PHASE_FILLING_COMMAND_FORM)

        remember = approval is not None and registered
        self._pending_approvals = {"s1": {approval.id: approval}} if remember else {}
        self._pending_approval_order = {"s1": [approval.id]} if remember else {}
        data = {"args": {"name": "requests"}}
        if approval is not None:
            data |= {"request_id": approval.id, "title": approval.title,
                     "prompt": approval.body, "type": approval.type,
                     "enum": approval.enum, "enum_labels": approval.enum_labels}
        frame = PhaseFrame(
            PHASE_APPROVING_REQUEST if approval else PHASE_FILLING_COMMAND_FORM,
            "answer_approval" if approval else "call_command",
            "user", "packages", data, [step] if step else None)
        self.runtime = SimpleNamespace(
            _approval_requests={},
            get_session=lambda _key: SimpleNamespace(
                cs=SimpleNamespace(phase=frame.phase, frame=frame)))

    def has_pending_approval(self, session_key):
        """Whether anything is waiting."""
        return bool(self._pending_approvals.get(session_key))

    def _register_pending_approval(self, session_key, req):
        """Remember one, exactly as ``BaseFrontend`` does."""
        self._pending_approvals.setdefault(session_key, {})[req.id] = req
        order = self._pending_approval_order.setdefault(session_key, [])
        if req.id not in order:
            order.append(req.id)

    def _current_approval_request(self, session_key):
        """Rebuild from the phase frame, as ``BaseFrontend`` does."""
        from state_machine.approval import StateMachineApprovalRequest

        data = self.runtime.get_session(session_key).cs.frame.data
        return StateMachineApprovalRequest(
            title=data["title"], body=data["prompt"], id=data["request_id"],
            type=data["type"], enum=data["enum"],
            enum_labels=data["enum_labels"],
            metadata={"render_result_on_resolve": True})


def _pending(adapter, **args):
    """Call the handler the way a parked frontend reaches it."""
    from sandbox.handlers.kernel import _frontend_pending

    token = park(adapter)
    try:
        return _frontend_pending(SimpleNamespace(),
                                 {"token": token, "session_key": "s1", **args})
    finally:
        unpark(token)


def test_details_hands_back_the_question_not_just_its_id():
    """The same projection the ``approval`` render made, so a client that
    reconnected draws the real dialog rather than a reconstruction of one."""
    from state_machine.approval import StateMachineApprovalRequest

    request = StateMachineApprovalRequest(
        title="Run a shell command", body="rm -rf /tmp/x", type="string",
        enum=["allow", "deny"], enum_labels=["Allow", "Deny"])
    adapter = _Waiting(approval=request)

    assert _pending(adapter).data == request.id          # unchanged without it
    assert _pending(adapter, details=True).data == {
        "kind": "approval", "payload": project_approval(request)}


def test_details_hands_back_a_pending_form_too():
    """A suspended callable's form is the same thing as an approval — a session
    blocked until a person answers — and a client that restores one but not the
    other still strands people."""
    from state_machine.conversation import FormStep

    adapter = _Waiting(step=FormStep("version", prompt="Which version?"))
    answer = _pending(adapter, details=True).data

    assert answer["kind"] == "form_field"
    assert answer["payload"]["name"] == "packages"
    assert answer["payload"]["field"]["name"] == "version"
    assert answer["payload"]["collected"] == {"name": "requests"}
    assert answer["payload"]["display"]["prompt"] == "Which version?"


def test_a_pending_form_stays_invisible_without_details():
    """This Request has only ever spoken about approvals. Widening what it says
    by default would change what an existing frontend believes it is holding —
    the REPL and Telegram both read a falsy answer as "nothing to answer"."""
    from state_machine.conversation import FormStep

    adapter = _Waiting(step=FormStep("version", prompt="Which version?"))

    assert _pending(adapter).data is None


def test_details_answers_none_when_nothing_is_waiting():
    assert _pending(_Waiting(), details=True).data is None


def test_a_question_only_the_phase_stack_remembers_is_still_reported():
    """The registration table is process memory; the phase stack is persisted.

    A kernel restart, or a frontend loaded after its session was restored,
    leaves the table empty while the session is still blocked — and the bus
    announcement that would have filled it fired once, before there was
    anything live to catch it. Answering ``None`` there tells a client to take
    down a dialog for a question that is still waiting, which is the failure
    this Request exists to prevent.
    """
    from state_machine.approval import StateMachineApprovalRequest

    request = StateMachineApprovalRequest(
        title="packages", body="Install a package?", type="boolean")
    adapter = _Waiting(approval=request, registered=False)

    assert adapter.has_pending_approval("s1") is False
    assert _pending(adapter).data == request.id
    assert _pending(adapter, details=True).data["payload"]["id"] == request.id


def test_rebuilding_one_registers_it_so_it_can_be_answered():
    """An id nobody registered is an id ``frontend.resolve`` refuses: it settles
    existence against this same table before it drives anything. Handing back a
    projection without registering it would answer "here is your question" and
    then "no such question" to the very next call."""
    from state_machine.approval import StateMachineApprovalRequest

    request = StateMachineApprovalRequest(
        title="packages", body="Install a package?", type="boolean")
    adapter = _Waiting(approval=request, registered=False)

    _pending(adapter, details=True)

    assert adapter.has_pending_approval("s1")
    assert adapter._pending_approval_order["s1"] == [request.id]


def test_a_live_blocked_request_beats_the_rebuild():
    """A tool blocked inside ``request_input`` is waiting on *that* object. The
    rebuild carries ``render_result_on_resolve``, whose render path waits on the
    guest lock the blocked call holds — answering it would deadlock against the
    call it was answering."""
    from state_machine.approval import StateMachineApprovalRequest

    request = StateMachineApprovalRequest(
        title="packages", body="Install a package?", type="boolean")
    adapter = _Waiting(approval=request, registered=False)
    adapter.runtime._approval_requests[request.id] = request

    _pending(adapter, details=True)

    handed = adapter._pending_approvals["s1"][request.id]
    assert handed is request
    assert not handed.metadata.get("render_result_on_resolve")


def test_a_session_reports_its_phase(box, tmp_path):
    """A frontend needs the phase to know whether the state machine is already
    collecting an answer — if it is, interpreting the line too would consume
    one keystroke twice."""
    from sandbox.handlers.kernel import _session_get

    ctx = SimpleNamespace(runtime=SimpleNamespace(
        sessions={"s1": SimpleNamespace(
            conversation_id=7, busy=True,
            cs=SimpleNamespace(phase="approving_request"))}))

    described = _session_get(ctx, {"key": "s1"}).data

    assert described["phase"] == "approving_request"
    assert described["busy"] is True
    assert described["conversation_id"] == 7


# ──────────────────────────────────────────────────────────────────────
# Re-entrancy: the frontend deadlock.
#
# A resident frontend calls in from ``poll``, which holds its box's single
# call lock. Anything that reaches ``runtime.handle_action`` runs the turn
# synchronously, and a turn renders — straight back into the box that is
# still waiting for this Request to answer. ``submit`` was detached for this
# reason; ``resolve`` and ``cancel`` were not, and answering an approval from
# an inline button froze the transport permanently.
# ──────────────────────────────────────────────────────────────────────

class _ReentrantAdapter:
    """An adapter whose state machine renders back into the caller, as a real
    one does. ``rendered`` only fills in if the Request did not block."""

    background_submit = True
    name = "reentrant"

    def __init__(self):
        self.rendered = []
        self.done = threading.Event()

    def _turn(self, label):
        """What handle_action does: run, and render on the way."""
        self.rendered.append(label)
        self.done.set()
        return SimpleNamespace(ok=True)

    def submit_text(self, session_key, text):
        return self._turn("submit")

    def cancel(self, session_key):
        return self._turn("cancel")

    def is_approval_pending(self, session_key, request_id=None):
        return True

    def resolve_approval(self, session_key, request_id, value):
        return self._turn("resolve")

    def resolve_next_approval(self, session_key, value):
        return self._turn("resolve_next")


@pytest.fixture
def reentrant():
    """A parked re-entrant adapter."""
    made = _ReentrantAdapter()
    made.token = park(made)
    yield made
    unpark(made.token)


@pytest.mark.parametrize("args,expected", [
    ({"input_kind": "text", "text": "hi"}, "submit"),
    ({}, "cancel"),
    ({"value": True}, "resolve_next"),
    ({"value": True, "request_id": "req-1"}, "resolve"),
])
def test_state_machine_driving_requests_leave_the_callers_thread(
        reentrant, args, expected):
    """Each of these drives a turn, so none may run on the caller's thread.

    Asserting on the *thread* rather than on a timeout: a deadlock would hang
    this test rather than fail it, and a test that hangs CI is worse than the
    bug. If the work ran inline it would be on this thread, and it is the
    inline case that cannot survive a render.
    """
    from sandbox.handlers.kernel import (_frontend_cancel, _frontend_resolve,
                                         _frontend_submit)

    handler = {"submit": _frontend_submit, "cancel": _frontend_cancel}.get(
        expected, _frontend_resolve)
    here = threading.get_ident()
    seen = []
    original = reentrant._turn
    reentrant._turn = lambda label: (seen.append(threading.get_ident()),
                                     original(label))[1]

    result = handler(None, {"token": reentrant.token, "session_key": "s1",
                            **args})

    assert result.ok
    assert reentrant.done.wait(5), f"{expected} never ran"
    assert seen and seen[0] != here, (
        f"{expected} ran on the caller's thread; a render would deadlock")
    assert reentrant.rendered == [expected]


def test_resolving_something_already_answered_still_says_so(reentrant):
    """Detaching must not cost the caller the verdict.

    A frontend branches on this to decide whether what someone typed was a
    yes/no or an ordinary message, so an optimistic True would swallow the
    next thing they said.
    """
    from sandbox.handlers.kernel import _frontend_resolve

    reentrant.is_approval_pending = lambda session_key, request_id=None: False

    result = _frontend_resolve(None, {"token": reentrant.token,
                                      "session_key": "s1", "value": True})

    assert result.ok
    assert result.data is False
    assert reentrant.rendered == []
