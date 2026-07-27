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
    assert "only available inside a loaded frontend" in result.error


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
    assert adapter.calls == [("resolve_next", "s1", True)]


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


def test_the_render_kinds_are_the_documented_ones():
    """The guest documents these and the adapter emits them; a typo on either
    side would silently show a person nothing."""
    assert set(KINDS) == {"messages", "attachments", "form_field", "approval",
                          "buttons", "error", "typing", "tool_status",
                          "stream_delta"}


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
    from plugins.BaseFrontend import BaseFrontend, FrontendCapabilities

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
