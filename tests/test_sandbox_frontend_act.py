"""``frontend.act`` — a frontend acting as one of its own sessions.

A frontend box is rooted ``frontend:<name>``, which names no session, so
``attended_now`` answers False for it forever: every unsafe Request it makes is
refused rather than asked, because there is nobody a dialog could be drawn for.
That is right for a frontend acting on its own initiative and wrong for one
serving a request somebody just made, which is the whole reason a client could
only ever *read* through an HTTP frontend.

``act`` says so, by rooting one Request at a session the frontend owns. Three
properties are what these tests are for, and each was a real failure before it
was a rule:

* **It does not wait.** A box serves one call at a time, and an approval
  renders back *into the calling box* to be seen. Inline, that is a deadlock
  until the dialog expires — the same shape ``handlers.kernel._drive`` exists
  to prevent.
* **The authority is the session's, not the type's.** Rooting at the session
  makes attendance decide, and attendance is what this same frontend declared.
  Say nobody is watching and the authority goes away.
* **Ownership is host-side.** The token says which frontend is asking; the
  runtime's session tags say which sessions it may speak about. Both halves,
  or one frontend answers for another's user.
"""

import threading
import time
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from guest.loader import unload_box
from sandbox import Sandbox
from sandbox.frontends import park, unpark
from sandbox.guest.requests import Request
from sandbox.policy import Chain, attended_now, classify

# A frontend that does nothing but reach for ``act``. Opened as a plain box
# rather than through the bridge, so the Request path is what is under test.
ACTOR = '''
"""A frontend that acts as its sessions."""

from guest.bases import BaseFrontend


class Actor(BaseFrontend):
    """Runs Requests on behalf of whoever asked over its transport."""

    name = "actor"

    def start(self, sdk):
        """Nothing to open."""
        return True

    def poll(self, sdk):
        """Nothing arrives on its own here."""
        return False

    def kick(self, sdk, session_key="s1", request_type="fs.list", args=None):
        """Start one Request and return its handle, holding nothing."""
        return sdk.frontend.act(session_key, request_type, args or {})

    def take(self, sdk, handle=""):
        """Collect it, whenever the answer turns up."""
        return sdk.frontend.collect(handle)
'''


class FakeSession:
    """One live session, tagged with the frontend that claimed it."""

    def __init__(self, frontend_name=None, attended=None):
        self.frontend_name = frontend_name
        self.attended = attended
        self.user_id = 1
        self.conversation_id = None


class FakeRuntime:
    """Just enough runtime for ownership and attendance to be answerable."""

    def __init__(self, sessions=None):
        self.sessions = sessions or {}
        self.active_session_key = "somebody-else"

    def is_attended(self, session_key):
        """The real rule: the frontend's opinion wins, else the active one."""
        session = self.sessions.get(session_key)
        if session is not None and session.attended is not None:
            return session.attended
        return session_key == self.active_session_key


class FakeAdapter:
    """Stands in for the native adapter a token resolves to."""

    def __init__(self, name="actor", runtime=None):
        self.name = name
        self.runtime = runtime if runtime is not None else FakeRuntime()
        self.attended_calls = []

    def mark_attended(self, session_key):
        """Record the declaration."""
        self.attended_calls.append((session_key, True))

    def mark_unattended(self, session_key):
        """Record the declaration."""
        self.attended_calls.append((session_key, False))

    def has_pending_approval(self, session_key):
        """Nothing is waiting here; the point is that the call resolved."""
        return False


@pytest.fixture
def runtime():
    """A runtime holding one session this frontend owns."""
    return FakeRuntime({"s1": FakeSession("actor", attended=True)})


@pytest.fixture
def adapter(runtime):
    """A parked adapter whose desk is always cleared."""
    made = FakeAdapter(runtime=runtime)
    made.token = park(made)
    yield made
    unpark(made.token)


@pytest.fixture
def box(runtime):
    """A sandbox that can answer Requests from a session's world.

    Installed as *the* sandbox for the duration, because the handler reaches it
    through ``bridge.get_sandbox``. That is deliberate rather than convenient:
    one process lends one sandbox, the same way it lends one HTTP port, and a
    detached Request has to land in the interpreter that is already answering
    for this frontend — with its approver, its context factory and its gate.
    """
    from sandbox.bridge import _SANDBOX, configure

    made = Sandbox()
    made.bind_context(lambda session_key=None: SimpleNamespace(
        session_key=session_key, runtime=runtime, config={}, db=None,
        user_id=1))
    previous = _SANDBOX
    configure(made)
    try:
        yield made
    finally:
        configure(previous)
        made.shutdown()


@pytest.fixture
def actor(box, tmp_path, adapter):
    """An opened, bound frontend box."""
    path = tmp_path / "frontend_actor.py"
    path.write_text(ACTOR, encoding="utf-8")
    opened = box.open(path, "Actor", name="frontend_actor")
    assert opened.call("__bind__", token=adapter.token).ok
    yield opened
    box.close("frontend_actor")
    unload_box("frontend_actor")


def _collect(actor, handle, tries=200):
    """Poll for an answer the way a frontend's own loop would."""
    for _ in range(tries):
        outcome = actor.call("take", handle=handle)
        if outcome.ok and outcome.data is not None:
            return outcome.data
        time.sleep(0.01)
    raise AssertionError(f"handle {handle!r} never produced an answer")


# ──────────────────────────────────────────────────────────────────────
# It does not wait. This is the point of the whole design.
# ──────────────────────────────────────────────────────────────────────

#: How long the stand-in dialog stays open. Long enough that an ``act`` which
#: waited for it could not possibly be mistaken for one that did not.
DIALOG_HOLD = 10.0


def test_act_answers_while_the_dialog_is_still_open(box, actor, adapter):
    """The handle comes back before anybody has approved anything.

    Timed rather than merely asserted, because the inline version *also*
    returns a handle eventually — it just returns it after the dialog closes,
    which is the bug. The clock is the only thing that tells the two apart.
    """
    asked = threading.Event()
    release = threading.Event()

    def approve(chain, request, decision):
        """Hold the dialog open, as a real one waits on a person."""
        asked.set()
        release.wait(DIALOG_HOLD)
        return True

    box.interpreter.set_approver(approve)
    try:
        started = time.monotonic()
        handle = actor.call("kick", request_type="session.add_tool",
                            args={"key": "s1", "tool": "t"})
        elapsed = time.monotonic() - started

        assert handle.ok and handle.data, "act must answer with a handle"
        assert asked.wait(5), "the Request never reached the approver"
        assert elapsed < DIALOG_HOLD / 2, (
            f"act waited {elapsed:.1f}s for the dialog — it must not wait")
    finally:
        release.set()


def test_the_box_is_free_while_an_act_waits_for_approval(box, actor, adapter):
    """The deadlock itself, reproduced from the side that used to hang.

    A dialog is drawn by calling ``render`` on the frontend that is asking. So
    the approver here does what a real one does — it calls back into the very
    box whose guest started this Request. Inline, that box is still held by the
    caller and the call cannot get in until the acquire bound gives up; the
    dialog is never drawn, and five minutes later it is denied by timeout.
    """
    reached = []

    def approve(chain, request, decision):
        """Render into the asking frontend, exactly as a dialog does."""
        reached.append(actor.call("take", handle="nothing"))
        return True

    box.interpreter.set_approver(approve)
    handle = actor.call("kick", request_type="session.add_tool",
                        args={"key": "s1", "tool": "t"}).data
    _collect(actor, handle)

    assert reached, "the approver never ran"
    assert reached[0].ok, (
        "the dialog could not reach the box that was asking: "
        f"{reached[0].error}")


def test_a_refusal_comes_back_as_an_answer_not_a_failure(box, actor, adapter):
    """A frontend forwards refusals; it does not treat them as its own fault."""
    box.interpreter.set_approver(lambda chain, request, decision: False)

    handle = actor.call("kick", request_type="session.add_tool",
                        args={"key": "s1", "tool": "t"}).data
    answer = _collect(actor, handle)

    assert answer["ok"] is False
    assert answer["code"] == "approval_declined"


# ──────────────────────────────────────────────────────────────────────
# The authority is the session's.
# ──────────────────────────────────────────────────────────────────────

def test_act_roots_the_chain_at_the_session(box, actor, adapter):
    """Which is what makes attendance the thing that decides."""
    seen = []

    def approve(chain, request, decision):
        """Remember who was asking."""
        seen.append(chain)
        return True

    box.interpreter.set_approver(approve)
    handle = actor.call("kick", request_type="session.add_tool",
                        args={"key": "s1", "tool": "t"}).data
    _collect(actor, handle)

    assert seen, "the Request never reached the approver"
    chain = seen[0]
    assert chain.root == "s1"
    # And it still says which frontend, so the ledger can tell this apart from
    # an agent tool call in the same session.
    assert "frontend:actor" in chain.render()


def test_attendance_is_what_the_frontend_declared(runtime):
    """The self-limiting property, stated against the reader that enforces it.

    A frontend's own root is unattended forever. Rooted at a session it owns,
    the answer becomes whatever it said through ``frontend.attend`` — so
    marking the session unattended takes the authority back.
    """
    own = Chain(root="frontend:actor")
    assert not attended_now(own, runtime=runtime)

    session = Chain(root="s1").push("frontend:actor")
    assert attended_now(session, runtime=runtime)

    runtime.sessions["s1"].attended = False
    assert not attended_now(session, runtime=runtime)


def test_classification_is_untouched_by_act():
    """``act`` moves who is asking. It does not make anything safe.

    Worth pinning as a negative: the failure to avoid is a future refactor
    deciding that a Request a frontend asked for is somehow pre-approved.
    """
    chain = Chain(root="s1").push("frontend:actor")
    assert classify(Request("fs.list", {"path": "."}), chain).safe
    assert not classify(Request("session.add_tool",
                                {"key": "s1", "tool": "t"}), chain).safe


# ──────────────────────────────────────────────────────────────────────
# Ownership: the token, and the sessions it may speak about.
# ──────────────────────────────────────────────────────────────────────

def test_act_refuses_another_frontends_session(actor, runtime):
    """The token proves who is asking, not what it may ask about."""
    runtime.sessions["theirs"] = FakeSession("telegram", attended=True)

    result = actor.call("kick", session_key="theirs",
                        request_type="fs.list", args={"path": "."})

    assert not result.ok
    assert "telegram" in result.error


def test_act_allows_a_session_nobody_has_claimed_yet(actor, runtime):
    """A brand-new thread has no owner, and must not be refused for it.

    ``_tag_session`` stamps ownership when a frontend first submits, so the
    very first request of a conversation is always for an untagged session.
    Refusing it would mean a frontend could never start one — the same reason
    ``_live_session_keys`` includes untagged sessions.
    """
    assert "fresh" not in runtime.sessions

    result = actor.call("kick", session_key="fresh",
                        request_type="fs.list", args={"path": "."})

    assert result.ok and result.data


def test_attend_refuses_another_frontends_session(adapter, runtime):
    """The gap this closed: attendance is what decides whether a dialog runs.

    ``mark_attended`` takes any string, so before this check one frontend could
    declare another's session attended — and thereby arrange for somebody
    else's user to be asked to approve things.
    """
    from sandbox.handlers.kernel import _frontend_attend

    runtime.sessions["theirs"] = FakeSession("telegram")
    refused = _frontend_attend(None, {"token": adapter.token,
                                      "session_key": "theirs",
                                      "present": True})

    assert not refused.ok
    assert adapter.attended_calls == []


def test_a_request_without_a_token_reaches_no_adapter(box, tmp_path):
    """What a tool or a script that imported the SDK holds: nothing."""
    from sandbox.handlers.kernel import _frontend_act

    refused = _frontend_act(None, {"token": "", "session_key": "s1",
                                   "request_type": "fs.list"})

    assert not refused.ok
    assert "sdk.frontend" in refused.error


# ──────────────────────────────────────────────────────────────────────
# What act will not carry.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["frontend.act", "frontend.collect",
                                  "http.drain", "http.respond", "http.push",
                                  "http.close"])
def test_act_refuses_the_transport_and_itself(actor, kind):
    """Recursion, and the socket the client is talking over.

    The ``http`` family belongs to the frontend's own transport rather than to
    anything a session may reach — a client closing the stream it is being
    served on is the shape to keep impossible.
    """
    result = actor.call("kick", request_type=kind, args={})

    assert not result.ok
    assert kind in result.error


def test_act_refuses_a_type_that_does_not_exist(actor):
    """Named rather than silently dropped: the client made a typo."""
    result = actor.call("kick", request_type="conv.summon", args={})

    assert not result.ok
    assert "conv.summon" in result.error


def test_the_kernel_supplies_the_token_for_inner_frontend_requests(actor,
                                                                   adapter):
    """A client must never be able to say who it is.

    ``frontend.*`` Requests resolve an adapter by token. One arriving in the
    args would be somebody else's claim about their own identity, so the
    kernel overwrites it with the token that got the call this far.
    """
    handle = actor.call("kick", request_type="frontend.pending",
                        args={"session_key": "s1",
                              "token": "stolen"}).data
    answer = _collect(actor, handle)

    # It resolved an adapter at all, which a bogus token could not have done.
    assert answer["ok"], answer


# ──────────────────────────────────────────────────────────────────────
# Collecting.
# ──────────────────────────────────────────────────────────────────────

def test_collect_delivers_once(box, actor, adapter):
    """One-shot, like a subagent report: two takers is worse than none."""
    handle = actor.call("kick", args={"path": "."}).data
    first = _collect(actor, handle)

    assert first["ok"]
    assert actor.call("take", handle=handle).data is None


def test_another_frontend_cannot_collect_a_handle(box, actor, adapter,
                                                  runtime):
    """Answering None rather than refusing: naming somebody else's handle is a
    mistake about *which* one, and saying so should not disclose it exists."""
    handle = actor.call("kick", args={"path": "."}).data
    _collect(actor, handle)     # make sure it has finished

    stranger = FakeAdapter(name="telegram", runtime=runtime)
    token = park(stranger)
    try:
        from sandbox.handlers.kernel import _frontend_collect
        answer = _frontend_collect(None, {"token": token, "handle": handle})
    finally:
        unpark(token)

    assert answer.ok and answer.data is None


def test_collect_answers_none_while_it_is_still_running(box, actor, adapter):
    """Which is what lets a frontend poll rather than block."""
    release = threading.Event()

    def approve(chain, request, decision):
        """Never come back within the test's patience."""
        release.wait(5)
        return False

    box.interpreter.set_approver(approve)
    try:
        handle = actor.call("kick", request_type="session.add_tool",
                            args={"key": "s1", "tool": "t"}).data
        assert actor.call("take", handle=handle).data is None
    finally:
        release.set()
