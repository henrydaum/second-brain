"""Where a package install says what it is doing.

Installing a package takes long enough to be worth narrating, and the narration
came out of a kernel handler, which knows a session key and nothing else. The
only channel reachable from there was ``push_message`` — the *conversation*
channel — so "Copying package files" arrived as a ``messages`` frame and landed
in the transcript, even when the person had run ``/packages`` from a settings
screen and was watching a command panel. It could not even stay: a push writes
no history row, so the lines disappeared on the next reload.

The fix addresses the narration to the call it belongs to. Two halves, both
here: the state machine has to *record* which command is running so a handler
can name it, and the handler has to emit on ``COMMAND_CALL_PROGRESSED`` instead
of pushing chat.
"""

import pytest

import state_machine  # noqa: F401  (break the runtime import cycle)

from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED, COMMAND_CALL_PROGRESSED
from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime
from sandbox.handlers.kernel import _command_progress
from state_machine.conversation import CallableSpec


class _CS:
    def __init__(self, cache=None):
        self.cache = cache if cache is not None else {}


class _Session:
    def __init__(self, cache=None):
        self.cs = _CS(cache)


class _Runtime:
    """Only what ``_command_progress`` reads, plus a record of what it pushed."""

    def __init__(self, sessions):
        self.sessions = sessions
        self.pushed = []

    def push_message(self, key, text, **kw):
        self.pushed.append((key, text, kw))


class _Ctx:
    def __init__(self, runtime, session_key="frontend:http:1"):
        self.runtime = runtime
        self.session_key = session_key


@pytest.fixture
def emitted():
    """Every progress and chat-push event raised while the test runs."""
    seen = {"progress": [], "chat": []}
    unsubs = [bus.subscribe(COMMAND_CALL_PROGRESSED, seen["progress"].append),
              bus.subscribe(CHAT_MESSAGE_PUSHED, seen["chat"].append)]
    yield seen
    for unsub in unsubs:
        unsub()


def _running(call_id="cmd:packages:abcd1234", name="packages"):
    return _Ctx(_Runtime({"frontend:http:1": _Session(
        {"_running_command": {"call_id": call_id, "name": name}})}))


def test_progress_addresses_the_running_command(emitted):
    """The panel that asked for the install is told how it is going."""
    ctx = _running()
    progress = _command_progress(ctx)
    assert progress is not None

    progress("Copying package files")

    assert emitted["progress"] == [{
        "session_key": "frontend:http:1",
        "call_id": "cmd:packages:abcd1234",
        "command_name": "packages",
        "narration": "Copying package files",
    }]


def test_progress_never_reaches_the_conversation(emitted):
    """The regression itself. Nothing here belongs in the transcript."""
    ctx = _running()
    _command_progress(ctx)("Running package setup")

    assert emitted["chat"] == []
    assert ctx.runtime.pushed == []


def test_a_progress_emit_claims_nothing_about_collected_answers(emitted):
    """``args`` is the other producer's field, and stating it empty here would
    blank the answers the panel is still showing."""
    _command_progress(_running())("Resolving dependency plan")

    assert "args" not in emitted["progress"][0]


def test_no_running_command_narrates_nowhere(emitted):
    """A package action outside a slash command — an agent calling the tool, a
    task — has no call to address. It says nothing rather than falling back to
    the chat, which is the behaviour this replaced."""
    ctx = _Ctx(_Runtime({"frontend:http:1": _Session({})}))

    assert _command_progress(ctx) is None
    assert emitted["chat"] == []


def test_an_unknown_session_narrates_nowhere():
    assert _command_progress(_Ctx(_Runtime({}))) is None


def test_no_session_key_narrates_nowhere():
    ctx = _Ctx(_Runtime({}), session_key=None)

    assert _command_progress(ctx) is None


# ── The other half: the state machine has to say which command is running ──


def test_a_running_command_names_itself_and_stops_afterwards(tmp_path):
    """Visible from inside the body, gone once it returns.

    The handler reaching for this runs *during* the call, so the identity has
    to be on ``cs`` for exactly that window — long enough to address, and not
    a moment after, or the next thing with progress to report would narrate it
    onto a command that had already finished.
    """
    db = Database(str(tmp_path / "progress.db"))
    runtime = ConversationRuntime(db=db, services={}, config={})
    session = runtime.get_session("repl")
    session.conversation_id = db.create_conversation("Notes", user_id=1)
    seen: list = []

    def handler(cs, _actor, _args):
        seen.append(dict(cs.cache.get("_running_command") or {}))
        return "done"

    runtime.commands = {"packages": CallableSpec("packages", handler)}
    runtime.handle_action("repl", "call_command",
                          {"name": "packages", "args": {}})

    assert seen and seen[0]["name"] == "packages"
    assert seen[0]["call_id"].startswith("cmd:packages:")
    assert "_running_command" not in session.cs.cache
