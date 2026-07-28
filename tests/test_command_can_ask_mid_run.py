"""A command that asks the user mid-run can be answered.

``handle_action`` holds the session lock across ``_dispatch``, and a slash
command's body runs inside that dispatch. So a command that blocked waiting
for the user was waiting for an answer that could only arrive as a *second*
action on another thread — which needed the very lock the command was sitting
on. ``/packages install`` deadlocked on exactly that, and it took the sandbox's
gate thread down with it, which is why the REPL went silent rather than merely
slow.

The fix is the one the agent turn already had: the plugin body runs outside the
lock (``RuntimeSession.unlocked``, installed on the state machine as
``cs.unlocked``) while every state-machine mutation around it keeps it. The
busy guard, not the lock, is what keeps a second action out meanwhile.

Commands should still ask *up front* wherever they can — see
``test_command_approval_declarations`` — but the mid-run path has to be
survivable, because a store command will reach it eventually.
"""

import threading

import state_machine  # noqa: F401  (break the runtime import cycle)

from pipeline.database import Database
from runtime.conversation_runtime import ConversationRuntime
from state_machine.conversation import CallableSpec


def _runtime(tmp_path):
    db = Database(str(tmp_path / "ask.db"))
    cid = db.create_conversation("Notes", user_id=1)
    runtime = ConversationRuntime(db=db, services={}, config={})
    session = runtime.get_session("repl")
    session.conversation_id = cid
    return runtime, session


def test_a_command_that_asks_mid_run_gets_an_answer(tmp_path):
    runtime, session = _runtime(tmp_path)
    asked = threading.Event()
    answer: list = []

    def handler(cs, _actor, _args):
        """A command body that stops and asks, the way an install does."""
        pending = runtime.request_input(
            "repl", "Install?", "This changes what the system can do.")
        asked.set()
        assert pending.wait(timeout=10), "nobody ever answered"
        answer.append(pending.approved)
        return "installed"

    # On ``runtime.commands``, not on the participant: ``handle_action``
    # calls ``refresh_specs`` first, which re-seeds the participant from here.
    runtime.commands = {"install": CallableSpec("install", handler)}

    result: list = []
    caller = threading.Thread(
        target=lambda: result.append(runtime.handle_action(
            "repl", "call_command", {"name": "install", "args": {}})),
        daemon=True)
    caller.start()

    assert asked.wait(5), "the command never reached its question"

    # The answer arrives the only way it can: another action, another thread.
    # Before the fix this blocked forever on the session lock.
    answering = threading.Thread(
        target=lambda: runtime.handle_action(
            "repl", "answer_approval", {"value": True}),
        daemon=True)
    answering.start()
    answering.join(timeout=5)
    caller.join(timeout=5)

    assert not caller.is_alive(), "the command never finished"
    assert answer == [True]
    assert result and result[0].ok


def test_the_lock_is_held_again_once_the_body_returns(tmp_path):
    """Parking is temporary, and the RLock's depth has to come back.

    Releasing an RLock once when it was held twice would leave it held and fix
    nothing; releasing it once when it was held once and forgetting to
    re-acquire would leave the dispatcher unlocked for the rest of its work.
    """
    runtime, session = _runtime(tmp_path)
    inside: list = []

    def handler(cs, _actor, _args):
        # ``acquire(blocking=False)`` from this same thread would always
        # succeed on a reentrant lock, so ask a *different* thread whether it
        # can take it; that is the question that matters.
        got: list = []

        def probe_lock():
            """Take it and give it straight back, on one thread throughout.

            An RLock may only be released by the thread that holds it, so the
            acquire and the release have to live together in here.
            """
            taken = session.lock.acquire(timeout=2)
            got.append(taken)
            if taken:
                session.lock.release()

        probe = threading.Thread(target=probe_lock, daemon=True)
        probe.start()
        probe.join(timeout=3)
        inside.append(bool(got and got[0]))
        return "done"

    runtime.commands = {"peek": CallableSpec("peek", handler)}

    result = runtime.handle_action(
        "repl", "call_command", {"name": "peek", "args": {}})

    assert result.ok
    assert inside == [True], "the body still ran under the session lock"
    # And the dispatcher has it back: a fresh action still works.
    assert runtime.handle_action(
        "repl", "call_command", {"name": "peek", "args": {}}).ok
