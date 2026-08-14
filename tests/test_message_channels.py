"""Which channel each population of text travels on, made executable.

``RuntimeResult.messages`` means the conversation and nothing else — the agent's
replies and the person's own words. A refusal is ``error``, an announcement is a
notification, and what a command answered with is ``callable_output``. That is
what lets a client draw a chat transcript from one field and put commands in a
panel of their own.

The rule has been broken three times, and every time the same way: a line that
was *about* a command was appended to ``messages`` because something downstream
happened to read ``messages``. "Back." and "Skipped." arrived in the transcript
whenever somebody used a settings form; "Cancelled." arrived whenever they
pressed its Cancel button; "Loaded conversation: X" was one dispatched action
away from doing the same. Each was found by eye, because nothing failed.

So these tests are structural rather than behavioural, in the style of
``test_kernel_boundary.py``: they pin the *complete set* of places that may write
to ``messages`` at all. A fourth writer fails here, which turns a straggler into
a deliberate one-line decision in this file instead of a line of chat nobody
said.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Where a ``RuntimeResult`` is built or filled in: the runtime and the state
# machine it drives. A frontend building one to render an error of its own
# (``BaseFrontend._unknown_command``) is not in scope — it never touches
# ``messages``, and the sweep below would report it if it did.
SCANNED_DIRS = ("runtime", "state_machine")

# Every site that may put text on ``messages``, as ``(module, function)``.
#
# All three are the agent's own reply at the end of its turn, which is the only
# population that belongs there. Two are fallbacks for a turn that ended without
# final text — still the agent's turn, still the thing the person is waiting to
# read.
#
# Adding a fourth means claiming something new is *conversation*. Before you do:
# an acknowledgement of something the person typed is ``callable_output``, an
# announcement is ``runtime.notifications.notify``, and a refusal is ``error``.
ALLOWED_MESSAGE_WRITERS = {
    ("runtime.conversation_runtime", "_drive_agent_turn"),
    # The one branch of ``add_action_result`` that no ``ActionResult`` currently
    # reaches — kept as the place a future conversational action would land, and
    # pinned by ``test_every_success_one_liner_is_form_navigation`` below so it
    # cannot be reached by accident.
    ("runtime.session", "add_action_result"),
    # Forwarding, not producing: the closing-race follow-up merges a second
    # dispatch's result into the first, and that dispatch's own reply is the
    # agent's. It moves every field alike, ``callable_output`` included.
    ("runtime.conversation_runtime", "handle_action"),
}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _iter_scanned_files():
    for dirname in SCANNED_DIRS:
        for path in sorted((ROOT / dirname).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def _enclosing_function(tree: ast.AST, node: ast.AST) -> str:
    """The nearest def containing ``node``, or ``"<module>"``."""
    best = "<module>"
    for candidate in ast.walk(tree):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(candidate, "end_lineno", None)
        if end is None or not (candidate.lineno <= node.lineno <= end):
            continue
        # Innermost wins: a nested helper is the more specific answer.
        best = candidate.name
    return best


def _messages_writers():
    """Every place ``messages`` is appended to, extended, or passed in.

    Two shapes, because there are two ways to fill the field: ``x.messages.
    append(...)`` / ``.extend(...)``, and ``RuntimeResult(messages=[...])``.

    The keyword form is matched only against a literal ``RuntimeResult(...)``
    call, and deliberately so: ``messages=`` is also how every provider request
    in ``conversation_loop`` is built, and a sweep that counted those would
    report five hits a reader has to learn to ignore — which is how a pinned set
    stops being read at all.
    """
    found = set()
    for path in _iter_scanned_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            appended = (isinstance(node.func, ast.Attribute)
                        and node.func.attr in {"append", "extend"}
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr == "messages")
            constructed = (isinstance(node.func, ast.Name)
                           and node.func.id == "RuntimeResult"
                           and any(kw.arg == "messages" for kw in node.keywords))
            if appended or constructed:
                found.add((_module_name(path), _enclosing_function(tree, node)))
    return found


def test_only_the_agents_own_reply_is_written_to_messages():
    """Pin the complete set. A new writer is a claim, and has to be made here."""
    writers = _messages_writers()

    unexpected = writers - ALLOWED_MESSAGE_WRITERS
    assert not unexpected, (
        "New writer(s) to RuntimeResult.messages: "
        f"{sorted(unexpected)}. `messages` is the conversation and nothing "
        "else. An answer to something the person typed goes on "
        "`callable_output`, an announcement through `runtime.notifications."
        "notify`, a refusal on `error`. If this really is the agent speaking, "
        "add it to ALLOWED_MESSAGE_WRITERS with a reason."
    )
    # The other direction too: a site that goes away should be removed here
    # rather than left as a stale claim about code that no longer exists.
    assert not ALLOWED_MESSAGE_WRITERS - writers, (
        "ALLOWED_MESSAGE_WRITERS names a site that no longer writes to "
        "messages — delete the stale entry."
    )


# ── The state machine's own one-liners ───────────────────────────────

def test_every_success_one_liner_is_form_navigation():
    """A successful action's ``message`` is always a form acknowledging itself.

    ``Cancel``, ``SkipForm`` and ``BackForm`` are the only actions that answer a
    success with prose, and all three are the same kind of thing: a form or an
    approval reporting on its own navigation. ``add_action_result`` reads
    ``FORM_NAVIGATION`` to route them, so an unmarked one silently becomes chat.

    Asserted over the source rather than by driving each action, because the
    point is that the set is *complete* — a fourth action added tomorrow is
    exactly the case that would slip through a test naming three.
    """
    tree = ast.parse((ROOT / "state_machine" / "action.py")
                     .read_text(encoding="utf-8"))
    unmarked = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ActionResult"):
            continue
        args = node.args
        # ActionResult(ok, action, message, ...) — positional, and a success
        # with no third argument has nothing to route.
        if len(args) < 3 or not isinstance(args[0], ast.Constant) or args[0].value is not True:
            continue
        if isinstance(args[2], ast.Constant) and args[2].value is None:
            continue
        marked = any(
            kw.arg == "data" and isinstance(kw.value, ast.Dict) and any(
                isinstance(key, ast.Name) and key.id == "FORM_NAVIGATION"
                for key in kw.value.keys)
            for kw in node.keywords)
        if not marked:
            unmarked.append(node.lineno)

    assert not unmarked, (
        f"ActionResult success message(s) at state_machine/action.py:{unmarked} "
        "carry no FORM_NAVIGATION mark, so they will be rendered as "
        "conversation. If the line really is something said in the "
        "conversation, say so here; otherwise mark it."
    )


# ── The two Requests that hand a RuntimeResult to a guest ────────────

def test_a_request_answer_carries_both_text_channels():
    """``conv.load`` and ``session.cancel`` hand over every channel.

    This is what decouples where the kernel puts a line from what a command can
    read back. Both used to build their text on ``messages`` *because* the
    commands reading them read ``messages`` — so a confirmation no client should
    have seen was pinned to the chat kind to keep one command working.
    """
    from sandbox.handlers.kernel import _runtime_answer

    answer = _runtime_answer(SimpleNamespace(
        ok=True, messages=["said"], callable_output=["answered"],
        error=None, data={"conversation_id": 7}))

    assert answer["messages"] == ["said"]
    assert answer["callable_output"] == ["answered"]
    assert answer["data"]["conversation_id"] == 7


def _cancel_command():
    """``/cancel``, loaded from source without going through discovery."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "command_cancel_under_test",
        ROOT / "bundled" / "commands" / "command_cancel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CancelCommand()


def _run_cancel(answer):
    return _cancel_command().run(SimpleNamespace(session=SimpleNamespace(
        get=lambda: {"key": "s"}, cancel=lambda: answer)), {})


@pytest.mark.parametrize("field", ["callable_output", "messages"])
def test_cancel_command_reads_a_dismissed_form_off_the_text_channel(field):
    """Dismissing a *form* answers with text, and that text is the output.

    Both channels, because the command reads ``callable_output`` first and falls
    back — and a fallback nothing exercises is a fallback that does not work.
    """
    assert _run_cancel({"ok": True, field: ["Cancelled."]}) == "Cancelled."


@pytest.mark.parametrize("outcome,expected", [
    ({"cancelled": True, "subagents_stopped": False}, "Cancelled."),
    ({"cancelled": True, "subagents_stopped": True},
     "Cancelled. Subagents stopped."),
    ({"cancelled": False}, "Nothing to cancel."),
])
def test_cancel_command_words_a_stopped_turn_from_state(outcome, expected):
    """Stopping a *turn* answers with state, because the kernel notifies.

    The command still has to say something: it was invoked by name, and one
    that answered with nothing would read as having silently failed. So the
    wording lives here, and the kernel's notification is what the Cancel
    *button* — which invokes no callable at all — delivers instead.
    """
    assert _run_cancel({"ok": True, "data": outcome}) == expected


# ── A failure says what failed ───────────────────────────────────────

def test_a_failed_command_names_itself():
    """``error`` carries the act, not only the complaint.

    ``ActionError.to_dict`` answers ``{code, message, details, retry_phase}``,
    which describes the error and not what was being attempted — so a failing
    ``/packages install``, an unrecognised slash command and "Still working."
    arrived as one shape and a client with a command panel had nothing to route
    the first back to.
    """
    from state_machine.conversation import (CallableSpec, ConversationState,
                                            Participant)
    from runtime.session import RuntimeResult

    def explode(cs, actor, args):
        raise RuntimeError("the disk is on fire")

    cs = ConversationState([Participant("user", "user", commands={
        "backups": CallableSpec("backups", handler=explode)})])

    out = RuntimeResult().add_action_result(
        cs.enact("call_command", {"name": "backups", "args": {}}, "user"))

    assert out.error["action"] == "call_command"
    assert out.error["name"] == "backups"
    assert "the disk is on fire" in out.error["message"]


def test_an_unknown_command_names_itself_too():
    """The failure worth naming most is the one with no spec behind it."""
    from state_machine.conversation import ConversationState, Participant
    from runtime.session import RuntimeResult

    cs = ConversationState([Participant("user", "user", commands={})])

    out = RuntimeResult().add_action_result(
        cs.enact("call_command", {"name": "nope", "args": {}}, "user"))

    assert out.error["action"] == "call_command"
    assert out.error["name"] == "nope"


def test_a_failure_still_names_the_phase_to_retry_from():
    """``retry_phase`` survives naming the callable.

    The tempting way to carry the name is to build the ``ActionError`` where the
    body raised — inside ``_CallableAction._run``, where ``spec`` is in scope.
    But ``self.error`` anchors ``retry_phase`` to ``cs.phase``, and there that
    is still ``calling_command``: a phase nobody can retry from. ``enact``
    builds it after the ``finally`` has reset the phase, and asks the action for
    the name instead.
    """
    from state_machine.conversation import (CallableSpec, ConversationState,
                                            Participant)

    def explode(cs, actor, args):
        raise RuntimeError("boom")

    cs = ConversationState([Participant("user", "user", commands={
        "x": CallableSpec("x", handler=explode)})])
    base = cs.phase

    result = cs.enact("call_command", {"name": "x", "args": {}}, "user")

    assert result.error.retry_phase == base


# ── Narrating progress ───────────────────────────────────────────────

def test_progress_abstains_when_no_command_is_running():
    """Silence, not a fallback to the chat.

    The whole reason ``ui.progress`` exists is that ``session.push``'s
    destination is the chat. A fallback to it when there is no call to address
    would reintroduce exactly the bug — and would do it on the paths nobody is
    watching, which is where it would be least noticed.
    """
    from sandbox.handlers.kernel import _ui_progress

    ctx = SimpleNamespace(runtime=None, session_key=None)

    assert _ui_progress(ctx, {"message": "half way"}).data is False


def test_progress_addresses_the_running_commands_own_call():
    from events.event_bus import bus
    from events.event_channels import COMMAND_CALL_PROGRESSED
    from sandbox.handlers.kernel import _ui_progress

    seen = []
    unsub = bus.subscribe(COMMAND_CALL_PROGRESSED, seen.append)
    try:
        session = SimpleNamespace(cs=SimpleNamespace(cache={
            "_running_command": {"call_id": "cc_1", "name": "tasks"}}))
        ctx = SimpleNamespace(
            runtime=SimpleNamespace(sessions={"s": session}), session_key="s")

        assert _ui_progress(ctx, {"message": "Resetting 12,000 rows"}).data is True
    finally:
        unsub()

    assert seen[-1]["call_id"] == "cc_1"
    assert seen[-1]["command_name"] == "tasks"
    assert seen[-1]["narration"] == "Resetting 12,000 rows"


def test_progress_is_neither_counted_nor_recorded():
    """A loop narrating itself must not invalidate prompts or flood the ledger.

    Same pair of exemptions ``llm.delta`` and ``http.push`` hold, for the same
    reason: the volume is per-iteration, and what it produces is a line on a
    screen rather than state any prompt could read back.
    """
    from sandbox import epoch
    from sandbox.guest.requests import UI_PROGRESS

    assert UI_PROGRESS in epoch.RENDERING
    assert UI_PROGRESS in epoch.UNCOUNTED
