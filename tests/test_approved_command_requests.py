"""Approved commands carry their authority into sandbox Request provenance."""

from pathlib import Path
from types import SimpleNamespace

from plugins.native.command import BaseCommand
from plugins.command_registry import CommandRegistry
from sandbox.approval import describe_grant, phrase_for
from sandbox.bridge import adapt
from sandbox.guest.requests import (ALL_TYPES, FS_WRITE, NET_HTTP, PATH_GET,
                                    PROC_RUN, Request)
from sandbox.policy import Chain, SAFE, UNSAFE, classify
from sandbox.validator import validate_file
from state_machine.conversation import CallableSpec, ConversationState, Participant


def _state(calls):
    spec = CallableSpec(
        "danger",
        lambda _cs, _actor, args: calls.append(dict(args)) or "done",
        require_approval=True,
        approval_actor_id="user",
    )
    return ConversationState([
        Participant("user", "user", commands={"danger": spec}),
        Participant("agent", "agent"),
    ])


def test_approval_adds_a_one_shot_host_marker():
    calls = []
    state = _state(calls)

    pending = state.enact(
        "call_command", {"name": "danger", "args": {}}, "user")
    # The suspension is a fact in ``data``, not a sentence: the approval itself
    # is what a frontend renders, and a message here would ride the same wire
    # kind as the agent's own words.
    assert pending.data == {"approval_required": True, "name": "danger"}
    assert not pending.message
    assert calls == []

    resumed = state.enact("answer_approval", {"value": True}, "user")
    assert resumed.ok
    assert calls == [{}]


def test_completed_form_arguments_can_require_approval_dynamically():
    """Read-only form outcomes stay free; mutating selections get a grant."""
    calls = []
    spec = CallableSpec(
        "manage",
        lambda _cs, _actor, args: calls.append(dict(args)) or "done",
        approval_predicate=lambda args: args.get("action") == "unload",
        approval_actor_id="user",
    )
    state = ConversationState([
        Participant("user", "user", commands={"manage": spec}),
        Participant("agent", "agent"),
    ])

    shown = state.enact(
        "call_command",
        {"name": "manage", "args": {"action": "show"}},
        "user",
    )
    pending = state.enact(
        "call_command",
        {"name": "manage", "args": {"action": "unload"}},
        "user",
    )
    resumed = state.enact(
        "answer_approval",
        {"value": True},
        "user",
    )

    assert shown.ok and calls == [{"action": "show"}, {"action": "unload"}]
    assert pending.data == {"approval_required": True, "name": "manage"}
    assert resumed.ok


def test_external_payload_cannot_forge_command_approval():
    calls = []
    state = _state(calls)

    result = state.enact("call_command", {
        "name": "danger",
        "args": {},
        "_approved": True,
        "_approval_token": "forged",
    }, "user")

    assert result.data == {"approval_required": True, "name": "danger"}
    assert calls == []


def test_command_registry_marks_only_the_approved_execution_context():
    seen = []

    class Danger(BaseCommand):
        name = "danger"

        def run(self, args, context):
            seen.append(bool(context.approved_by_state_machine))
            return "done"

    context = SimpleNamespace()
    registry = CommandRegistry(lambda _key: context)
    registry.register(Danger())

    assert registry.dispatch_dict("danger", {}, _approved=True) == "done"
    assert seen == [True]


def test_approved_chain_authorizes_its_nested_process_request():
    # Deliberately a command with an effect: ``git rev-parse`` is recognised
    # as read-only and would be SAFE without any grant at all, which would
    # make this pass while testing nothing.
    request = Request(PROC_RUN, {"argv": ["git", "pull"]})

    assert classify(request, Chain().push("update")).level == UNSAFE
    assert classify(
        request, Chain(approved=frozenset({PROC_RUN})).push("update")
    ).level == SAFE


def test_approval_does_not_authorize_undeclared_requests():
    """The grant is the declaration, not a skeleton key.

    A command approved for the shell has not thereby been approved to make
    network calls — which is the whole point of scoping the grant, since
    egress is the control that makes generous local access safe.
    """
    granted = Chain(approved=frozenset({PROC_RUN})).push("update")

    assert classify(Request(PROC_RUN, {"argv": ["git", "pull"]}),
                    granted).level == SAFE
    assert classify(Request(NET_HTTP, {"url": "https://example.invalid"}),
                    granted).level == UNSAFE
    assert classify(Request(FS_WRITE, {"path": "/etc/passwd"}),
                    granted).level == UNSAFE


def test_an_empty_grant_authorizes_nothing():
    """A command declaring no Requests buys nothing with its approval."""
    chain = Chain(approved=frozenset()).push("update")
    assert classify(Request(PROC_RUN, {"argv": ["git", "pull"]}),
                    chain).level == UNSAFE


def test_a_grant_cannot_be_widened_by_descending():
    """``push`` copies the grant down; nothing along the way can add to it."""
    chain = Chain(approved=frozenset({PROC_RUN})).push("update")
    deeper = chain.push("helper").push("deeper")

    assert deeper.approved == frozenset({PROC_RUN})
    assert classify(Request(NET_HTTP, {"url": "https://example.invalid"}),
                    deeper).level == UNSAFE


def test_the_dialog_names_the_scope_rather_than_just_the_command():
    """The user has to be told what the yes covers, or it is not answerable."""
    prompt = describe_grant("update", [PATH_GET, PROC_RUN])

    assert "/update wants to:" in prompt
    assert "run shell commands" in prompt
    assert "look up application folders" in prompt
    # Consequence leads: nobody reads past the first line of a dialog.
    assert prompt.index("run shell commands") < prompt.index("look up")


def test_a_command_declaring_nothing_falls_back_to_the_bare_question():
    """A command with no consequential effect still has to render a dialog."""
    assert describe_grant("legacy", []) == "Approve /legacy?"


def test_every_request_type_has_a_human_phrase():
    """A dotted name in an approval dialog is a question nobody can answer.

    Totality is the point: a Request added without a phrase would render as
    ``fs.some_new_verb`` in front of a user, which is exactly the failure the
    dialog exists to prevent.
    """
    bare = sorted(k for k in ALL_TYPES if phrase_for(k) == k)
    assert not bare, f"Request types with no human phrase: {bare}"


def test_the_prompt_is_rendered_from_the_same_declaration_as_the_grant():
    """The question asked and the authority handed over share one source.

    ``requests`` is deliberately not copied onto the adapter — it means
    something to the sandbox, not to the kernel — so the declaration is read
    back from the validator, which is where the bridge got it too.
    """
    path = Path("bundled/commands/command_update.py")
    declared = validate_file(path).declarations["requests"]
    module = adapt(path)
    command = next(
        value for value in vars(module).values()
        if isinstance(value, type) and value.__name__.endswith("UpdateCommand"))

    assert command.approval_prompt == describe_grant("update", declared)


def test_update_declares_exactly_what_its_approval_grants():
    """The dialog, the declaration and the policy must not disagree.

    ``/update`` is the one approval-gated command, so its declaration is the
    live example of the grant. If it drifts, a user approving the command is
    consenting to something other than what runs.
    """
    report = validate_file(Path("bundled/commands/command_update.py"))
    assert report.ok, report.render()
    assert set(report.declarations["requests"]) == {PATH_GET, PROC_RUN}


def test_both_dialogs_speak_one_vocabulary():
    """The up-front grant and the per-Request dialog describe a capability
    the same way.

    They ask different questions — "what will this command be allowed to do"
    versus "may this specific thing happen now" — but a user who approved
    "run shell commands" and then sees "Run shell commands: git pull" is
    reading one system. Two hand-written tables would drift.
    """
    from sandbox.approval import _action_line, phrase_for

    for kind in sorted(ALL_TYPES):
        phrase = phrase_for(kind)
        headline = _action_line(Request(kind, {})).split(":")[0].split("\n")[0]
        assert headline.lower().startswith(phrase.lower()[:12]), (
            f"{kind} reads as {headline!r} at execution time but "
            f"{phrase!r} in a grant")


def test_no_request_reaches_a_user_as_a_dotted_name():
    """``_action_line`` used to fall through to ``fs.some_verb`` for 69 of
    the 97 types.

    That was invisible while nothing wired an approver, because the dialog
    never rendered. The moment one was wired, most approvals would have asked
    a person to authorise ``session.add_tool``.
    """
    from sandbox.approval import _action_line

    bare = [k for k in sorted(ALL_TYPES)
            if _action_line(Request(k, {})) == f"`{k}`"]
    assert not bare, f"Requests rendering as a bare name: {bare}"


def test_an_update_is_not_announced_as_a_removal():
    """Overstating what is about to happen erodes the dialog's credibility.

    The family fallback lumps install/remove/reload together, which named a
    removal that was never going to occur.
    """
    from sandbox.approval import _action_line
    from sandbox.guest.requests import PLUGIN_UPDATE

    line = _action_line(Request(PLUGIN_UPDATE, {}))
    assert "remove" not in line.lower()
    assert "update" in line.lower()
