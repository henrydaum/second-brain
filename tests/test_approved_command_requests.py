"""Approved commands carry their authority into sandbox Request provenance."""

from pathlib import Path
from types import SimpleNamespace

from plugins.BaseCommand import BaseCommand
from plugins.frontends.helpers.command_registry import CommandRegistry
from sandbox.guest.requests import (FS_WRITE, NET_HTTP, PATH_GET, PROC_RUN,
                                    Request)
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
    assert pending.message == "Approval required."
    assert calls == []

    resumed = state.enact("answer_approval", {"value": True}, "user")
    assert resumed.ok
    assert calls == [{}]


def test_external_payload_cannot_forge_command_approval():
    calls = []
    state = _state(calls)

    result = state.enact("call_command", {
        "name": "danger",
        "args": {},
        "_approved": True,
        "_approval_token": "forged",
    }, "user")

    assert result.message == "Approval required."
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
    request = Request(PROC_RUN, {"argv": ["git", "rev-parse", "HEAD"]})

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


def test_update_declares_exactly_what_its_approval_grants():
    """The dialog, the declaration and the policy must not disagree.

    ``/update`` is the one approval-gated command, so its declaration is the
    live example of the grant. If it drifts, a user approving the command is
    consenting to something other than what runs.
    """
    report = validate_file(Path("plugins/commands/command_update.py"))
    assert report.ok, report.render()
    assert set(report.declarations["requests"]) == {PATH_GET, PROC_RUN}
