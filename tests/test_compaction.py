"""Compaction as an act somebody can ask for.

The loop has always been able to compact; what is new is that ``/compact``
runs the same act outside a turn. Three things have to hold for that to be
worth having:

- every way of doing nothing has a *name*, because the loop discards those
  and a person does not
- the two durable effects — the marker row and the checkpoint flag — land,
  since without them the next turn's ``replace_conversation_messages``
  overwrites the full transcript with the stub compaction leaves behind
- a drive already holding the history list is refused rather than raced
"""

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import sandbox  # noqa: F401  - installs the ``guest`` package alias
from events.event_bus import bus
from events.event_channels import SESSION_COMPACTED
from runtime.compaction import Compaction, compact_history
from runtime.conversation_runtime import ConversationRuntime
from sandbox.guest.codes import ERROR_UNAVAILABLE
from sandbox.guest.requests import SESSION_COMPACT
from state_machine.serialization import latest_compaction
from tests.support import call_handler, make_runtime

ROOT = Path(__file__).resolve().parents[1]


class _Compactor:
    """The compactor service as the loop sees it: loaded, and one export."""

    loaded = True

    def __init__(self, summary="Earlier summary."):
        self.summary = summary
        self.calls = []

    def compact(self, **kwargs):
        self.calls.append(kwargs)
        return self.summary


def _history(n=6):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"message {i} " + "x" * 200}
            for i in range(n)]


# ──────────────────────────────────────────────────────────────────────
# Every way of doing nothing has a name.
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("runtime,history,expected", [
    (SimpleNamespace(services={"compactor": _Compactor()}, sessions={}),
     [{"role": "user", "content": "hi"}], "nothing to compact"),
    (SimpleNamespace(services={}, sessions={}), None,
     "compactor service is not loaded"),
    (SimpleNamespace(services={"compactor": SimpleNamespace(loaded=False)},
                     sessions={}), None, "compactor service is not loaded"),
    (SimpleNamespace(services={"compactor": _Compactor("")}, sessions={}),
     None, "returned no summary"),
])
def test_each_refusal_says_which_one_it_was(runtime, history, expected):
    """The loop swallows all four identically; a command cannot.

    "Nothing happened" covers a conversation too short to summarize, a
    compactor that is not installed, and a model that answered with nothing —
    three different things to be told, and the difference is the whole reason
    ``compact_history`` reports instead of returning early.
    """
    history = _history() if history is None else history
    length = len(history)

    outcome = compact_history(runtime, "chat", history)

    assert not outcome.ok
    assert expected in outcome.reason
    assert len(history) == length


def test_compacting_twice_in_a_row_refuses_instead_of_summarizing_a_summary():
    """The head guard used to be ``len(history) <= 2``, which a compacted
    history clears — it has two bridge rows plus two tail rows. So a second
    ``/compact`` spent a model call to re-summarize its own summary, kept the
    identical tail, and reported saving nothing while quietly degrading the
    only record of the earlier conversation.

    Automatic compaction never showed this: four rows are not context
    pressure. Typing the command twice is a thing a person does.
    """
    compactor = _Compactor()
    runtime = SimpleNamespace(services={"compactor": compactor}, sessions={})
    history = _history(8)
    assert compact_history(runtime, "chat", history).ok

    again = compact_history(runtime, "chat", history)

    assert not again.ok
    assert "nothing to compact" in again.reason
    assert len(compactor.calls) == 1        # no second model call


def test_one_more_turn_makes_it_worth_compacting_again():
    """The guard asks whether the rows a compaction would *drop* are all
    already summaries — so real conversation arriving on top clears it. A
    guard that did not would be a lockout."""
    compactor = _Compactor()
    runtime = SimpleNamespace(services={"compactor": compactor}, sessions={})
    history = _history(8)
    compact_history(runtime, "chat", history)
    history.extend(_history(2))

    assert compact_history(runtime, "chat", history).ok
    assert len(compactor.calls) == 2


def test_a_refusal_leaves_the_history_exactly_as_it_was():
    """A failed compaction must not be a partial one."""
    history = _history()
    before = [dict(row) for row in history]

    compact_history(SimpleNamespace(services={}, sessions={}), "chat", history)

    assert history == before


# ──────────────────────────────────────────────────────────────────────
# What it cost, and what it saved.
# ──────────────────────────────────────────────────────────────────────

def test_the_report_measures_provider_visible_characters():
    """``chars_saved`` is what a person asked for, so it is measured over what
    actually occupies the context window — the rendered content, not the raw
    rows, whose ``author`` and attachment records never reach a model."""
    history = _history(8)
    expected_before = sum(len(row["content"]) for row in history)

    outcome = compact_history(
        SimpleNamespace(services={"compactor": _Compactor()}, sessions={}),
        "chat", history)

    assert outcome.ok
    assert outcome.messages_before == 8
    assert outcome.messages_after == 4      # two bridge rows + two tail rows
    assert outcome.chars_before == expected_before
    assert outcome.chars_after == sum(len(row["content"]) for row in history)
    assert outcome.chars_saved == outcome.chars_before - outcome.chars_after
    assert outcome.chars_saved > 0
    assert outcome.summary_chars == len("Earlier summary.")


def test_every_field_reaches_the_wire():
    """Derived from ``fields``, not from a list somebody has to remember.

    The hand-enumerated version is the trap ``Result.to_dict`` already paid
    for: a field added to the report but not to the serializer reaches nobody,
    and the symptom is a command reading zero rather than anything that
    raises.
    """
    from dataclasses import fields

    payload = Compaction(True).as_dict()

    assert set(payload) == {f.name for f in fields(Compaction)} | {"chars_saved"}


def test_the_wire_shape_carries_the_derived_number():
    """``chars_saved`` crosses rather than being recomputed on the far side.

    Two subtractions of one pair of numbers is how the command and the kernel
    come to disagree about what just happened.
    """
    outcome = compact_history(
        SimpleNamespace(services={"compactor": _Compactor()}, sessions={}),
        "chat", _history())

    payload = outcome.as_dict()

    assert payload["chars_saved"] == payload["chars_before"] - payload["chars_after"]
    assert payload["ok"] is True


# ──────────────────────────────────────────────────────────────────────
# compact_session: the durable half.
# ──────────────────────────────────────────────────────────────────────

def _session(busy=False):
    return SimpleNamespace(
        lock=threading.RLock(), busy=busy, conversation_id=1,
        has_compaction_checkpoint=False, history=_history())


def _runtime_with(session, **kwargs):
    runtime = SimpleNamespace(
        sessions={"chat": session} if session else {},
        services={"compactor": _Compactor()}, db=None, **kwargs)
    runtime.compact_session = (
        lambda key: ConversationRuntime.compact_session(runtime, key))
    return runtime


def test_compacting_a_session_nobody_opened_says_so():
    """A refusal answers with the same shape a success does, zeroed — a
    caller reading ``chars_saved`` should not have to branch on ``ok`` first
    to know the key is there."""
    outcome = _runtime_with(None).compact_session("chat")

    assert outcome == {"ok": False, "reason": "no active session",
                       "messages_before": 0, "messages_after": 0,
                       "chars_before": 0, "chars_after": 0,
                       "chars_saved": 0, "summary_chars": 0}


def test_a_driving_turn_is_refused_rather_than_raced():
    """``session.busy`` is set only around the agent turn, so it is exactly
    "a drive owns this history list right now" — and compaction rewrites that
    list by slice assignment underneath whoever is iterating it.

    A command's own phase does not set the flag, which is what keeps this from
    refusing itself.
    """
    session = _session(busy=True)
    before = [dict(row) for row in session.history]

    outcome = _runtime_with(session).compact_session("chat")

    assert not outcome["ok"]
    assert outcome["reason"] == "the agent is mid-turn"
    assert session.history == before


def test_compacting_writes_the_marker_and_sets_the_checkpoint(tmp_path):
    """The pairing that keeps the transcript. Without the marker the reload
    has nothing to replay from; without the flag the post-turn
    ``replace_conversation_messages`` overwrites the full transcript with the
    four rows compaction leaves behind.
    """
    compactor = _Compactor()
    runtime, session, _ = make_runtime(
        tmp_path, services={"compactor": compactor}, session_key="chat")
    session.history = _history(8)
    seen = []
    unsubscribe = bus.subscribe(SESSION_COMPACTED, seen.append)
    try:
        outcome = runtime.compact_session("chat")
    finally:
        unsubscribe()

    assert outcome["ok"], outcome["reason"]
    assert session.has_compaction_checkpoint is True
    rows = runtime.db.get_conversation_messages(session.conversation_id)
    assert latest_compaction(rows) is not None
    assert len(session.history) == 4
    assert session.history[0]["author"] == "compaction"
    assert compactor.calls[0]["session_key"] == "chat"


def test_the_command_path_narrates_nothing_into_the_conversation(tmp_path):
    """The automatic path narrates because the history changes under the
    user's feet mid-turn. Here they asked, and the command they are watching
    narrates itself — a second ``persist=False`` notification would be noise.
    """
    notices = []
    runtime, session, _ = make_runtime(
        tmp_path, services={"compactor": _Compactor()}, session_key="chat")
    runtime.on_notice = notices.append
    session.history = _history(8)

    runtime.compact_session("chat")

    assert notices == []


# ──────────────────────────────────────────────────────────────────────
# The handler: the one hop the runtime and the command tests both skip.
# ──────────────────────────────────────────────────────────────────────

def test_the_handler_reports_a_refusal_as_a_failed_result():
    """A named reason has to survive the hop, or the command shows the
    generic fallback and the whole point of naming them is lost."""
    ctx = SimpleNamespace(runtime=_runtime_with(_session(busy=True)),
                          session_key="chat")

    result = call_handler(SESSION_COMPACT, ctx, {})

    assert not result.ok
    assert result.error == "the agent is mid-turn"


def test_the_handler_scopes_to_the_context_not_an_argument():
    """There is no ``key`` on the wire, so a guest naming somebody else's
    session has nothing to name it with. The argument dict is ignored
    entirely — pinned here because adding one later would look harmless."""
    ctx = SimpleNamespace(runtime=_runtime_with(_session()),
                          session_key="chat")

    result = call_handler(SESSION_COMPACT, ctx, {"key": "somebody-else"})

    assert result.ok
    assert result.data["messages_before"] == 6
    assert result.data["chars_saved"] > 0


def test_the_handler_says_so_when_the_kernel_cannot_compact_at_all():
    """A runtime with no ``compact_session`` is "this kernel cannot", which
    ``_need`` reports as unavailable — a different thing from "there is
    nothing to compact", and a plugin retrying the second must not retry the
    first."""
    ctx = SimpleNamespace(runtime=SimpleNamespace(), session_key="chat")

    result = call_handler(SESSION_COMPACT, ctx, {})

    assert not result.ok
    assert result.code == ERROR_UNAVAILABLE


# ──────────────────────────────────────────────────────────────────────
# The command.
# ──────────────────────────────────────────────────────────────────────

def _compact_command():
    """``/compact``, loaded from source without going through discovery."""
    spec = importlib.util.spec_from_file_location(
        "command_compact_under_test",
        ROOT / "bundled" / "commands" / "command_compact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CompactCommand()


class _Failed(Exception):
    """Stands in for ``sdk.Failed``. One class, defined once: the command
    catches ``sdk.Failed``, so a per-call class would sail straight past the
    ``except`` and the test would prove nothing about the handler."""

    def __init__(self, error):
        super().__init__(error)
        self.error = error


def _fake_sdk(answer, progress):
    from sandbox.guest.sdk import _Markdown

    def compact():
        if isinstance(answer, Exception):
            raise answer
        return answer

    return SimpleNamespace(
        Failed=_Failed, md=_Markdown,
        ui=SimpleNamespace(progress=progress.append),
        session=SimpleNamespace(compact=compact))


def test_the_card_states_both_sizes_and_the_share_saved():
    """The absolute figure alone does not say whether it was worth doing:
    40,000 characters saved means something different out of 45,000 than out
    of 400,000."""
    progress = []
    output = _compact_command().run(_fake_sdk({
        "ok": True, "messages_before": 34, "messages_after": 4,
        "chars_before": 45000, "chars_after": 5000, "chars_saved": 40000,
        "summary_chars": 1820}, progress), {})

    assert progress == ["Compacting conversation..."]
    assert "34 -> 4" in output
    assert "45,000 -> 5,000" in output
    assert "40,000 chars (89%)" in output
    assert "1,820 chars" in output


def test_a_refusal_reaches_the_person_as_its_reason():
    """The reason is the whole value of the failure — "could not compact" on
    its own is what the loop's silent return already said."""
    output = _compact_command().run(
        _fake_sdk(_Failed("there is nothing to compact yet"), []), {})

    assert "nothing to compact yet" in output
