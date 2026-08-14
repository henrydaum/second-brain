"""Summarize a conversation's history in place, and say what that cost.

This is the whole of what "compacting" means, in one function with two
callers. The loop's context-safety escort calls it when the context gets
tight (``ConversationLoop._compact``); ``ConversationRuntime.compact_session``
calls it when a person asks for the same thing by typing ``/compact``.

It lives here rather than on the loop because the loop is turn-scoped —
``_active_db`` and ``_active_conversation_id`` are set at the top of
``drive()`` and cleared in its ``finally`` — so a compaction asked for outside
a turn had no way to reach the two things that make it durable: the marker row
and the checkpoint flag. Without those the next turn's
``replace_conversation_messages`` overwrites the full transcript with the
four-row stub this leaves behind.

Nothing here swallows exceptions. The loop's wrapper does, because a turn must
never break over compaction; the command path needs to know what happened, and
a guard that returns silently is indistinguishable from one that worked.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable

from events.event_bus import bus
from events.event_channels import SESSION_COMPACTED
from state_machine.serialization import save_compaction_marker
from runtime.conversation_loop import (ConversationLoop, _for_provider,
                                       _truncate_middle)

logger = logging.getLogger("Compaction")

# How much of each row the summarizer is shown, and how much of the whole
# transcript. Both were literals inside ``_compact``; they are named here
# because a caller reporting sizes should not have to guess at them.
ROW_CHARS = 1000
TRANSCRIPT_CHARS = 20000


@dataclass
class Compaction:
    """What one compaction did, or why it did nothing.

    ``chars`` is measured over the rendered provider content, which is what
    actually occupies the context window — not over the raw rows, whose
    ``author`` and attachment records never reach a model.
    """

    ok: bool
    reason: str = ""
    messages_before: int = 0
    messages_after: int = 0
    chars_before: int = 0
    chars_after: int = 0
    summary_chars: int = 0

    @property
    def chars_saved(self) -> int:
        """How much smaller the history got. Negative is possible in principle
        — a two-message conversation whose summary is longer than it was — and
        is worth reporting honestly rather than clamping to zero."""
        return self.chars_before - self.chars_after

    def as_dict(self) -> dict[str, Any]:
        """The wire shape.

        Derived from ``fields`` rather than enumerated, which is the lesson
        ``Result.to_dict``/``from_dict`` already paid for: a hand-written list
        means a field added here reaches nobody, and the symptom is a key the
        command reads as zero rather than anything that raises.

        ``chars_saved`` is the one addition, because a *property* is not a
        field and it is the number the person actually asked for. Sending it
        beats recomputing it on the far side — that is how the two come to
        disagree about the same subtraction.
        """
        return {**asdict(self), "chars_saved": self.chars_saved}


def _rendered_size(history) -> int:
    """Characters of provider-visible content across a history list."""
    total = 0
    for row in history:
        content = _for_provider(row).get("content")
        if isinstance(content, str):
            total += len(content)
        elif content is not None:
            total += len(str(content))
    return total


def _shrink_for_tail(msg: dict[str, Any]) -> dict[str, Any]:
    """Aggressively truncate any oversized message preserved through
    compaction. Without this, a huge ``role: tool`` result in the last
    two messages would survive compaction intact and the post-compact
    retry would overflow again."""
    cap = ConversationLoop.MAX_TOOL_RESULT_CHARS
    content = msg.get("content")
    if not isinstance(content, str) or len(content) <= cap:
        return msg
    return {**msg, "content": _truncate_middle(content, cap)}


def render_transcript(history) -> str:
    """The history as the summarizer is shown it.

    Rendered, not raw: a summary is written *for* a model, so it should read
    what the model read — including that a file arrived.

    An authored row is labelled SYSTEM rather than by its role. The label is
    computed here rather than by ``_for_provider``, which strips ``author`` —
    so without this a cancel notice entered the summary as "USER: [The user
    cancelled the previous turn…]" and that misattribution is what survives
    into every later context, long after the row itself has been compacted
    away.

    Keeps head + tail so the summary covers both how the conversation started
    and what was most recently said, instead of silently dropping everything
    after the first 20k chars.
    """
    transcript = "\n".join(
        f"{'SYSTEM' if row.get('author') else (row.get('role') or '').upper()}: "
        f"{(_for_provider(row).get('content') or '')[:ROW_CHARS]}"
        for row in history)
    return _truncate_middle(transcript, TRANSCRIPT_CHARS)


def _nothing_to_fold_in(history) -> bool:
    """Whether a compaction would only summarize its own last summary.

    The head guard has always been ``len(history) <= 2`` — "there is nothing
    but the tail we would keep anyway". Once a compaction has run, the head is
    two ``author="compaction"`` bridge rows, so a four-row history reads as
    long enough while holding nothing new: the model call re-summarizes a
    summary, keeps the identical tail, and reports saving nothing.

    Automatic compaction never showed this — it fires on real context
    pressure, which four rows are not — but ``/compact`` twice in a row is a
    thing a person does, and degrading the summary each time is the worst
    version of doing nothing.

    So the question is about the rows a compaction would *drop*: if every one
    of them is already a summary, there is nothing to fold in. One more turn
    of real conversation puts non-bridge rows back in that slice and this
    stops matching, which is what makes it a guard rather than a lockout.
    """
    return all(row.get("author") == "compaction" for row in history[:-2])


def compact_history(runtime, session_key: str | None, history, *,
                    db=None, conversation_id: int | None = None,
                    on_notice: Callable[[str], Any] | None = None) -> Compaction:
    """Summarize the head of ``history`` in place via the compactor service.

    ``history`` is mutated by slice assignment, so the caller's list object is
    the one that changes — for the loop that list *is* ``session.history``,
    which is what makes a compaction visible to the next turn.

    ``db`` and ``conversation_id`` are optional only because the loop's are:
    outside a ``drive()`` they are ``None``, and a compaction with no marker
    is an in-memory shrink the next reload will undo. Callers that have them
    should always pass them.
    """
    if len(history) <= 2 or _nothing_to_fold_in(history):
        return Compaction(False, "there is nothing to compact yet")
    if runtime is None:
        return Compaction(False, "no runtime is available")

    compactor = (getattr(runtime, "services", {}) or {}).get("compactor")
    if compactor is None or not getattr(compactor, "loaded", False):
        logger.warning("Compactor service is not loaded. History will not shrink via summary.")
        return Compaction(False, "the compactor service is not loaded")

    chars_before = _rendered_size(history)
    transcript = render_transcript(history)
    if on_notice:
        on_notice("Compacting conversation...")
    summary = compactor.compact(session_key=session_key, transcript=transcript)
    if not summary:
        logger.warning("Compaction returned no summary. History will not shrink via summary.")
        return Compaction(False, "the compactor returned no summary")

    old_count = len(history)
    if db is not None and conversation_id is not None:
        save_compaction_marker(db, conversation_id, summary)
        session = getattr(runtime, "sessions", {}).get(session_key)
        if session is not None:
            session.has_compaction_checkpoint = True
    bus.emit(SESSION_COMPACTED, {
        "session_key": session_key,
        "conversation_id": conversation_id,
        "messages_compacted": old_count,
        "summary": summary,
    })
    tail = [_shrink_for_tail(m) for m in history[-2:]]
    history[:] = [
        {"role": "user", "author": "compaction", "content": (
            "[Conversation summary from earlier]\n"
            "Earlier turns were compacted away; only this summary remains "
            "visible. The full transcript is preserved in the "
            "conversation_messages table and is queryable if a SQL/history "
            "tool is installed. If the user references something absent from "
            "this summary, say you can't see that far back (or query for it) "
            "— never deny it was said.\n"
            f"{summary}"
        )},
        {"role": "assistant", "author": "compaction", "content": "Understood - I have the earlier context."},
        *tail,
    ]
    if on_notice:
        on_notice(f"Compacted {old_count} messages.")
    return Compaction(
        True,
        messages_before=old_count,
        messages_after=len(history),
        chars_before=chars_before,
        chars_after=_rendered_size(history),
        summary_chars=len(summary),
    )
