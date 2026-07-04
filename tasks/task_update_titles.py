"""One-shot conversation titler.

Fired by the ``update_titles`` cron job (seeded ``* * * * *`` — every
minute — via ``default_jobs``; the sweep is one cheap SELECT and exits
immediately when nothing is ripe). A conversation with new messages since
the last sweep is titled once it is *ripe*: the agent has replied AND
either the first agent reply is at least ``title_delay_minutes`` old
(default 10 — the delay buys extra context so similar openers don't
collapse into identical titles) or the conversation already carries
``_EARLY_MESSAGES`` user/agent messages (busy conversations earn their
title early). Only titles that still look kernel-generated ("New
Conversation", "New conversation (Main)", empty, or a "/clear"-stamped
"... (cleared)") are replaced, and only once — a real title (from us or a
user rename) is never overwritten, matching the major chat providers. The
high-water mark advances when a row is processed or already titled (so
nothing replays every tick), but *not* while a row merely isn't ripe yet
— it must come back next minute. Each write emits
``CONVERSATION_CHANGED`` with ``action='retitled'`` so frontends refresh
sidebars/banners live.
"""

dependencies_files = ['services/service_litellm.py']
dependencies_pip = []

import json
import logging
import re
import time

from events.event_bus import bus
from events.event_channels import CONVERSATION_CHANGED
from plugins.BaseTask import BaseTask, TaskResult
from runtime.agent_scope import resolve_agent_llm
from runtime.token_stripper import strip_model_tokens

logger = logging.getLogger("TaskUpdateTitles")

UPDATE_TITLES = "update_titles"
"""Plugin-owned event channel for periodic conversation retitling."""

_MAX_LEN = 80
# A conversation with this many user/agent messages is titled without
# waiting out the delay — it already carries enough context.
_EARLY_MESSAGES = 6

# Titles that still look kernel-generated and therefore may be replaced.
# Covers "New Conversation", "New conversation (Main)" and friends.
_DEFAULT_TITLE = re.compile(r"^new conversation\b", re.IGNORECASE)


def _needs_title(title) -> bool:
    """True while the conversation has never been given a real title."""
    text = str(title or "").strip()
    return not text or bool(_DEFAULT_TITLE.match(text)) or text.endswith("(cleared)")

_SYSTEM_PROMPT = (
    "You label conversations with short, concrete titles. "
    "You output only the title — never a sentence, greeting, or explanation."
)

_USER_TEMPLATE = (
    "<conversation>\n"
    "{transcript}\n"
    "</conversation>\n\n"
    "Write a 2-6 word title summarizing what the conversation is about.\n"
    "Rules:\n"
    "- Output only the title, no preamble, no quotes, no markdown\n"
    "- Be concrete and specific, not generic\n"
    "- Use title case\n\n"
    "Examples:\n"
    "Conversation about Rolls-Royce Cullinan pricing -> Cullinan Price\n"
    "Conversation planning a Virginia holiday -> Virginia Holiday Getaway\n"
    "Conversation debugging a SQLite migration -> SQLite Migration Bug\n\n"
    "Title:"
)


class UpdateTitles(BaseTask):
    """Update titles."""
    name = "update_titles"
    trigger = "event"
    trigger_channels = [UPDATE_TITLES]
    requires_services = ["llm"]
    writes = []
    timeout = 600
    event_payload_schema = {"type": "object", "properties": {}, "required": []}
    default_jobs = {
        "update_titles": {"channel": UPDATE_TITLES, "cron": "* * * * *", "payload": {}},
    }

    config_settings = [
        ("Title Update LLM Profile", "title_update_llm_profile",
         "Agent profile whose LLM is used to generate conversation titles. "
         "'default' follows the default LLM.",
         "default", {"type": "text"}),

        ("Title Delay (minutes)", "title_delay_minutes",
         "How long after the agent's first reply a conversation waits before "
         "being titled — the wait accumulates context so similar openers get "
         "distinct titles. Busy conversations are titled as soon as they "
         "reach 6 user/agent messages. 0 titles right after the first reply.",
         10, {"type": "slider", "range": (0, 60, 60), "is_float": False}),
    ]

    # New-message gate (vs the high-water mark) lives in SQL so the
    # every-minute sweep is one indexed SELECT that usually returns nothing.
    _CANDIDATES_SQL = """
        SELECT c.id    AS id,
               c.title AS title,
               (SELECT COUNT(*) FROM conversation_messages m
                  WHERE m.conversation_id = c.id) AS total_count,
               (SELECT COUNT(*) FROM conversation_messages m
                  WHERE m.conversation_id = c.id
                    AND m.role IN ('user', 'assistant')) AS content_count,
               (SELECT MIN(m.timestamp) FROM conversation_messages m
                  WHERE m.conversation_id = c.id
                    AND m.role = 'assistant') AS first_agent_ts
        FROM conversations c
        WHERE (SELECT COUNT(*) FROM conversation_messages m
                 WHERE m.conversation_id = c.id)
              > COALESCE(c.last_title_check_message_count, 0)
        ORDER BY c.updated_at DESC
    """

    def _candidates(self, db) -> list[dict]:
        """Conversations with messages the sweep hasn't seen yet, as dicts."""
        out = db.query(self._CANDIDATES_SQL, max_rows=100)
        columns = out.get("columns") or []
        return [dict(zip(columns, row)) for row in out.get("rows") or []]

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        """Run event."""
        db = getattr(context, "db", None)
        if db is None:
            return TaskResult.failed("No database available.")

        profile_name = (context.config.get("title_update_llm_profile") or "default").strip() or "default"
        llm = resolve_agent_llm(profile_name, context.config, context.services)
        if llm is None or not getattr(llm, "loaded", False):
            logger.info("Title update skipped: LLM service for profile '%s' not loaded.", profile_name)
            return TaskResult(success=True)

        try:
            candidates = self._candidates(db)
        except Exception as e:
            return TaskResult.failed(f"Failed to list conversations for title check: {e}")

        if not candidates:
            return TaskResult(success=True)

        try:
            delay_minutes = float(context.config.get("title_delay_minutes", 10))
        except (TypeError, ValueError):
            delay_minutes = 10.0
        now = time.time()

        updated = 0
        for row in candidates:
            conversation_id = row.get("id")
            message_count = int(row.get("total_count") or 0)
            if not _needs_title(row.get("title")):
                # Titled once (by us or by the user) — never overwrite.
                # Advance the mark so the row leaves the candidate list.
                try:
                    db.update_conversation_title_check_count(conversation_id, message_count)
                except Exception:
                    pass
                continue
            first_agent_ts = row.get("first_agent_ts")
            if not first_agent_ts:
                continue  # agent hasn't replied yet; revisit next tick
            ripe = (now - float(first_agent_ts)) >= delay_minutes * 60 \
                or int(row.get("content_count") or 0) >= _EARLY_MESSAGES
            if not ripe:
                continue  # don't advance the mark — it must come back
            try:
                self._process_conversation(db, llm, conversation_id, message_count)
                updated += 1
            except Exception as e:
                logger.warning("Title update failed for conversation %s: %s", conversation_id, e)
                # Still advance the high-water mark so a permanently bad
                # conversation doesn't block the sweep next tick.
                try:
                    db.update_conversation_title_check_count(conversation_id, message_count)
                except Exception:
                    pass

        logger.info("Title update sweep: processed %d/%d conversations.", updated, len(candidates))
        return TaskResult(success=True)

    def _process_conversation(self, db, llm, conversation_id, message_count: int) -> None:
        """Internal helper to handle process conversation."""
        messages = db.get_conversation_messages(conversation_id) or []
        # Always advance the high-water mark — even if we skip or fail —
        # so an empty / un-titleable conversation does not replay.
        try:
            transcript = _transcript(messages)
            if not transcript:
                return
            response = llm.invoke([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_TEMPLATE.format(transcript=transcript)},
            ])
            if getattr(response, "error", None):
                return
            title = _sanitize(getattr(response, "content", ""))
            if title:
                db.update_conversation_title(conversation_id, title)
                # Frontends (sidebars, pinned banners) refresh off this;
                # the kernel only emits created/deleted/recategorized.
                bus.emit(CONVERSATION_CHANGED, {"action": "retitled", "conversation_id": conversation_id})
                logger.info("Updated conversation %s title to '%s'.", conversation_id, title)
        finally:
            db.update_conversation_title_check_count(conversation_id, message_count)


# ======================================================================
# Pure helpers
# ======================================================================

def _transcript(messages: list[dict]) -> str:
    """Internal helper to handle transcript."""
    lines = []
    for msg in messages[:12]:
        role = (msg.get("role") or "").upper()
        if role == "TOOL":
            continue
        content = msg.get("content") or ""
        if role == "ASSISTANT":
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    content = parsed.get("content") or ""
            except Exception:
                pass
        content = " ".join(content.split()).strip()
        if not content:
            continue
        if len(content) > 300:
            content = content[:300].rstrip() + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _sanitize(text: str) -> str:
    """Internal helper to handle sanitize."""
    title, _ = strip_model_tokens(text or "")
    title = title.strip()
    if not title:
        return ""
    title = title.splitlines()[0].strip()
    title = title.strip().strip("\"'`*#-: ")
    title = " ".join(title.split())
    title = title[:_MAX_LEN].strip()
    generic = {"new conversation", "conversation", "chat", "untitled", "title"}
    if not title or title.casefold() in generic:
        return ""
    return title
