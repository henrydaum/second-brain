"""Periodic conversation-title refresher.

Fired by the ``update_titles`` cron job (default ``*/30 * * * *``). For
every conversation whose message count has advanced by at least
``_THRESHOLD`` since the last sweep, asks the configured LLM for a fresh
short title and writes it back. The high-water mark is updated even when
the LLM returns nothing, so a flaky reply does not replay every tick.
"""

import json
import logging

from events.event_channels import UPDATE_TITLES
from plugins.BaseTask import BaseTask, TaskResult
from runtime.agent_scope import resolve_agent_llm
from runtime.token_stripper import strip_model_tokens

logger = logging.getLogger("TaskUpdateTitles")

_MAX_LEN = 80
_THRESHOLD = 4

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

    config_settings = [
        ("Title Update LLM Profile", "title_update_llm_profile",
         "Agent profile whose LLM is used to generate conversation titles. "
         "'default' follows the default LLM.",
         "default", {"type": "text"}),
    ]

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
            candidates = db.list_conversations_for_title_check(_THRESHOLD)
        except Exception as e:
            return TaskResult.failed(f"Failed to list conversations for title check: {e}")

        if not candidates:
            return TaskResult(success=True)

        updated = 0
        for row in candidates:
            conversation_id = row.get("id")
            message_count = int(row.get("message_count") or 0)
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
