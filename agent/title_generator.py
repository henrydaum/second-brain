"""
Conversation title generation.

Best-effort, async title generation for conversations. When the first
assistant reply lands, the runtime fires `maybe_generate_async` which
spins up a daemon thread, asks the LLM for a short title, and writes it
back to the database — only if the current title is still a generic
fallback.
"""

import json
import logging
import threading

from runtime.token_stripper import strip_model_tokens

logger = logging.getLogger("TitleGenerator")

_MAX_LEN = 80

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


class TitleGenerator:
    """Generates conversation titles asynchronously via the LLM service."""

    def __init__(self, db, services: dict):
        self.db = db
        self.services = services
        self._lock = threading.Lock()
        self._pending: set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_generate_async(self, conversation_id: int):
        """Best-effort async title generation for a conversation.

        Runs only when the conversation still has an auto-generated fallback
        title, so manual or already-upgraded titles are left untouched.
        """
        if not conversation_id:
            return

        with self._lock:
            if conversation_id in self._pending:
                return
            self._pending.add(conversation_id)

        thread = threading.Thread(
            target=self._worker,
            args=(conversation_id,),
            daemon=True,
            name=f"TitleGen-{conversation_id}",
        )
        thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker(self, conversation_id: int):
        try:
            self._generate(conversation_id)
        except Exception as e:
            logger.debug(f"Conversation title generation failed for {conversation_id}: {e}")
        finally:
            with self._lock:
                self._pending.discard(conversation_id)

    def _generate(self, conversation_id: int):
        conversation = self.db.get_conversation(conversation_id)
        if not conversation:
            return

        messages = self.db.get_conversation_messages(conversation_id)
        if len(messages) < 2:
            return

        current_title = _normalize(conversation.get("title"))
        fallback_title = _fallback_title(messages)
        if not _should_replace(current_title, fallback_title):
            return

        llm = self.services.get("llm")
        if llm is None or not getattr(llm, "loaded", False):
            return
        if getattr(llm, "active", None) is None and hasattr(llm, "active"):
            return

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
        if not title:
            return

        latest = self.db.get_conversation(conversation_id)
        if not latest:
            return
        latest_title = _normalize(latest.get("title"))
        if not _should_replace(latest_title, fallback_title):
            return

        self.db.update_conversation_title(conversation_id, title)
        logger.info(f"Updated conversation {conversation_id} title to '{title}'")


# ======================================================================
# Pure helper functions (no state, easily testable)
# ======================================================================

def _transcript(messages: list[dict]) -> str:
    lines = []
    for msg in messages[:6]:
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


def _fallback_title(messages: list[dict]) -> str:
    for msg in messages:
        if (msg.get("role") or "") == "user":
            return _truncate(msg.get("content") or "")
    return "New conversation"


def _should_replace(current_title: str, fallback_title: str) -> bool:
    if not current_title:
        return True
    lowered = current_title.casefold()
    if lowered in {"new conversation", "conversation", "new chat", "chat"}:
        return True
    if current_title == fallback_title:
        return True
    return False


def _truncate(text: str) -> str:
    text = " ".join((text or "").replace("\n", " ").split()).strip()
    if not text:
        return "New conversation"
    return text[:_MAX_LEN]


def _normalize(text: str | None) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def _sanitize(text: str) -> str:
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
