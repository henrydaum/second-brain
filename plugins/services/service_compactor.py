"""Conversation compaction service."""

import logging

from plugins.BaseService import BaseService, EXTENSION

logger = logging.getLogger("CompactorService")


class CompactorService(BaseService):
    """Summarize conversation history when the active LLM context is tight."""

    model_name = "Conversation Compactor"
    lifecycle = EXTENSION

    SYSTEM_PROMPT = (
        "Produce a continuation summary of this Second Brain conversation that a "
        "fresh assistant instance can resume from without re-reading the transcript. "
        "Cover, in order: the user's goal and their current request; decisions made "
        "and why; files, tables, config keys, and conversation/task IDs touched "
        "(exact paths and identifiers, never paraphrases); tool results that are "
        "still relevant; anything promised or in progress; and the concrete next "
        "step. Prefer exact identifiers over description — a wrong or vague path "
        "is worse than a long one. Omit pleasantries and abandoned approaches, "
        "unless knowing an approach failed prevents repeating the mistake."
    )

    def compact(self, *, runtime, session_key: str | None, transcript: str) -> str | None:
        """Return a continuation summary for a rendered transcript."""
        if not transcript:
            return ""
        llm = self._llm_for_session(runtime, session_key) or self.services.get("llm")
        if llm is None or not getattr(llm, "loaded", False):
            logger.warning("Compaction skipped: LLM service is not loaded.")
            return None
        response = llm.chat_with_tools([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ], None)
        if getattr(response, "is_error", False):
            logger.warning("Compaction failed: %s", getattr(response, "error", None) or "unknown error")
            return None
        return (getattr(response, "content", "") or "").strip()

    @staticmethod
    def _llm_for_session(runtime, session_key: str | None):
        if not runtime or not session_key:
            return None
        try:
            from runtime.runtime_config import active_llm
            return active_llm(runtime, runtime.sessions.get(session_key))
        except Exception:
            logger.exception("Failed to resolve session LLM for compaction")
            return None


def build_services(config: dict) -> dict:
    return {"compactor": CompactorService()}
