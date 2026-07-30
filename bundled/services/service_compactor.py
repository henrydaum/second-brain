"""Conversation compaction service."""

from guest.bases import BaseService


class CompactorService(BaseService):
    """Summarize conversation history when the active LLM context is tight."""

    name = "compactor"
    description = "Summarize conversation history when the active LLM context is tight."
    lifecycle = "extension"
    exports = ["compact"]
    requests = ["agent.complete"]

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

    def start(self, sdk):
        """The service holds no resources between calls."""
        return True

    def stop(self, sdk):
        """The service holds no resources between calls."""
        return None

    def compact(
        self,
        sdk,
        *,
        session_key: str | None = None,
        transcript: str,
    ) -> str | None:
        """Return a continuation summary for a rendered transcript."""
        if not transcript:
            return ""
        try:
            response = sdk.agent.complete(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                session_key=session_key,
            )
        except sdk.Failed as exc:
            sdk.log(f"Compaction failed: {exc}", level="warning")
            return None
        return (response.get("content") or "").strip()
