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
        "You are the agent in charge of summarizing this conversation. "
        "The agent from this conversation has hit their context limit. "
        "In order to continue, the context must be compacted by you. "
        "Their memory of this conversation will be replaced with your summary. "
        "Please produce a summary of this conversation that a fresh assistant "
        "instance can resume from without re-reading the transcript. "
        "Cover the Who, What, When, Where, Why, and How of the conversation. "
        "Keep in mind that the agent's memory will be wiped, but the user "
        "still has access to the full transcript. The user cannot read your "
        "summary. Avoid unnecessary details. Focus on the pith."
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
