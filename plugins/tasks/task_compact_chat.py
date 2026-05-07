import logging

from plugins.BaseTask import BaseTask, TaskResult
from events.event_bus import bus
from events.event_channels import COMPACT_CHAT, CHAT_MESSAGE_PUSHED

logger = logging.getLogger("TaskCompactChat")


class CompactChat(BaseTask):
    """Summarize the head of a conversation history.

    The conversation runtime emits ``compact_chat`` with the pre-rendered
    transcript and a request token; this task picks up the event,
    summarizes via the same LLM as the requesting session, and posts
    the summary back through ``runtime.finish_compaction(token, summary)``.

    The runtime call site blocks on the matching token, so the loop
    behaves exactly as it did before but the LLM work now has its own
    task_runs row, observable cancellation, and an isolated failure
    surface.
    """

    name = "compact_chat"
    trigger = "event"
    trigger_channels = [COMPACT_CHAT]
    event_payload_schema = {
        "type": "object",
        "properties": {
            "request_token": {"type": "string"},
            "session_key": {"type": "string"},
            "transcript": {"type": "string"},
        },
        "required": ["request_token", "transcript"],
    }
    requires_services = ["llm"]
    writes = []
    timeout = 120

    SYSTEM_PROMPT = (
        "Summarize this Second Brain conversation so the assistant can continue "
        "with minimal loss."
    )

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        token = (payload.get("request_token") or "").strip()
        transcript = payload.get("transcript") or ""
        session_key = payload.get("session_key")
        if not token:
            return TaskResult.failed("request_token is required")
        if not transcript:
            self._finish(context, token, summary="")
            return TaskResult(success=True)

        runtime = getattr(context, "runtime", None)
        if runtime is None:
            return TaskResult.failed("ConversationRuntime is not wired into the orchestrator.")

        llm = self._llm_for_session(runtime, session_key) or context.services.get("llm")
        if llm is None or not getattr(llm, "loaded", False):
            self._finish(context, token, summary=None, error="LLM not loaded")
            return TaskResult.failed("LLM service is not loaded.")

        self._push(session_key, "Compacting…")
        try:
            response = llm.chat_with_tools([
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ], None)
        except Exception as e:
            self._finish(context, token, summary=None, error=str(e))
            self._push(session_key, f"Compaction failed: {e}")
            return TaskResult.failed(f"Compaction LLM call failed: {e}")

        if getattr(response, "is_error", False):
            err = getattr(response, "error", None) or "unknown error"
            self._finish(context, token, summary=None, error=str(err))
            self._push(session_key, f"Compaction failed: {err}")
            return TaskResult.failed(f"Compaction LLM returned error: {err}")

        summary = (getattr(response, "content", "") or "").strip()
        self._finish(context, token, summary=summary)
        self._push(session_key, "Compaction complete.")
        return TaskResult(success=True)

    @staticmethod
    def _llm_for_session(runtime, session_key: str | None):
        if not session_key:
            return None
        try:
            from runtime.runtime_config import active_llm
            session = runtime.sessions.get(session_key)
            return active_llm(runtime, session)
        except Exception:
            logger.exception("Failed to resolve session LLM for compaction")
            return None

    @staticmethod
    def _finish(context, token: str, summary: str | None, error: str | None = None):
        runtime = getattr(context, "runtime", None)
        if runtime is None:
            return
        try:
            runtime.finish_compaction(token, summary, error=error)
        except Exception:
            logger.exception("finish_compaction call failed")

    @staticmethod
    def _push(session_key: str | None, text: str):
        payload = {"message": text, "source": "compact_chat", "kind": "note"}
        if session_key:
            payload["session_key"] = session_key
        bus.emit(CHAT_MESSAGE_PUSHED, payload)
