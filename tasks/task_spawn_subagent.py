"""Task plugin for spawn subagent."""

dependencies_files = ['services/service_litellm.py']
dependencies_pip = []

from pathlib import Path

from attachments import parse_attachment
from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
from plugins.BaseTask import BaseTask, TaskResult
from state_machine.serialization import save_state_marker


# Plugin-owned bus channel, shared with the schedule-subagent tool (which tags
# Timekeeper jobs with it) and imported from here. The task owns the channel as
# its subscriber and is the common dependency of the scheduling feature, so the
# constant lives here rather than in a separate file. The kernel deliberately
# does not reserve this channel.
SPAWN_SUBAGENT = "subagent.spawn"


def cancelled_set(session) -> set:
    """The session's ephemeral set of child cids cancelled at their timeout.

    Written by tool_spawn_agent (wait timeout) and service_subagents (barrier
    timeout); consumed here to keep a cancelled child from starting late and
    from echoing a stale completion notice after its failure was reported.
    """
    current = getattr(session, "cancelled_subagents", None)
    if current is None:
        current = set()
        session.cancelled_subagents = current
    return current


def _consume_cancelled(runtime, payload, cid) -> bool:
    """True (and forget the entry) when the parent already cancelled this cid."""
    key = (payload.get("report_session_key") or "").strip()
    if not key:
        return False
    session = (getattr(runtime, "sessions", {}) or {}).get(key)
    cancelled = getattr(session, "cancelled_subagents", None) if session else None
    if cancelled and cid in cancelled:
        cancelled.discard(cid)
        return True
    return False


class SpawnSubagent(BaseTask):
    """Spawn subagent."""
    name = "spawn_subagent"
    trigger = "event"
    trigger_channels = [SPAWN_SUBAGENT]
    requires_services = ["llm"]
    writes = []
    max_workers = 1
    event_payload_schema = {
        "type": "object",
        "properties": {
            "conversation_id": {"type": "integer", "description": "Conversation ID"},
            "title": {"type": "string", "description": "Conversation title if a new conversation is needed."},
            "prompt": {"type": "string", "description": "Prompt"},
            "attachments": {"type": "array", "description": "Optional file paths"},
            "report_session_key": {"type": "string", "description": "Session key whose message queue receives the child's report — the agent-facing channel, unrelated to the user-facing conversation notification system. Used by the spawn_agent tool; scheduled jobs leave it unset."},
            "report_conversation_id": {"type": "integer", "description": "Conversation the spawn was made from; the report is dropped if the session has since switched to a different conversation."},
        },
        "required": ["prompt"],
    }

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        """Run event."""
        runtime, db = getattr(context, "runtime", None), getattr(context, "db", None)
        if runtime is None or db is None:
            return TaskResult.failed("ConversationRuntime and database are required.")
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return TaskResult.failed("prompt is required.")
        cid = _conversation_id(payload)
        if cid is None or db.get_conversation(cid) is None:
            cid = _create_conversation(db, payload)
            _remember_conversation(context, payload, cid)
        if _consume_cancelled(runtime, payload, cid):
            # The parent hit its timeout before this run was even dispatched —
            # don't start a child whose failure was already reported.
            return TaskResult.failed(f"cancelled before start — the parent timed out waiting (conversation #{cid}).")
        if cid == runtime.active_conversation_id:
            msg = "spawn_subagent cannot run in the active conversation. Switch away or choose another conversation."
            bus.emit(CHAT_MESSAGE_PUSHED, {"message": msg, "source": self.name, "kind": "alert"})
            return TaskResult.failed(msg)

        session_key = f"spawn_subagent:{cid}"
        session = runtime.sessions.get(session_key)
        if session is not None and session.busy:
            return TaskResult.failed(f"spawn_subagent is already running for conversation #{cid}.")

        try:
            attachments = _attachments(payload.get("attachments"), context)
            runtime.open_session(session_key, conversation_id=cid)
            out = runtime.iterate_agent_turn(
                session_key,
                prompt,
                attachments=attachments,
            )
        except Exception as e:
            _report(runtime, payload, cid, ok=False, text=str(e))
            return TaskResult.failed(str(e))
        finally:
            runtime.close_session(session_key)
        if not out.ok:
            error = (out.error or {}).get("message") or "\n".join(out.messages) or "spawn_subagent failed."
            _report(runtime, payload, cid, ok=False, text=error)
            return TaskResult.failed(error)
        _report(runtime, payload, cid, ok=True, text="\n".join(out.messages))
        return TaskResult(success=True, data={"conversation_id": cid})


_NOTICE_CAP = 1000


def _report(runtime, payload, cid, *, ok: bool, text: str):
    """Queue the child's report on the requesting session's message queue.

    This is the agent-facing return channel of spawn_agent — deliberately
    separate from the user-facing conversation notification system
    (notification_mode / CHAT_MESSAGE_PUSHED), which spawned children have
    turned off.

    Best-effort: the session may be gone (skip silently — the transcript
    survives in conversation #cid). The notice is drained at the parent
    turn's next loop boundary, or at the start of its next turn. A child
    the parent already cancelled at its timeout queues nothing — the
    timeout failure is the authoritative report.
    """
    key = (payload.get("report_session_key") or "").strip()
    if not key:
        return
    if _consume_cancelled(runtime, payload, cid):
        return
    session = (getattr(runtime, "sessions", {}) or {}).get(key)
    if session is None:
        return
    # Deliver only to the conversation the spawn was made from: a session
    # that has since switched conversations must not receive the notice
    # (the result still lives in the child's own conversation #cid).
    target_cid = payload.get("report_conversation_id")
    if target_cid is not None and getattr(session, "conversation_id", None) != target_cid:
        return
    title = (payload.get("title") or "Subagent").strip()
    body = (text or "").strip()
    if len(body) > _NOTICE_CAP:
        body = body[:_NOTICE_CAP] + " ..."
    state = "finished" if ok else "FAILED"
    notice = (f"[Background agent '{title}' {state}] {body} "
              f"(full transcript: conversation #{cid})")
    try:
        with session.lock:
            session.pending_user_messages.append(notice)
    except Exception:
        pass


def _conversation_id(payload):
    """Internal helper to handle conversation ID."""
    try:
        return int(payload.get("conversation_id"))
    except (TypeError, ValueError):
        return None


def _create_conversation(db, payload):
    """Internal helper to create conversation."""
    tk = payload.get("_timekeeper") or {}
    title = (payload.get("title") or "Scheduled subagent").strip()
    cid = db.create_conversation(title, kind="user", category="Scheduled (one-time)" if tk.get("one_time") else "Scheduled")
    save_state_marker(db, cid, {"conversation_id": cid, "active_agent_profile": "default", "profile_override": "default"})
    return cid


def _remember_conversation(context, payload, cid):
    """Internal helper to handle remember conversation."""
    job_name = (payload.get("_timekeeper") or {}).get("job_name")
    tk = (getattr(context, "services", None) or {}).get("timekeeper")
    if not job_name or tk is None or not hasattr(tk, "update_job"):
        return
    job = tk.get_job(job_name)
    if job is not None:
        tk.update_job(job_name, {"payload": {**(job.get("payload") or {}), "conversation_id": cid}})


def _attachments(paths, context):
    """Internal helper to handle attachments."""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for raw in paths or []:
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Attachment not found: {path}")
        out.append(parse_attachment(str(path), services=getattr(context, "services", {}), config={"max_chars": 4000}))
    return out
