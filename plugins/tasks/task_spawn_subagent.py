from pathlib import Path

from attachments import parse_attachment
from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED, SPAWN_SUBAGENT
from plugins.BaseTask import BaseTask, TaskResult
from state_machine.serialization import save_state_marker


class SpawnSubagent(BaseTask):
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
        },
        "required": ["prompt"],
    }

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
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
            return TaskResult.failed(str(e))
        if not out.ok:
            return TaskResult.failed((out.error or {}).get("message") or "\n".join(out.messages) or "spawn_subagent failed.")
        return TaskResult(success=True, data={"conversation_id": cid})


def _conversation_id(payload):
    try:
        return int(payload.get("conversation_id"))
    except (TypeError, ValueError):
        return None


def _create_conversation(db, payload):
    tk = payload.get("_timekeeper") or {}
    title = (payload.get("title") or "Scheduled subagent").strip()
    cid = db.create_conversation(title, kind="user", category="Scheduled (one-time)" if tk.get("one_time") else "Scheduled")
    save_state_marker(db, cid, {"conversation_id": cid, "active_agent_profile": "default", "profile_override": "default"})
    return cid


def _remember_conversation(context, payload, cid):
    job_name = (payload.get("_timekeeper") or {}).get("job_name")
    tk = (getattr(context, "services", None) or {}).get("timekeeper")
    if not job_name or tk is None or not hasattr(tk, "update_job"):
        return
    job = tk.get_job(job_name)
    if job is not None:
        tk.update_job(job_name, {"payload": {**(job.get("payload") or {}), "conversation_id": cid}})


def _attachments(paths, context):
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for raw in paths or []:
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Attachment not found: {path}")
        out.append(parse_attachment(str(path), services=getattr(context, "services", {}), config={"max_chars": 4000}))
    return out
