"""Run a background agent on a prompt, now — the immediate-run counterpart
to schedule_subagent.

The child runs in its own conversation via the spawn_subagent task. With
wait=true (default) this tool blocks until the child finishes and returns
its result; with wait=false the parent keeps working and a completion
notice is queued back to this session (the subagents service holds the
turn open at its end until pending children finish). The timeout is a hard
cutoff: a child that exceeds it is cancelled and reported as failed —
never silently backgrounded.
"""

dependencies_files = ['tasks/task_spawn_subagent.py', 'services/service_subagents.py']
dependencies_pip = []

import time
from pathlib import Path

from events.event_bus import bus
from plugins.BaseTool import BaseTool, ToolResult
from state_machine.serialization import save_state_marker
from ..tasks.task_spawn_subagent import SPAWN_SUBAGENT, cancelled_set

DEFAULT_TIMEOUT = 300
POLL_SECONDS = 1.0
TERMINAL_STATUSES = {"DONE", "FAILED"}


def find_run(db, cid) -> dict | None:
    """Locate the child's task_runs row by the conversation id in its payload.

    The payload is built with conversation_id as the first key and more keys
    after it, so the serialized form is unambiguous for an integer id.
    """
    like = f'%"conversation_id": {int(cid)},%'
    try:
        with db.lock:
            row = db.conn.execute(
                "SELECT run_id, status, error FROM task_runs "
                "WHERE task_name = 'spawn_subagent' AND payload_json LIKE ? "
                "ORDER BY created_at DESC LIMIT 1", (like,)).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return {"run_id": row[0], "status": row[1], "error": row[2]}


def cancel_child(runtime, cid) -> None:
    """Best-effort cancellation of the child session's turn."""
    child = (getattr(runtime, "sessions", {}) or {}).get(f"spawn_subagent:{cid}")
    event = getattr(child, "cancel_event", None)
    if event is not None:
        event.set()


def pending_map(session) -> dict:
    """The session's ephemeral {cid: deadline} map, created on demand."""
    current = getattr(session, "pending_subagents", None)
    if current is None:
        current = {}
        session.pending_subagents = current
    return current


class SpawnAgent(BaseTool):
    """Spawn agent."""
    name = "spawn_agent"
    description = (
        "Spawn an agent to work on a prompt in its own conversation, right now. "
        "The prompt must be complete and self-contained — the agent cannot ask you "
        "follow-up questions, and it cannot use tools that require user approval. "
        "wait=true (default) blocks and returns the agent's result; wait=false runs "
        "it in the background while you continue, and its completion notice arrives "
        "in this conversation before your turn ends. The timeout is a hard cutoff: "
        "an agent still running at its deadline is cancelled and reported as failed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "Complete, self-contained instructions for the agent."},
            "title": {"type": "string", "description": "Short title for the agent's conversation."},
            "attachments": {"type": "array", "items": {"type": "string"}, "description": "Optional file paths to attach."},
            "wait": {"type": "boolean", "description": "true (default): block and return the result. false: run in the background and continue."},
            "timeout_seconds": {"type": "integer", "description": "Max seconds the agent may run before it is cancelled. Capped by the subagent_timeout_seconds setting (default 300)."},
        },
        "required": ["prompt"],
    }
    requires_services = []
    max_calls = 5
    background_safe = True
    plan_mode_safe = False
    config_settings = [
        ("Subagent Timeout", "subagent_timeout_seconds",
         "Max seconds a spawned agent may run. A child still running at this deadline is cancelled and reported as failed.",
         DEFAULT_TIMEOUT, {"type": "integer"}),
    ]
    agent_prompt = (
        "## Spawning agents\n"
        "Use spawn_agent for long independent work or parallel research: wait=false lets "
        "you keep calling tools while children run; their results are delivered to you "
        "automatically — never poll for them. Results also persist in each child's own "
        "conversation. The timeout is a hard cutoff — children that exceed it are "
        "cancelled and reported as failed, so size each prompt to finish inside the "
        "budget. Only report results you actually received; a timed-out agent produced "
        "none."
    )

    def run(self, context, **kwargs) -> ToolResult:
        """Run spawn agent."""
        session_key = getattr(context, "session_key", "") or ""
        if session_key.startswith("spawn_subagent:"):
            return ToolResult.failed("spawn_agent cannot be used from a spawned agent (no recursive spawning).")

        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.failed("No prompt provided.")
        runtime, db = getattr(context, "runtime", None), getattr(context, "db", None)
        if runtime is None or db is None:
            return ToolResult.failed("ConversationRuntime and database are required.")

        attachments = [str(p).strip() for p in (kwargs.get("attachments") or []) if str(p).strip()]
        for raw in attachments:
            if not Path(raw).expanduser().is_file():
                return ToolResult.failed(f"Attachment not found: {raw}")

        ceiling = int((getattr(context, "config", None) or {}).get("subagent_timeout_seconds") or DEFAULT_TIMEOUT)
        timeout = min(int(kwargs.get("timeout_seconds") or ceiling), ceiling)
        title = (kwargs.get("title") or "Subagent").strip()

        cid = runtime.create_conversation(title, kind="user", category="Subagent",
                                          user_id=getattr(context, "user_id", 1))
        if cid is None:
            return ToolResult.failed("Could not create a conversation for the agent.")
        # notification_mode off: the child's output belongs to the parent agent
        # (delivered via the session message queue), never pushed to the user's
        # chat. Scheduled subagents keep the default "on" — the push is their
        # only delivery surface.
        save_state_marker(db, cid, {"conversation_id": cid, "active_agent_profile": "default",
                                    "profile_override": "default", "notification_mode": "off"})

        # conversation_id first: find_run() matches on this key's serialized position.
        bus.emit(SPAWN_SUBAGENT, {
            "conversation_id": cid,
            "title": title,
            "prompt": prompt,
            "attachments": attachments,
            "notify_session_key": session_key or None,
        })

        session = (getattr(runtime, "sessions", {}) or {}).get(session_key)
        if not kwargs.get("wait", True):
            if session is not None:
                pending_map(session)[cid] = time.time() + timeout
            return ToolResult(
                True, data={"conversation_id": cid, "wait": False},
                llm_summary=(f"Spawned background agent '{title}' in conversation #{cid}. "
                             "Keep working — its completion notice will be delivered to you "
                             "automatically; do not poll for it."))

        deadline = time.time() + timeout
        cancel_event = getattr(session, "cancel_event", None)
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                cancel_child(runtime, cid)
                return ToolResult.failed(f"Cancelled — agent stopped (partial transcript in conversation #{cid}).")
            run = find_run(db, cid)
            if run is not None and run["status"] in TERMINAL_STATUSES:
                return self._result(db, cid, title, run)
            time.sleep(POLL_SECONDS)

        # Hard cutoff: cancel the child and report failure. The cancelled set
        # suppresses the task's own completion notice for this cid so the
        # model never sees a stale "finished" echo after the failure.
        cancel_child(runtime, cid)
        if session is not None:
            cancelled_set(session).add(cid)
        return ToolResult.failed(
            f"Agent '{title}' timed out after {timeout}s and was cancelled — it produced no "
            f"result (partial transcript in conversation #{cid}). Retry with a smaller prompt "
            f"or a larger timeout_seconds.")

    @staticmethod
    def _result(db, cid, title, run) -> ToolResult:
        """Build the tool result from a finished child run."""
        if run["status"] == "FAILED":
            return ToolResult.failed(f"Agent '{title}' failed: {run.get('error') or 'unknown error'} "
                                     f"(conversation #{cid})")
        text = ""
        try:
            messages = db.get_conversation_messages(cid)
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and (msg.get("content") or "").strip():
                    text = msg["content"].strip()
                    break
        except Exception:
            pass
        return ToolResult(
            True, data={"conversation_id": cid, "wait": True},
            llm_summary=(text or "(the agent produced no final text)")
                        + f"\n\n(agent '{title}' ran in conversation #{cid})")
