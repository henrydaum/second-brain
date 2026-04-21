import json
import time
import uuid

from Stage_3.BaseTool import BaseTool, ToolResult
from Stage_3.subagent_runtime import SUBAGENT_RUN_CHANNEL
from event_bus import bus


class AskSubagent(BaseTool):
    name = "ask_subagent"
    description = (
        "Delegate a complex question to a separate subagent run, wait for it to do its own research with its own tool use, and return a concise final answer. "
        "Use this when the task is big enough to benefit from a focused pass, such as multi-step research, file synthesis, or comparing documents. "
        "Good use cases: 'read these files and summarize the key risks', 'research this topic across my files and the web', 'compare two documents and give me the main differences'. "
        "Do not use this for simple lookups or quick one-tool tasks. This tool is synchronous and blocking: it triggers the existing event-driven run_subagent task, waits for completion, and returns the stored result so it can be reviewed later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What the subagent should do. Be explicit about the goal, scope, and output format. Example: 'Compare these two files and give me the top 5 differences.'",
            },
            "input_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional local file paths to attach. Use this when the subagent should read or analyze specific files directly.",
            },
            "title": {
                "type": "string",
                "description": "Optional short title for the run. Useful for labeling the saved conversation and run history.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Maximum time to wait before giving up. Note: the parent agent's per-tool max_calls budget is usually a tighter ceiling than this timeout. Default 600.",
                "default": 600,
            },
            "poll_interval_seconds": {
                "type": "number",
                "description": "How often to check whether the subagent finished. Usually leave this alone. Default 1.0.",
                "default": 1.0,
            },
        },
        "required": ["prompt"],
    }
    requires_services = ["llm"]
    agent_enabled = True
    max_calls = 2
    background_safe = False

    def run(self, context, **kwargs):
        llm = context.services.get("llm")
        if llm is None or not getattr(llm, "loaded", False):
            return ToolResult.failed("LLM service is not loaded.")
        if getattr(llm, "active", None) is None and hasattr(llm, "active"):
            return ToolResult.failed("No active LLM profile is loaded.")
        if context.orchestrator is None:
            return ToolResult.failed("Orchestrator is not available.")
        if not bus.has_subscribers(SUBAGENT_RUN_CHANNEL):
            return ToolResult.failed("No subscriber is listening on subagent.run.")

        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult.failed("prompt is required.")

        input_paths = kwargs.get("input_paths") or []
        if not isinstance(input_paths, list):
            input_paths = [str(input_paths)]
        input_paths = [str(p) for p in input_paths if str(p).strip()]

        title = str(kwargs.get("title") or "").strip()
        timeout_seconds = self._coerce_timeout(kwargs.get("timeout_seconds"), default=600)
        poll_interval = self._coerce_poll(kwargs.get("poll_interval_seconds"), default=1.0)

        request_token = f"asksub:{uuid.uuid4().hex}"
        payload = {
            "prompt": prompt,
            "title": title,
            "job_name": "ask_subagent",
            "input_paths": input_paths,
            "_ask_subagent": {
                "request_token": request_token,
                "requested_at": time.time(),
            },
        }

        bus.emit(SUBAGENT_RUN_CHANNEL, payload)

        deadline = time.time() + timeout_seconds
        run_row = None
        while time.time() < deadline:
            run_row = self._find_task_run(context, request_token)
            if run_row is not None:
                break
            time.sleep(poll_interval)

        if run_row is None:
            return ToolResult.failed(
                f"Timed out waiting for the subagent run to be enqueued after {timeout_seconds}s."
            )

        run_id = run_row["run_id"]
        last_status = run_row["status"]

        while time.time() < deadline:
            run_row = self._get_task_run(context, run_id)
            if run_row is None:
                return ToolResult.failed(f"Subagent task run disappeared: {run_id}")
            last_status = str(run_row["status"] or "")
            if last_status == "DONE":
                return self._build_success_result(context, run_id, prompt, title, input_paths)
            if last_status == "FAILED":
                error_text = self._extract_error(run_row)
                return ToolResult.failed(f"Subagent run failed: {error_text}")
            time.sleep(poll_interval)

        return ToolResult.failed(
            f"Timed out waiting for subagent completion after {timeout_seconds}s. Last status: {last_status}. Run id: {run_id}"
        )

    def _build_success_result(self, context, run_id: str, prompt: str, title: str, input_paths: list[str]) -> ToolResult:
        sub_row = self._get_subagent_run(context, run_id)
        if sub_row is None:
            return ToolResult.failed(
                f"Subagent task run {run_id} completed but no row was found in subagent_runs."
            )

        final_answer = str(sub_row.get("final_answer") or "").strip()
        conversation_id = sub_row.get("conversation_id")
        pushes = self._get_pushes(context, run_id)
        inputs = self._get_inputs(context, run_id)

        concise = final_answer if len(final_answer) <= 1200 else final_answer[:1200].rstrip() + "\n[truncated]"
        summary_lines = [
            f"Subagent completed. Run id: {run_id}.",
        ]
        if conversation_id is not None:
            summary_lines.append(f"Conversation id: {conversation_id}.")
        summary_lines.append("Final answer:")
        summary_lines.append(concise or "[empty final answer]")

        return ToolResult(
            data={
                "run_id": run_id,
                "conversation_id": conversation_id,
                "title": title,
                "prompt": prompt,
                "input_paths": input_paths,
                "inputs": inputs,
                "pushes": pushes,
                "final_answer": final_answer,
            },
            llm_summary="\n".join(summary_lines),
        )

    def _find_task_run(self, context, request_token: str):
        sql = (
            "SELECT run_id, status, error, payload_json, started_at, finished_at "
            "FROM task_runs "
            "WHERE task_name = ? AND payload_json LIKE ? "
            "ORDER BY rowid DESC LIMIT 1"
        )
        like_pattern = f'%{request_token}%'
        with context.db.lock:
            cur = context.db.conn.execute(sql, ("run_subagent", like_pattern))
            row = cur.fetchone()
        return dict(row) if row else None

    def _get_task_run(self, context, run_id: str):
        with context.db.lock:
            cur = context.db.conn.execute(
                "SELECT run_id, status, error, payload_json, started_at, finished_at FROM task_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def _get_subagent_run(self, context, run_id: str):
        with context.db.lock:
            cur = context.db.conn.execute(
                "SELECT run_id, job_name, conversation_id, title, prompt, final_answer, created_at FROM subagent_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def _get_pushes(self, context, run_id: str):
        with context.db.lock:
            cur = context.db.conn.execute(
                "SELECT push_index, kind, title, message, sent_at FROM subagent_run_pushes WHERE run_id = ? ORDER BY push_index",
                (run_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def _get_inputs(self, context, run_id: str):
        with context.db.lock:
            cur = context.db.conn.execute(
                "SELECT input_index, path, modality FROM subagent_run_inputs WHERE run_id = ? ORDER BY input_index",
                (run_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _extract_error(run_row: dict) -> str:
        error_text = run_row.get("error")
        if error_text:
            return str(error_text)
        return "unknown error"

    @staticmethod
    def _coerce_timeout(value, default: int) -> int:
        try:
            timeout = int(value if value is not None else default)
        except Exception:
            timeout = default
        return max(5, min(timeout, 900))

    @staticmethod
    def _coerce_poll(value, default: float) -> float:
        try:
            interval = float(value if value is not None else default)
        except Exception:
            interval = default
        return max(0.2, min(interval, 5.0))
