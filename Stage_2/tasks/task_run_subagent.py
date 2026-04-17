import json
import logging
import time
from pathlib import Path

from Stage_1.registry import get_modality
from Stage_2.BaseTask import BaseTask, TaskResult
from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from Stage_3.tool_registry import ToolRegistry
from frontend.token_stripper import strip_model_tokens
from Stage_3.subagent_runtime import (
    PushSubagentMessageTool,
    SubagentPushRecord,
    SUBAGENT_RUN_CHANNEL,
)

logger = logging.getLogger("TaskRunSubagent")

_MAX_PARSED_CHARS = 4000


class RunSubagent(BaseTask):
    name = "run_subagent"
    trigger = "event"
    trigger_channels = [SUBAGENT_RUN_CHANNEL]
    event_payload_schema = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What the scheduled subagent should do.",
            },
            "title": {
                "type": "string",
                "description": "Optional user-facing title for the run.",
            },
            "job_name": {
                "type": "string",
                "description": "Optional stable internal name for the run.",
            },
            "input_paths": {
                "type": "array",
                "description": "Optional list of file paths to include as inputs.",
            },
        },
        "required": ["prompt"],
    }
    requires_services = ["llm"]
    writes = []
    output_schema = """
        CREATE TABLE IF NOT EXISTS subagent_runs (
            run_id TEXT PRIMARY KEY,
            job_name TEXT,
            conversation_id INTEGER,
            title TEXT,
            prompt TEXT,
            final_answer TEXT,
            created_at REAL
        );

        CREATE TABLE IF NOT EXISTS subagent_run_inputs (
            run_id TEXT,
            input_index INTEGER,
            path TEXT,
            modality TEXT,
            PRIMARY KEY (run_id, input_index)
        );

        CREATE TABLE IF NOT EXISTS subagent_run_pushes (
            run_id TEXT,
            push_index INTEGER,
            kind TEXT,
            title TEXT,
            message TEXT,
            sent_at REAL,
            PRIMARY KEY (run_id, push_index)
        );
    """
    timeout = 900

    def run_event(self, run_id: str, payload: dict, context) -> TaskResult:
        llm = context.services.get("llm")
        if llm is None or not getattr(llm, "loaded", False):
            return TaskResult.failed("LLM service is not loaded.")

        if getattr(llm, "active", None) is None and hasattr(llm, "active"):
            return TaskResult.failed("No active LLM profile is loaded.")

        if context.tool_registry is None:
            return TaskResult.failed("Tool registry is not available to background tasks.")

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return TaskResult.failed("Subagent payload is missing 'prompt'.")

        title = (payload.get("title") or "").strip()
        timekeeper_meta = payload.get("_timekeeper") or {}
        job_name = (
            (payload.get("job_name") or "").strip()
            or str(timekeeper_meta.get("job_name") or "").strip()
            or "scheduled_subagent"
        )
        conversation_title = title or job_name or prompt[:80].replace("\n", " ").strip() or "Scheduled subagent run"

        input_paths = self._normalize_input_paths(payload.get("input_paths"))
        input_rows, compiled_prompt, image_paths = self._compile_prompt(prompt, input_paths, context)
        push_records: list[SubagentPushRecord] = []

        sub_registry = self._build_subagent_registry(context, run_id, job_name, push_records)
        conversation_id = context.db.create_conversation(conversation_title[:200])

        def _on_message(msg: dict):
            role = msg.get("role", "")
            if role not in {"user", "assistant", "tool"}:
                return
            content = msg.get("content") or ""
            save_content = content
            if msg.get("tool_calls"):
                save_content = json.dumps({
                    "content": content,
                    "tool_calls": msg["tool_calls"],
                })
            context.db.save_message(
                conversation_id,
                role,
                save_content,
                tool_call_id=msg.get("tool_call_id"),
                tool_name=msg.get("name"),
            )

        agent = Agent(
            llm,
            sub_registry,
            context.config,
            system_prompt=lambda: self._build_subagent_prompt(context, sub_registry),
            on_message=_on_message,
        )

        try:
            final_answer = agent.chat(compiled_prompt, image_paths=image_paths)
        except Exception as e:
            logger.error(f"Subagent run {run_id} failed: {e}", exc_info=True)
            return TaskResult.failed(str(e))

        clean_answer, _ = strip_model_tokens(final_answer or "")
        if clean_answer.startswith("Error: no active LLM profile loaded"):
            return TaskResult.failed(clean_answer)

        self._persist_run(context, run_id, job_name, conversation_id, conversation_title, prompt, clean_answer, input_rows, push_records)
        return TaskResult(success=True)

    def _build_subagent_registry(self, context, run_id: str, job_name: str, push_records: list[SubagentPushRecord]) -> ToolRegistry:
        registry = ToolRegistry(context.db, context.config, context.services)
        registry.orchestrator = context.orchestrator

        for tool in context.tool_registry.tools.values():
            if getattr(tool, "agent_enabled", False) and getattr(tool, "background_safe", True):
                registry.register(tool)

        def _record_push(record: SubagentPushRecord):
            push_records.append(record)

        registry.register(PushSubagentMessageTool(run_id, job_name, _record_push))
        return registry

    def _build_subagent_prompt(self, context, sub_registry) -> str:
        base = build_system_prompt(context.db, context.orchestrator, sub_registry, context.services)
        return (
            base
            + "\n\n## Scheduled subagent\n"
            + "You are running unattended on a schedule.\n"
            + "Work as if no one will answer follow-up questions during this run.\n"
            + "Do not rely on permission dialogs or back-and-forth clarification.\n"
            + "Do not ask questions.\n"
            + "push_subagent_message is the main way to send a user-visible message during the run.\n"
            + "If you do not call push_subagent_message, the run is mostly invisible to the user aside from logs and stored history.\n"
            + "Use push_subagent_message for reminders, alerts, briefs, findings, check-ins, or anything the user should actually see in chat.\n"
            + "If the user asked to be reminded, notified, briefed, or updated, you should usually call push_subagent_message before finishing.\n"
            + "Stay silent only when the job is intentionally background-only and no user-facing update is needed.\n"
            + "Do not use update_memory as a substitute for messaging the user. Memory is for durable preferences, standing instructions, and long-term context that should shape future behavior across sessions.\n"
            + "A one-off reminder, brief, nudge, or status update belongs in push_subagent_message, not memory.\n"
            + "Finish with a concise final answer that can be stored and reviewed later.\n"            
        )

    def _compile_prompt(self, prompt: str, input_paths: list[str], context):
        prompt_parts = [prompt]
        input_rows = []
        image_paths = []

        for idx, raw_path in enumerate(input_paths):
            path = str(raw_path).strip()
            if not path:
                continue

            ext = Path(path).suffix.lower()
            modality = get_modality(ext) if ext else "unknown"
            input_rows.append({
                "run_id": None,  # filled during insert
                "input_index": idx,
                "path": path,
                "modality": modality,
            })

            if modality == "image":
                image_paths.append(path)
                prompt_parts.append(f"\n[Input image: {path}]")
                continue

            if modality in ("text", "tabular"):
                parsed_block = self._parse_input_file(path, modality, context)
                prompt_parts.append(parsed_block)
                continue

            if modality in ("audio", "video", "container"):
                prompt_parts.append(
                    f"\n[Input file: {path} ({modality}). This file type is not auto-parsed for the subagent.]"
                )
                continue

            prompt_parts.append(
                f"\n[Input file: {path} (type: {modality}). This file type is not auto-parsed for the subagent.]"
            )

        return input_rows, "\n".join(prompt_parts).strip(), image_paths

    def _parse_input_file(self, path: str, modality: str, context) -> str:
        try:
            result = context.parse(path, modality, config={"max_chars": _MAX_PARSED_CHARS})
        except Exception as e:
            return f"\n[Input file: {path}. Parsing failed: {e}]"

        if not getattr(result, "success", False):
            return f"\n[Input file: {path}. Parsing failed: {getattr(result, 'error', 'unknown error')}]"

        output = getattr(result, "output", None)
        if output is None:
            return f"\n[Input file: {path}. No parsed content was produced.]"

        if isinstance(output, str):
            raw = output
        elif isinstance(output, dict):
            df = output.get("default")
            raw = df.to_string(max_rows=50) if df is not None else str(output)
        else:
            raw = str(output)

        if len(raw) > _MAX_PARSED_CHARS:
            raw = raw[:_MAX_PARSED_CHARS] + "\n[Content truncated]"

        return f"\n[Input file: {path}]\n{raw}"

    def _persist_run(self, context, run_id: str, job_name: str, conversation_id: int, title: str,
                     prompt: str, final_answer: str, input_rows: list[dict], push_records: list[SubagentPushRecord]):
        run_row = (
            run_id,
            job_name,
            conversation_id,
            title,
            prompt,
            final_answer,
            time.time(),
        )
        filled_inputs = [
            (run_id, row["input_index"], row["path"], row["modality"])
            for row in input_rows
        ]
        push_rows = [
            (run_id, idx, record.kind, record.title, record.message, record.sent_at)
            for idx, record in enumerate(push_records)
        ]

        with context.db.lock:
            context.db.conn.execute(
                """
                INSERT OR REPLACE INTO subagent_runs
                (run_id, job_name, conversation_id, title, prompt, final_answer, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                run_row,
            )
            context.db.conn.execute("DELETE FROM subagent_run_inputs WHERE run_id = ?", (run_id,))
            context.db.conn.execute("DELETE FROM subagent_run_pushes WHERE run_id = ?", (run_id,))
            if filled_inputs:
                context.db.conn.executemany(
                    """
                    INSERT INTO subagent_run_inputs
                    (run_id, input_index, path, modality)
                    VALUES (?, ?, ?, ?)
                    """,
                    filled_inputs,
                )
            if push_rows:
                context.db.conn.executemany(
                    """
                    INSERT INTO subagent_run_pushes
                    (run_id, push_index, kind, title, message, sent_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    push_rows,
                )
            context.db.conn.commit()

    @staticmethod
    def _normalize_input_paths(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
