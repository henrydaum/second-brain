import json
import logging
import time
from pathlib import Path

from plugins.services.helpers.parser_registry import get_modality
from plugins.BaseTask import BaseTask, TaskResult
from runtime.token_stripper import strip_model_tokens
from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED
from agent.subagent_runtime import (
    MessageTool,
    SubagentPushRecord,
    SUBAGENT_RUN_CHANNEL,
    SUBAGENT_NOTIFICATION_MODES,
    SUBAGENT_DEFAULT_NOTIFICATION_MODE,
)

logger = logging.getLogger("TaskRunSubagent")

_MAX_PARSED_CHARS = 4000


class RunSubagent(BaseTask):
    """Drive a scheduled subagent run through ConversationRuntime.

    Each scheduled subagent owns a state-machine session keyed by
    ``subagent:{job_name}``. Firing the timekeeper amounts to dispatching
    one ``send_text`` action on that session, which goes through the same
    enact() path as a user turn — same approvals, same forms, same cancel
    semantics. The MessageTool is registered as a per-session pinned tool
    so the agent's pushes still flow through CHAT_MESSAGE_PUSHED.
    """

    name = "run_subagent"
    trigger = "event"
    trigger_channels = [SUBAGENT_RUN_CHANNEL]
    event_payload_schema = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What the scheduled subagent should do."},
            "title": {"type": "string", "description": "Optional user-facing title for the run."},
            "job_name": {"type": "string", "description": "Optional stable internal name for the run."},
            "input_paths": {"type": "array", "description": "Optional list of file paths to include as inputs."},
            "agent": {"type": "string", "description": "Optional agent profile name to run under."},
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
            agent TEXT,
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
        runtime = getattr(context, "runtime", None)
        if runtime is None:
            return TaskResult.failed("ConversationRuntime is not wired into the orchestrator.")

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return TaskResult.failed("Subagent payload is missing 'prompt'.")

        # Resolve target agent profile.
        requested_agent = (payload.get("agent") or "").strip()
        target_agent = (requested_agent or context.config.get("active_agent_profile") or "").strip() or "default"
        agent_profiles = context.config.get("agent_profiles", {}) or {}
        if target_agent not in agent_profiles:
            return TaskResult.failed(f"Unknown agent profile: '{target_agent}'.")

        title = (payload.get("title") or "").strip()
        timekeeper_meta = payload.get("_timekeeper") or {}
        job_name = (str(timekeeper_meta.get("job_name") or "").strip() or "scheduled_subagent")
        conversation_title = title or job_name or prompt[:80].replace("\n", " ").strip() or "Scheduled subagent run"
        is_scheduled = bool(timekeeper_meta)

        mode = self._resolve_mode(payload, timekeeper_meta, context, is_scheduled, job_name)

        # Persistent conversation_id for scheduled jobs (so prior turns are
        # preserved). One-shot runs allocate fresh.
        conversation_id, prior_history = self._resolve_conversation(
            context, is_scheduled, job_name, conversation_title,
        )

        pending_user_messages = 0
        if is_scheduled:
            try:
                pending_user_messages = context.db.count_pending_inbox(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to count pending inbox for {job_name}: {e}")
        has_pending = pending_user_messages > 0
        if has_pending and mode == "off":
            mode = "important"

        push_records: list[SubagentPushRecord] = []
        message_tool = None
        if mode != "off":
            def _record_push(record: SubagentPushRecord):
                push_records.append(record)
            message_tool = MessageTool(run_id, job_name, _record_push)

        session_key = runtime.subagent_session_key(job_name)
        # Recreate the session each run so per-run extras (mode, pending
        # messages, conversation history) reflect the current invocation.
        runtime.create_subagent_session(
            session_key,
            agent_profile=target_agent,
            conversation_id=conversation_id,
            extra_tool_instances=[message_tool] if message_tool else [],
            subagent_meta={
                "job_name": job_name,
                "run_id": run_id,
                "is_scheduled": is_scheduled,
            },
            system_prompt_extras={
                "subagent_mode": mode,
                "subagent_has_pending_messages": has_pending,
            },
            history=prior_history,
        )

        input_paths = self._normalize_input_paths(payload.get("input_paths"))
        input_rows, compiled_prompt, image_paths = self._compile_prompt(prompt, input_paths, context)

        # Drive one user turn through the state machine. The runtime takes
        # care of history persistence, agent loop, approvals, and forms.
        action_payload = {"text": compiled_prompt}
        if image_paths:
            action_payload["image_paths"] = image_paths
        try:
            result = runtime.handle_action(session_key, "send_text", action_payload)
        except Exception as e:
            logger.error(f"Subagent run {run_id} failed: {e}", exc_info=True)
            return TaskResult.failed(str(e))

        if not result.ok:
            err = (result.error or {}).get("message") or "Subagent run failed."
            return TaskResult.failed(err)

        # Drain /message replies once the run produced an answer.
        if is_scheduled:
            try:
                context.db.mark_inbox_consumed(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to mark inbox consumed for {job_name}: {e}")

        # Persist the (possibly compacted) in-memory history so the next
        # wake reloads from the rewritten log instead of the full transcript.
        if is_scheduled:
            session = runtime.sessions.get(session_key)
            if session is not None:
                try:
                    context.db.replace_conversation_messages(conversation_id, list(session.history))
                except Exception as e:
                    logger.warning(f"Failed to persist compacted history for {job_name}: {e}")

        final_answer = "\n".join(m for m in result.messages if m).strip()
        clean_answer, _ = strip_model_tokens(final_answer)
        if clean_answer.startswith("Error: no active LLM profile loaded"):
            return TaskResult.failed(clean_answer)

        # Fallback push: in 'all' mode or when the user left a /message reply,
        # guarantee a single user-visible response.
        if (mode == "all" or has_pending) and not push_records and clean_answer.strip():
            self._emit_fallback_push(run_id, job_name, conversation_title, clean_answer, push_records)

        self._persist_run(
            context, run_id, job_name, conversation_id, conversation_title, prompt,
            clean_answer, input_rows, push_records, target_agent,
        )
        return TaskResult(success=True)

    # ──────────────────────────────────────────────────────────────────────

    def _resolve_mode(self, payload: dict, timekeeper_meta: dict, context, is_scheduled: bool, job_name: str) -> str:
        override_mode_raw = payload.get("next_notifications")
        override_mode = None
        if override_mode_raw is not None:
            candidate = str(override_mode_raw).strip().lower()
            if candidate in SUBAGENT_NOTIFICATION_MODES:
                override_mode = candidate

        if override_mode is not None:
            mode = override_mode
        else:
            mode = (payload.get("notifications") or SUBAGENT_DEFAULT_NOTIFICATION_MODE)
            mode = str(mode).strip().lower()
            if mode not in SUBAGENT_NOTIFICATION_MODES:
                mode = SUBAGENT_DEFAULT_NOTIFICATION_MODE

        # One-shot --notify override is consumed once.
        if is_scheduled and override_mode_raw is not None:
            timekeeper = context.services.get("timekeeper")
            if timekeeper is not None and getattr(timekeeper, "loaded", False):
                try:
                    job = timekeeper.get_job(job_name)
                    if job is not None:
                        cleared = dict(job.get("payload") or {})
                        cleared.pop("next_notifications", None)
                        timekeeper.update_job(job_name, {"payload": cleared})
                except Exception as e:
                    logger.warning(f"Failed to clear next_notifications for job '{job_name}': {e}")
        return mode

    def _resolve_conversation(self, context, is_scheduled: bool, job_name: str,
                              conversation_title: str) -> tuple[int, list[dict]]:
        if not is_scheduled:
            return context.db.create_conversation(conversation_title[:200]), []

        timekeeper = context.services.get("timekeeper")
        existing = None
        if timekeeper is not None and getattr(timekeeper, "loaded", False):
            job = timekeeper.get_job(job_name)
            if job is not None:
                existing = (job.get("payload") or {}).get("conversation_id")

        if existing is not None:
            try:
                conv_id = int(existing)
            except (TypeError, ValueError):
                conv_id = None
            if conv_id is not None and context.db.get_conversation(conv_id) is not None:
                from agent.history_utils import messages_to_history
                rows = context.db.get_conversation_messages(conv_id)
                return conv_id, messages_to_history(rows)

        conv_id = context.db.create_conversation(conversation_title[:200])
        if timekeeper is not None and getattr(timekeeper, "loaded", False):
            try:
                job = timekeeper.get_job(job_name)
                if job is not None:
                    new_payload = dict(job.get("payload") or {})
                    new_payload["conversation_id"] = conv_id
                    timekeeper.update_job(job_name, {"payload": new_payload})
            except Exception as e:
                logger.warning(f"Failed to persist conversation_id for job '{job_name}': {e}")
        return conv_id, []

    def _emit_fallback_push(self, run_id: str, job_name: str, title: str,
                            final_answer: str, push_records: list[SubagentPushRecord]) -> None:
        message = final_answer.strip()
        push_title = (title or job_name or "").strip()
        sent_at = time.time()
        push_records.append(SubagentPushRecord(
            kind="brief", title=push_title, message=message, sent_at=sent_at,
        ))
        bus.emit(CHAT_MESSAGE_PUSHED, {
            "message": message, "title": push_title, "kind": "brief",
            "source": "subagent", "source_id": run_id, "job_name": job_name,
        })

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
            input_rows.append({"run_id": None, "input_index": idx, "path": path, "modality": modality})

            if modality == "image":
                image_paths.append(path)
                prompt_parts.append(f"\n[Input image: {path}]")
                continue
            if modality in ("text", "tabular"):
                prompt_parts.append(self._parse_input_file(path, modality, context))
                continue
            prompt_parts.append(f"\n[Input file: {path} (type: {modality}). This file type is not auto-parsed for the subagent.]")

        return input_rows, "\n".join(prompt_parts).strip(), image_paths

    def _parse_input_file(self, path: str, modality: str, context) -> str:
        try:
            result = context.services.get("parser").parse(path, modality, config={"max_chars": _MAX_PARSED_CHARS})
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
                     prompt: str, final_answer: str, input_rows: list[dict],
                     push_records: list[SubagentPushRecord], agent_name: str):
        run_row = (run_id, job_name, conversation_id, title, prompt, final_answer, agent_name, time.time())
        filled_inputs = [(run_id, row["input_index"], row["path"], row["modality"]) for row in input_rows]
        push_rows = [
            (run_id, idx, record.kind, record.title, record.message, record.sent_at)
            for idx, record in enumerate(push_records)
        ]
        with context.db.lock:
            context.db.conn.execute(
                """
                INSERT OR REPLACE INTO subagent_runs
                (run_id, job_name, conversation_id, title, prompt, final_answer, agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_row,
            )
            context.db.conn.execute("DELETE FROM subagent_run_inputs WHERE run_id = ?", (run_id,))
            context.db.conn.execute("DELETE FROM subagent_run_pushes WHERE run_id = ?", (run_id,))
            if filled_inputs:
                context.db.conn.executemany(
                    "INSERT INTO subagent_run_inputs (run_id, input_index, path, modality) VALUES (?, ?, ?, ?)",
                    filled_inputs,
                )
            if push_rows:
                context.db.conn.executemany(
                    "INSERT INTO subagent_run_pushes (run_id, push_index, kind, title, message, sent_at) VALUES (?, ?, ?, ?, ?, ?)",
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
