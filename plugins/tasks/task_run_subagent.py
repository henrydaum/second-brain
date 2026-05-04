import json
import logging
import time
from pathlib import Path

from plugins.services.helpers.parser_registry import get_modality
from plugins.services.llmService import LLMRouter
from plugins.BaseTask import BaseTask, TaskResult
from agent.agent import Agent
from agent.history_utils import messages_to_history
from agent.system_prompt import build_system_prompt
from agent.tool_registry import ToolRegistry
from runtime.agent_scope import load_scope, resolve_agent_llm, scoped_registry
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
            "agent": {
                "type": "string",
                "description": "Optional agent profile name to run under. This lets callers choose a specialist agent profile with its own model and scope. Defaults to the active profile.",
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
        router = context.services.get("llm")
        if router is None or not getattr(router, "loaded", False):
            return TaskResult.failed("LLM service is not loaded.")

        if context.tool_registry is None:
            return TaskResult.failed("Tool registry is not available to background tasks.")

        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return TaskResult.failed("Subagent payload is missing 'prompt'.")

        # Resolve which agent profile this run should use. Empty or missing
        # "agent" falls back to the currently active profile.
        requested_agent = (payload.get("agent") or "").strip()
        target_agent = (
            requested_agent
            or context.config.get("active_agent_profile") or ""
        ).strip() or "default"

        agent_profiles = context.config.get("agent_profiles", {}) or {}
        if target_agent not in agent_profiles:
            return TaskResult.failed(f"Unknown agent profile: '{target_agent}'.")

        # Resolve the LLM the agent profile points at. Load it if necessary so
        # subagents pinning to a non-default LLM don't silently fall back.
        llm = resolve_agent_llm(target_agent, context.config, context.services)
        if llm is None:
            return TaskResult.failed(f"No LLM resolved for agent profile '{target_agent}'.")
        if not getattr(llm, "loaded", False):
            try:
                llm.load()
            except Exception as e:
                return TaskResult.failed(f"Failed to load agent profile '{target_agent}': {e}")
            if not getattr(llm, "loaded", False):
                return TaskResult.failed(f"Agent profile '{target_agent}' did not load.")

        try:
            scope = load_scope(target_agent, context.config)
        except ValueError as e:
            return TaskResult.failed(f"Invalid scope for agent '{target_agent}': {e}")

        sub_db = context.db

        title = (payload.get("title") or "").strip()
        timekeeper_meta = payload.get("_timekeeper") or {}
        job_name = (
            str(timekeeper_meta.get("job_name") or "").strip()
            or "scheduled_subagent"
        )
        conversation_title = title or job_name or prompt[:80].replace("\n", " ").strip() or "Scheduled subagent run"

        is_scheduled = bool(timekeeper_meta)

        # One-shot override from /message --notify=<mode> takes precedence over
        # the job's stored notifications mode for this run only. We clear it
        # before the agent runs so a crash mid-run doesn't leave it sticky.
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
                    logger.warning(
                        f"Failed to clear next_notifications for job '{job_name}': {e}"
                    )

        input_paths = self._normalize_input_paths(payload.get("input_paths"))
        input_rows, compiled_prompt, image_paths = self._compile_prompt(prompt, input_paths, context)
        push_records: list[SubagentPushRecord] = []

        # Scheduled jobs (anything coming via timekeeper) thread their runs
        # into a single persistent conversation so /message replies and prior
        # turns naturally appear in context. One-shot ask_subagent calls keep
        # the original behavior — fresh conversation per run.
        conversation_id, prior_history = self._resolve_conversation(
            context, is_scheduled, job_name, conversation_title,
        )

        # Did the user leave a /message reply since the last wake? If so, we
        # must guarantee a response back to chat regardless of mode.
        pending_user_messages = 0
        if is_scheduled:
            try:
                pending_user_messages = context.db.count_pending_inbox(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to count pending inbox for {job_name}: {e}")
        has_pending = pending_user_messages > 0

        # If the user explicitly asked something but the job is set to "off",
        # upgrade to "important" so the message tool is actually available.
        if has_pending and mode == "off":
            mode = "important"

        sub_registry = self._build_subagent_registry(
            context, run_id, job_name, push_records, scope, sub_db, mode,
        )

        # Scheduled runs persist the conversation in a single bulk rewrite at
        # the end (see replace_conversation_messages below), so any in-memory
        # compaction the agent does survives to the next wake. One-shot runs
        # use the per-message stream — they don't outlive the run.
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
            system_prompt=lambda: build_system_prompt(
                sub_db, context.orchestrator, sub_registry, context.services,
                scope=scope,
                profile_name=(scope.profile_name if scope else (context.config.get("active_agent_profile") or "default")),
                subagent_mode=mode,
                subagent_has_pending_messages=has_pending,
            ),
            on_message=(None if is_scheduled else _on_message),
        )
        if prior_history:
            agent.history = prior_history

        try:
            final_answer = agent.chat(compiled_prompt, image_paths=image_paths)
        except Exception as e:
            logger.error(f"Subagent run {run_id} failed: {e}", exc_info=True)
            # Leave /message replies pending so the next wake retries them
            # instead of silently dropping the user's question on a crash.
            return TaskResult.failed(str(e))

        # Drain /message replies only after a successful run — the agent has
        # actually seen them in prior_history and produced an answer.
        if is_scheduled:
            try:
                context.db.mark_inbox_consumed(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to mark inbox consumed for {job_name}: {e}")

        clean_answer, _ = strip_model_tokens(final_answer or "")
        if clean_answer.startswith("Error: no active LLM profile loaded"):
            return TaskResult.failed(clean_answer)

        # Persist the agent's final in-memory history. If chat() compacted
        # mid-run, this is what survives — the next wake reloads from here
        # instead of replaying the full uncompacted log.
        if is_scheduled:
            try:
                context.db.replace_conversation_messages(conversation_id, agent.history)
            except Exception as e:
                logger.warning(f"Failed to persist compacted history for {job_name}: {e}")

        # Fallback push: guarantee the user sees a response when either (a) the
        # job runs in "all" mode, or (b) the user just left a /message reply
        # — silence is never the right answer to a direct question.
        if (mode == "all" or has_pending) and not push_records and clean_answer.strip():
            self._emit_fallback_push(
                run_id, job_name, conversation_title, clean_answer, push_records,
            )

        self._persist_run(
            context, run_id, job_name, conversation_id, conversation_title, prompt,
            clean_answer, input_rows, push_records, target_agent,
        )
        return TaskResult(success=True)

    def _resolve_conversation(self, context, is_scheduled: bool, job_name: str,
                              conversation_title: str) -> tuple[int, list[dict]]:
        """Return (conversation_id, prior_history) for this run.

        Scheduled jobs reuse a stable conversation_id stored on the timekeeper
        job's payload so each wake-up sees prior turns and any /message replies
        Henry sent in between. The id is allocated lazily on first run.
        Non-scheduled runs get a fresh conversation, matching prior behavior.
        """
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
                rows = context.db.get_conversation_messages(conv_id)
                return conv_id, messages_to_history(rows)
            # Stale id (conversation was deleted) — fall through and re-allocate.

        conv_id = context.db.create_conversation(conversation_title[:200])
        if timekeeper is not None and getattr(timekeeper, "loaded", False):
            try:
                job = timekeeper.get_job(job_name)
                if job is not None:
                    new_payload = dict(job.get("payload") or {})
                    new_payload["conversation_id"] = conv_id
                    timekeeper.update_job(job_name, {"payload": new_payload})
            except Exception as e:
                logger.warning(
                    f"Failed to persist conversation_id for job '{job_name}': {e}"
                )
        return conv_id, []

    def _build_subagent_registry(self, context, run_id: str, job_name: str,
                                 push_records: list[SubagentPushRecord],
                                 scope, sub_db, mode: str) -> ToolRegistry:
        # Build from background-safe tools, then apply the profile's tool scope.
        base_registry = ToolRegistry(sub_db, context.config, context.services)
        base_registry.orchestrator = context.orchestrator
        base_registry.is_subagent = True
        for tool in context.tool_registry.tools.values():
            if not getattr(tool, "background_safe", True):
                continue
            base_registry.tools[tool.name] = tool

        registry = scoped_registry(base_registry, scope, db=sub_db)
        registry.orchestrator = context.orchestrator
        registry.is_subagent = True

        if mode != "off":
            def _record_push(record: SubagentPushRecord):
                push_records.append(record)

            # `message` is the user-visible channel; only register it when the
            # job's notification mode allows pushing.
            registry.register(MessageTool(run_id, job_name, _record_push))
        return registry

    def _emit_fallback_push(self, run_id: str, job_name: str, title: str,
                            final_answer: str, push_records: list[SubagentPushRecord]) -> None:
        message = final_answer.strip()
        push_title = (title or job_name or "").strip()
        sent_at = time.time()
        push_records.append(SubagentPushRecord(
            kind="brief",
            title=push_title,
            message=message,
            sent_at=sent_at,
        ))
        bus.emit(CHAT_MESSAGE_PUSHED, {
            "message": message,
            "title": push_title,
            "kind": "brief",
            "source": "subagent",
            "source_id": run_id,
            "job_name": job_name,
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
        run_row = (
            run_id,
            job_name,
            conversation_id,
            title,
            prompt,
            final_answer,
            agent_name,
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
                (run_id, job_name, conversation_id, title, prompt, final_answer, agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
