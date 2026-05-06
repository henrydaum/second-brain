import logging
import time
from pathlib import Path

from plugins.services.helpers.parser_registry import get_modality
from plugins.BaseTask import BaseTask, TaskResult
from runtime.token_stripper import strip_model_tokens
from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED, SESSION_TURN_COMPLETED, SUBAGENT_RUN
from plugins.tools.tool_notify import NotifyTool, NotificationRecord
from plugins.tasks.helpers.notifications import NOTIFICATION_MODES, notification_mode

logger = logging.getLogger("TaskRunSubagent")

_MAX_PARSED_CHARS = 4000


class RunSubagent(BaseTask):
    """Drive a scheduled subagent run through ConversationRuntime.

    Each scheduled subagent owns a state-machine session keyed by
    ``subagent:{job_name}``. Firing the timekeeper amounts to dispatching
    one ``send_text`` action on that session, which goes through the same
    enact() path as a user turn — same approvals, same forms, same cancel
    semantics. The NotifyTool is registered as a per-session pinned tool
    so the agent's pushes still flow through CHAT_MESSAGE_PUSHED.
    """

    name = "run_subagent"
    trigger = "event"
    trigger_channels = [SUBAGENT_RUN]
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

        conversation_id = self._resolve_conversation(
            context, runtime, payload, is_scheduled, job_name, conversation_title,
        )
        if is_scheduled and job_name and conversation_id is not None:
            try:
                context.db.set_conversation_origin(conversation_id, f"cron:{job_name}")
            except Exception as e:
                logger.warning(f"Failed to stamp origin for {job_name}: {e}")

        pending_user_messages = 0
        if is_scheduled:
            try:
                pending_user_messages = context.db.count_pending_inbox(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to count pending inbox for {job_name}: {e}")
        has_pending = pending_user_messages > 0
        if has_pending and mode == "off":
            mode = "important"

        push_records: list[NotificationRecord] = []
        notify_tool = None
        if mode != "off":
            def _record_push(record: NotificationRecord):
                push_records.append(record)
            notify_tool = NotifyTool(source="subagent", source_id=run_id, job_name=job_name, recorder=_record_push)

        session_key = runtime.subagent_session_key(job_name)
        input_paths = self._normalize_input_paths(payload.get("input_paths"))
        compiled_prompt, image_paths = self._compile_prompt(prompt, input_paths, context)

        turn_events = []
        unsub = bus.subscribe(SESSION_TURN_COMPLETED, lambda event: turn_events.append(event) if (event or {}).get("session_key") == session_key else None)
        try:
            runtime.load_conversation(
                session_key, conversation_id, agent_profile=target_agent,
                system_prompt_extras={
                    "subagent_mode": mode,
                    "subagent_has_pending_messages": has_pending,
                    "subagent_run_id": run_id,
                    "subagent_job_name": job_name,
                },
            )
            if notify_tool:
                runtime.add_session_tool(session_key, notify_tool)
            result = runtime.iterate_agent_turn(session_key, compiled_prompt, image_paths=image_paths)
        except Exception as e:
            logger.error(f"Subagent run {run_id} failed: {e}", exc_info=True)
            return TaskResult.failed(str(e))
        finally:
            unsub()
            runtime.unload_conversation(session_key)

        if not result.ok:
            err = (result.error or {}).get("message") or "Subagent run failed."
            return TaskResult.failed(err)

        # Drain /message replies once the run produced an answer.
        if is_scheduled:
            try:
                context.db.mark_inbox_consumed(conversation_id)
            except Exception as e:
                logger.warning(f"Failed to mark inbox consumed for {job_name}: {e}")

        final_answer = ((turn_events[-1] or {}).get("final_text") if turn_events else "") or "\n".join(m for m in result.messages if m).strip()
        clean_answer, _ = strip_model_tokens(final_answer)
        if clean_answer.startswith("Error: no active LLM profile loaded"):
            return TaskResult.failed(clean_answer)

        # Fallback push: in 'all' mode or when the user left a /message reply,
        # guarantee a single user-visible response.
        if (mode == "all" or has_pending) and not push_records and clean_answer.strip():
            self._emit_fallback_push(run_id, job_name, conversation_title, clean_answer, push_records)

        return TaskResult(success=True)

    # ──────────────────────────────────────────────────────────────────────

    def _resolve_mode(self, payload: dict, timekeeper_meta: dict, context, is_scheduled: bool, job_name: str) -> str:
        override_mode_raw = payload.get("next_notifications")
        override_mode = None
        if override_mode_raw is not None:
            candidate = str(override_mode_raw).strip().lower()
            if candidate in NOTIFICATION_MODES:
                override_mode = candidate

        if override_mode is not None:
            mode = override_mode
        else:
            mode = notification_mode(payload.get("notifications"))

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

    def _resolve_conversation(self, context, runtime, payload: dict, is_scheduled: bool, job_name: str,
                              conversation_title: str) -> int:
        existing = payload.get("conversation_id")
        if not is_scheduled and existing is not None:
            try:
                conv_id = int(existing)
            except (TypeError, ValueError):
                conv_id = None
            if conv_id is not None and context.db.get_conversation(conv_id) is not None:
                return conv_id
        origin = f"cron:{job_name}" if is_scheduled and job_name else None
        if not is_scheduled:
            return runtime.create_conversation(conversation_title[:200], kind="subagent")

        timekeeper = context.services.get("timekeeper")
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
                return conv_id

        conv_id = runtime.create_conversation(conversation_title[:200], kind="subagent", origin=origin)
        if timekeeper is not None and getattr(timekeeper, "loaded", False):
            try:
                job = timekeeper.get_job(job_name)
                if job is not None:
                    new_payload = dict(job.get("payload") or {})
                    new_payload["conversation_id"] = conv_id
                    timekeeper.update_job(job_name, {"payload": new_payload})
            except Exception as e:
                logger.warning(f"Failed to persist conversation_id for job '{job_name}': {e}")
        return conv_id

    def _emit_fallback_push(self, run_id: str, job_name: str, title: str,
                            final_answer: str, push_records: list[NotificationRecord]) -> None:
        message = final_answer.strip()
        push_title = (title or job_name or "").strip()
        sent_at = time.time()
        push_records.append(NotificationRecord(
            kind="brief", title=push_title, message=message, sent_at=sent_at,
        ))
        bus.emit(CHAT_MESSAGE_PUSHED, {
            "message": message, "title": push_title, "kind": "brief",
            "source": "subagent", "source_id": run_id, "job_name": job_name,
        })

    def _compile_prompt(self, prompt: str, input_paths: list[str], context):
        prompt_parts = [prompt]
        image_paths = []

        for idx, raw_path in enumerate(input_paths):
            path = str(raw_path).strip()
            if not path:
                continue
            ext = Path(path).suffix.lower()
            modality = get_modality(ext) if ext else "unknown"

            if modality == "image":
                image_paths.append(path)
                prompt_parts.append(f"\n[Input image: {path}]")
                continue
            if modality in ("text", "tabular"):
                prompt_parts.append(self._parse_input_file(path, modality, context))
                continue
            prompt_parts.append(f"\n[Input file: {path} (type: {modality}). This file type is not auto-parsed for the subagent.]")

        return "\n".join(prompt_parts).strip(), image_paths

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

    @staticmethod
    def _normalize_input_paths(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
