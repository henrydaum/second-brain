import logging
import time
from pathlib import Path

from plugins.BaseTask import BaseTask, TaskResult
from runtime.token_stripper import strip_model_tokens
from events.event_bus import bus
from events.event_channels import CHAT_MESSAGE_PUSHED, SESSION_TURN_COMPLETED, SUBAGENT_RUN
from plugins.tools.tool_notify import NotifyTool, NotificationRecord
from plugins.tasks.helpers.notifications import NOTIFICATION_MODES, category_for_job, notification_mode
from state_machine.serialization import latest_state

logger = logging.getLogger("TaskRunSubagent")

_MAX_PARSED_CHARS = 4000


class RunSubagent(BaseTask):
    """Drive a scheduled subagent run through ConversationRuntime.

    Each scheduled subagent owns a state-machine session keyed by
    ``subagent:{job_name}``. Firing the timekeeper amounts to dispatching
    one ``send_text`` action on that session, which goes through the same
    enact() path as a user turn — same approvals, same forms, same cancel
    semantics.

    Conversation binding: every job carries a concrete ``conversation_id``
    (eagerly created at scheduling time, or supplied explicitly). The agent
    profile is read from that conversation's state marker — there is no
    per-job ``agent`` parameter. The NotifyTool is only attached when the
    chosen conversation is **not** the user's currently-focused one;
    otherwise the cron's output is already visible to the user in their
    session.
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
            "conversation_id": {
                "description": "Integer id of the conversation this run writes into.",
            },
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

        title = (payload.get("title") or "").strip()
        timekeeper_meta = payload.get("_timekeeper") or {}
        explicit_name = (
            str(timekeeper_meta.get("job_name") or "").strip()
            or str(payload.get("job_name") or "").strip()
        )
        job_name = explicit_name or f"adhoc:{run_id}"
        conversation_title = title or job_name or prompt[:80].replace("\n", " ").strip() or "Scheduled subagent run"
        is_scheduled = bool(timekeeper_meta)
        is_one_time = bool(timekeeper_meta.get("one_time"))

        if timekeeper_meta.get("source"):
            source = str(timekeeper_meta.get("source"))
        elif payload.get("_ask_subagent"):
            source = "ask_subagent"
        else:
            source = "unknown"

        mode = self._resolve_mode(payload, timekeeper_meta, context, is_scheduled, job_name)

        conversation_id, treat_as_active = self._resolve_conversation(
            context, runtime, payload, is_scheduled, is_one_time, job_name, conversation_title,
        )
        if conversation_id is None:
            return TaskResult.failed("Could not resolve a conversation for this subagent run.")

        target_agent = self._resolve_agent_from_conversation(context, conversation_id)
        agent_profiles = context.config.get("agent_profiles", {}) or {}
        if target_agent not in agent_profiles:
            return TaskResult.failed(f"Unknown agent profile: '{target_agent}'.")

        # Handoff: when the target conversation is the user's currently-
        # focused one, run under their session_key instead of opening a
        # parallel subagent session. Single session, single lock, messages
        # stream live to the frontend, and no fallback push is needed.
        active_key = runtime.active_session_key
        handoff = bool(treat_as_active and active_key)
        session_key = active_key if handoff else runtime.subagent_session_key(job_name)

        conv_row = context.db.get_conversation(conversation_id) if context.db else None
        conv_title = (conv_row or {}).get("title") or "(untitled)"
        conv_category = (conv_row or {}).get("category") or "Main"
        logger.info(
            f"Subagent fire: source={source} job='{job_name}' run_id={run_id} "
            f"session={session_key} → conversation #{conversation_id} "
            f"'{conv_title}' [{conv_category}] mode={mode} handoff={handoff}"
        )

        push_records: list[NotificationRecord] = []
        notify_tool = None
        # NotifyTool only lives on cron sessions that are not the user's active
        # conversation. When ``treat_as_active`` is True the run's output is
        # already streaming into the user's foreground session — a notify push
        # would just duplicate it.
        if mode != "off" and not treat_as_active:
            def _record_push(record: NotificationRecord):
                push_records.append(record)
            notify_tool = NotifyTool(source="subagent", source_id=run_id, job_name=job_name, recorder=_record_push, conversation_id=conversation_id)

        input_paths = self._normalize_input_paths(payload.get("input_paths"))
        compiled_prompt, sub_attachments = self._compile_prompt(prompt, input_paths, context)

        turn_events = []
        unsub = bus.subscribe(SESSION_TURN_COMPLETED, lambda event: turn_events.append(event) if (event or {}).get("session_key") == session_key else None)
        try:
            if not handoff:
                runtime.load_conversation(
                    session_key, conversation_id, agent_profile=target_agent,
                    system_prompt_extras={
                        "subagent_mode": mode,
                        "subagent_run_id": run_id,
                        "subagent_job_name": job_name,
                    },
                )
                if notify_tool:
                    runtime.add_session_tool(session_key, notify_tool)
            result = runtime.iterate_agent_turn(session_key, compiled_prompt, attachments=sub_attachments)
        except Exception as e:
            logger.error(f"Subagent run {run_id} failed: {e}", exc_info=True)
            return TaskResult.failed(str(e))
        finally:
            unsub()
            if not handoff:
                runtime.unload_conversation(session_key)

        if not result.ok:
            err = (result.error or {}).get("message") or "Subagent run failed."
            return TaskResult.failed(err)

        final_answer = ((turn_events[-1] or {}).get("final_text") if turn_events else "") or "\n".join(m for m in result.messages if m).strip()
        clean_answer, _ = strip_model_tokens(final_answer)
        if clean_answer.startswith("Error: no active LLM profile loaded"):
            return TaskResult.failed(clean_answer)

        # Fallback push: in 'all' mode, guarantee a single user-visible
        # response. Skipped when the run is already in the user's active
        # conversation — they have already seen the agent's reply, no
        # separate push needed.
        if mode == "all" and not push_records and not treat_as_active and clean_answer.strip():
            self._emit_fallback_push(run_id, job_name, conversation_title, clean_answer, push_records, context, conversation_id)

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

    def _resolve_conversation(self, context, runtime, payload: dict, is_scheduled: bool,
                              is_one_time: bool, job_name: str,
                              conversation_title: str) -> tuple[int | None, bool]:
        """Pick the conversation this run should write into.

        Returns ``(conversation_id, treat_as_active)``. ``treat_as_active`` is
        True when the chosen conversation matches the user's currently active
        conversation — used downstream to skip NotifyTool registration.
        """
        active_id = runtime.active_conversation_id
        existing = payload.get("conversation_id")

        if existing is not None:
            try:
                conv_id = int(existing)
            except (TypeError, ValueError):
                conv_id = None
            if conv_id is not None and context.db.get_conversation(conv_id) is not None:
                return conv_id, conv_id == active_id
            # Stored id refers to a deleted conversation (or was malformed).
            # Fall through and mint a fresh one — the timekeeper-persist
            # branch below will rewrite the job's payload so subsequent fires
            # use the new conversation.

        # No usable id on the payload — eagerly create one. ask_subagent
        # (non-scheduled) makes throwaway "Subagent" conversations; scheduled
        # jobs split into recurring vs one-time buckets.
        category = category_for_job(is_scheduled, is_one_time)
        conv_id = runtime.create_conversation(conversation_title[:200], kind="subagent", category=category)

        if is_scheduled:
            timekeeper = context.services.get("timekeeper")
            if timekeeper is not None and getattr(timekeeper, "loaded", False):
                try:
                    job = timekeeper.get_job(job_name)
                    if job is not None:
                        new_payload = dict(job.get("payload") or {})
                        new_payload["conversation_id"] = conv_id
                        timekeeper.update_job(job_name, {"payload": new_payload})
                except Exception as e:
                    logger.warning(f"Failed to persist conversation_id for job '{job_name}': {e}")

        return conv_id, conv_id == active_id

    @staticmethod
    def _resolve_agent_from_conversation(context, conversation_id: int) -> str:
        """Read the agent profile off the conversation's latest state marker.

        Falls back to the global active agent profile, then "default", so a
        brand-new conversation that has never been briefed still has a sane
        default to run under.
        """
        try:
            rows = context.db.get_conversation_messages(conversation_id) or []
        except Exception:
            rows = []
        marker = latest_state(rows) or {}
        candidate = (marker.get("profile_override") or marker.get("active_agent_profile") or "").strip()
        if candidate:
            return candidate
        return (context.config.get("active_agent_profile") or "").strip() or "default"

    def _emit_fallback_push(self, run_id: str, job_name: str, title: str,
                            final_answer: str, push_records: list[NotificationRecord],
                            context, conversation_id: int | None) -> None:
        from plugins.tasks.helpers.notifications import load_conversation_suffix
        message = final_answer.strip() + load_conversation_suffix(getattr(context, "db", None), conversation_id)
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
        from attachments import parse_attachment as build_attachment

        prompt_parts = [prompt]
        attachments = []

        for raw_path in input_paths:
            path = str(raw_path).strip()
            if not path:
                continue
            attachments.append(
                build_attachment(
                    path,
                    services=getattr(context, "services", None),
                    config={"max_chars": _MAX_PARSED_CHARS},
                )
            )
            prompt_parts.append(f"\n[Input file: {path}]")

        return "\n".join(prompt_parts).strip(), attachments

    @staticmethod
    def _normalize_input_paths(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]
