from __future__ import annotations

"""Drive one participant's turn through cs.enact() until end_turn.

This file is the agent-side equivalent of PokerMonster's `run_game` inner
loop. While turn priority belongs to a participant, repeatedly:
    1. ask the participant for the next action  (`_next_action`)
    2. enact it through the state machine       (`cs.enact(...)`)
    3. translate the action's events back into provider-shaped history rows

There is exactly ONE `cs.enact(...)` call site in this file (in `drive`),
labeled with a comment block so it is easy to find.

The class is named `ConversationLoop` (not `AgentMachine`) because the same
shape supports user-user or agent-agent conversations in the future. Today
only the agent path is wired; a user-side `_next_action` would just block
until input arrives.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

from state_machine.serialization import save_compaction_marker, save_history_message
from runtime.token_stripper import strip_model_tokens

logger = logging.getLogger("ConversationLoop")


def _clean(text: str | None) -> str:
    return strip_model_tokens(text or "")[0]


def _truncate_middle(text: str, max_chars: int) -> str:
    """Cap a string by keeping the head and tail and inserting a marker.

    Used to keep oversized tool results from blowing the context window
    while preserving enough signal that the LLM can tell what kind of
    payload was elided.
    """
    if not text or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n…[truncated {len(text) - max_chars} chars]…\n{text[-tail:]}"


class ConversationLoop:
    """Drive a participant's turn until they end it.

    For an agent: ask the LLM, translate the response into typed actions
    (`send_text`, `call_tool`, `end_turn`), dispatch each through
    `cs.enact()`. Tool execution lives inside `CallTool` (via the shared
    `_CallableAction._run` path), so this loop never touches the registry
    directly — it only orchestrates.
    """

    OVER_BUDGET_MESSAGE = "I've made too many tool calls. Could you try a more specific question?"
    MAX_TOOL_RESULT_CHARS = 12000

    def __init__(
        self,
        llm,
        tool_registry,
        config: dict,
        system_prompt: str | Callable[[], str],
        on_tool_start=None,
        on_tool_result=None,
        on_notice=None,
        cancel_event=None,
        runtime=None,
        session_key: str | None = None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config
        self.system_prompt = system_prompt
        self.on_tool_start = on_tool_start
        self.on_tool_result = on_tool_result
        self.on_notice = on_notice
        self.cancel_event = cancel_event
        self.runtime = runtime
        self.session_key = session_key
        self.cancelled = False
        self.running = False
        self._tool_call_counts: dict[str, int] = {}
        # Pending tool calls from the latest LLM response. The loop drains this
        # one-per-iteration so each tool call goes through its own `enact()`.
        self._pending_tool_calls: list[dict[str, Any]] = []
        # The LLM's accompanying text for the current tool-call batch. It
        # rides along on the FIRST CallTool action of the batch so the
        # provider transcript keeps its assistant-text-with-tool-calls shape.
        self._assistant_text_for_pending: str | None = None
        self._final_text: str | None = None
        self._active_db = None
        self._active_conversation_id = None

    @property
    def max_tool_calls(self) -> int:
        return (
            getattr(self.tool_registry, "max_tool_calls", 0)
            or sum(getattr(t, "max_calls", 1) for t in getattr(self.tool_registry, "tools", {}).values())
            or 1
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public entrypoint
    # ──────────────────────────────────────────────────────────────────────

    def drive(
        self,
        cs,
        actor_id: str,
        history: list[dict[str, Any]],
        db=None,
        conversation_id: int | None = None,
    ) -> tuple[str | None, list[dict[str, Any]], list[str]]:
        """Run iterations of choose-action / enact / record until turn ends.

        `history` is the provider-shaped transcript and is mutated in place;
        `new_messages` is what was appended this turn (returned for adapters).

        Attachments queued on ``cs.pending_attachments`` are bundled and
        passed to the LLM on the first call of the turn; the bundle is
        then cleared (``per_turn`` lifecycle) or kept for the next turn
        (``persistent`` lifecycle).
        """
        self.running = True
        self.cancelled = False
        self._tool_call_counts.clear()
        self._pending_tool_calls.clear()
        self._assistant_text_for_pending = None
        self._final_text = None
        self._active_db = db
        self._active_conversation_id = conversation_id

        new_messages: list[dict[str, Any]] = []
        attachments: list[str] = []

        from attachments.attachment import AttachmentBundle
        bundle = AttachmentBundle.from_iterable(cs.pending_attachments)
        if getattr(cs, "attachment_lifecycle", "per_turn") == "per_turn":
            cs.pending_attachments = []

        # Generous upper bound so multi-call rounds (k tool calls per LLM turn,
        # potentially several rounds) cannot infinite-loop.
        max_iterations = (self.max_tool_calls + 1) * 4

        try:
            for _ in range(max_iterations):
                if self._cancelled() or cs.turn_priority != actor_id:
                    break

                action_type, content = self._next_action(cs, history, bundle)
                if not action_type:
                    break
                if bundle:
                    # Only the first LLM call of the turn sees the bundle.
                    bundle = AttachmentBundle()

                if self._cancelled():
                    break
                if action_type == "call_tool":
                    budget_error = self._tool_budget_error(content)
                    if budget_error:
                        from state_machine.errors import ActionResult
                        result = ActionResult.fail("call_tool", budget_error, code="tool_budget_exceeded")
                        self._absorb(result, action_type, content, history, new_messages, attachments, db, conversation_id)
                        continue
                started = self._tool_started(action_type, content)
                try:
                    # ──────────────────── THE enact() SITE ────────────────────
                    result = cs.enact(action_type, content, actor_id)
                    # ──────────────────────────────────────────────────────────
                except Exception as e:
                    self._tool_finished(started, error=str(e))
                    raise
                self._tool_finished(started, result=result)

                self._absorb(result, action_type, content, history, new_messages, attachments, db, conversation_id)

                if action_type == "end_turn" or not result.ok:
                    break

            if cs.turn_priority == actor_id:
                if not self._cancelled():
                    self._over_budget_summary(cs, actor_id, history, new_messages, attachments, db, conversation_id)
                cs.enact("end_turn", None, actor_id)

            return self._final_text, new_messages, attachments
        finally:
            self._active_db = None
            self._active_conversation_id = None
            self.running = False

    # ──────────────────────────────────────────────────────────────────────
    # Picking the next action (the LLM half of the loop)
    # ──────────────────────────────────────────────────────────────────────

    def _next_action(
        self,
        cs,
        history: list[dict[str, Any]],
        bundle,
    ) -> tuple[str | None, Any]:
        """Return `(action_type, content)` for the agent's next move.

        Drains pending tool calls from the previous LLM response one at a time
        before issuing the next LLM request. When the LLM returns text-only,
        emits `send_text` first and then `end_turn` on the following iteration.
        """
        # 1) Still have pending tool calls? Issue one. The first call of a
        #    batch carries the assistant's accompanying text (if any).
        if self._cancelled():
            return None, None
        if self._pending_tool_calls:
            tc = self._pending_tool_calls.pop(0)
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except json.JSONDecodeError as e:
                args = {"__invalid_arguments__": str(e)}
            content = {
                "name": tc.get("name"),
                "args": args,
                "_tool_call_id": tc.get("id"),
                "_assistant_text": self._assistant_text_for_pending,
            }
            self._assistant_text_for_pending = None  # only first call carries it
            return "call_tool", content

        # 2) Final text was already emitted but turn isn't ended → end it.
        if self._final_text is not None and cs.turn_priority == "agent":
            text = self._final_text
            return "end_turn", {"final_text": text}

        # 3) Otherwise call the LLM for the next response.
        from attachments.attachment import AttachmentBundle
        schemas = self.tool_registry.get_all_schemas() if self.tool_registry else None
        response = self._invoke(self._messages(history), schemas or None, bundle, history)

        if getattr(response, "has_tool_calls", False):
            self._pending_tool_calls = list(response.tool_calls)
            self._assistant_text_for_pending = getattr(response, "content", None)
            self._compact_if_needed(response, history)
            # Recurse to immediately return the first call as an action.
            return self._next_action(cs, history, AttachmentBundle())

        # Text-only response: emit `send_text` now; next iteration will end_turn.
        text = _clean(getattr(response, "content", ""))
        self._final_text = text
        self._compact_if_needed(response, history)
        return "send_text", text

    # ──────────────────────────────────────────────────────────────────────
    # Translating action results back into provider-shaped history rows
    # ──────────────────────────────────────────────────────────────────────

    def _absorb(
        self,
        result,
        action_type: str,
        content: Any,
        history: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        attachments: list[str],
        db,
        conversation_id,
    ) -> None:
        """Read the action's outcome and append matching history rows."""
        if action_type == "send_text":
            text = content if isinstance(content, str) else ""
            self._record({"role": "assistant", "content": text}, history, new_messages, db, conversation_id)
            return

        if action_type == "call_tool":
            tc_id = (content or {}).get("_tool_call_id") or "tc_unknown"
            name = (content or {}).get("name") or "unknown"
            args = (content or {}).get("args") or {}
            assistant_text = (content or {}).get("_assistant_text")
            assistant_msg = {
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, default=str)},
                }],
            }
            self._record(assistant_msg, history, new_messages, db, conversation_id)

            tool_text, tool_paths = self._format_tool_result(name, result, args)
            attachments.extend(tool_paths)
            self._record(
                {"role": "tool", "tool_call_id": tc_id, "name": name, "content": tool_text},
                history, new_messages, db, conversation_id,
            )
            return

        if action_type == "end_turn":
            # Final text, if any, was already recorded as a SendText. EndTurn
            # itself does not emit a history row.
            return

    def _format_tool_result(self, name: str, result, args: dict[str, Any]) -> tuple[str, list[str]]:
        """Serialize the action's outcome into `(text, attachment_paths)`."""
        if "__invalid_arguments__" in (args or {}):
            return json.dumps({"error": f"Invalid arguments: {args['__invalid_arguments__']}"}), []

        # The `call_tool` action's data carries the underlying ToolResult.
        payload = (getattr(result, "data", None) or {})
        tool_result = payload.get("result")

        # Action-level failure (legality, exec error) → tool error message.
        if not getattr(result, "ok", True):
            err = getattr(result, "error", None)
            return json.dumps({"error": err.message if err else "Tool failed."}), []

        # ToolResult-level failure.
        if tool_result is not None and not getattr(tool_result, "success", True):
            return json.dumps({"error": getattr(tool_result, "error", "Tool failed.")}), []

        paths = list(getattr(tool_result, "attachment_paths", []) or [])
        try:
            text = (
                getattr(tool_result, "llm_summary", None)
                or json.dumps(getattr(tool_result, "data", None), default=str)
            )
            return _truncate_middle(text, self.MAX_TOOL_RESULT_CHARS), paths
        except (TypeError, ValueError) as e:
            return json.dumps({"error": f"Result serialization failed: {e}"}), []

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = self.system_prompt() if callable(self.system_prompt) else self.system_prompt
        return [{"role": "system", "content": prompt}, *[m for m in history if m.get("role") != "system"]]

    def _tool_budget_error(self, content: Any) -> str | None:
        name = (content or {}).get("name") or "unknown"
        tool = (getattr(self.tool_registry, "tools", {}) or {}).get(name) if self.tool_registry else None
        if not tool:
            return None
        used, limit = self._tool_call_counts.get(name, 0), getattr(tool, "max_calls", 1)
        if used >= limit:
            return f"Tool '{name}' has reached its call limit ({limit}). Try a different approach."
        self._tool_call_counts[name] = used + 1
        return None

    def _invoke(self, messages, tools, attachments=None, history=None):
        from plugins.services.llmService import is_context_limit_error

        bundle = attachments or None
        try:
            response = self.llm.chat_with_tools(messages, tools, attachments=bundle)
        except Exception as e:
            if history is None or not is_context_limit_error(e):
                raise
            logger.warning("Context limit hit, compacting and retrying: %s", e)
            response = self._retry_after_overflow(tools, history)
        if getattr(response, "is_error", False):
            err = getattr(response, "error", None) or getattr(response, "content", None) or "LLM provider error."
            if history is not None and is_context_limit_error(err):
                logger.warning("Context limit hit (response error), compacting and retrying: %s", err)
                response = self._retry_after_overflow(tools, history)
            else:
                raise RuntimeError(err)
        return response

    def _retry_after_overflow(self, tools, history):
        """Compact + retry. If retry still overflows, drop history down to
        a single emergency stub and retry once more. Only after THAT
        fails do we surface the unrecoverable error."""
        from plugins.services.llmService import is_context_limit_error

        self._compact(history)
        try:
            return self.llm.chat_with_tools(self._messages(history), tools, attachments=None)
        except Exception as retry_error:
            if not is_context_limit_error(retry_error):
                raise
            logger.warning("Post-compact retry still over context, doing emergency truncation: %s", retry_error)

        self._emergency_truncate(history)
        try:
            response = self.llm.chat_with_tools(self._messages(history), tools, attachments=None)
        except Exception as final_error:
            if is_context_limit_error(final_error):
                raise RuntimeError("Context limit reached even after compacting. Use /new to start fresh.") from final_error
            raise
        if getattr(response, "is_error", False):
            err = getattr(response, "error", None) or "LLM provider error."
            if is_context_limit_error(err):
                raise RuntimeError("Context limit reached even after compacting. Use /new to start fresh.")
            raise RuntimeError(err)
        return response

    def _emergency_truncate(self, history) -> None:
        """Last-resort shrink that does NOT call the LLM. Keeps only the
        most recent user message (and any in-flight tool_call/result pair
        that immediately follows it), aggressively truncating any string
        content. Used when compaction itself can't help — either because
        the compact_chat task didn't run, the summary came back empty, or
        the post-compact retry still overflowed."""
        if not history:
            return
        last_user_idx = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
        if last_user_idx is None:
            keep = history[-1:]
        else:
            keep = history[last_user_idx:]
        cap = 2000
        shrunk = []
        for msg in keep:
            content = msg.get("content")
            if isinstance(content, str) and len(content) > cap:
                msg = {**msg, "content": _truncate_middle(content, cap)}
            shrunk.append(msg)
        original_count = len(history)
        history[:] = [
            {"role": "user", "content": "[Earlier conversation dropped to fit context. Continue from the message below.]"},
            {"role": "assistant", "content": "Understood."},
            *shrunk,
        ]
        logger.warning(f"Emergency-truncated history from {original_count} -> {len(history)} messages.")
        if self.on_notice:
            self.on_notice(f"Context overflow: dropped earlier messages to keep going (was {original_count}).")

    def _compact_if_needed(self, response, history) -> None:
        # Proactive compaction: trigger before hitting the context limit when
        # the model's context_size is set. context_size == 0 disables proactive
        # compaction; reactive compaction in `_invoke` is the safety net.
        ctx, tok = getattr(self.llm, "context_size", 0), getattr(response, "prompt_tokens", 0)
        if not ctx or not tok or tok / ctx < 0.80 or len(history) <= 2:
            return
        self._compact(history)

    def _compact(self, history) -> None:
        """Summarize the head of `history` in place by delegating to the
        ``compact_chat`` task. The runtime call blocks until the task
        finishes (or times out)."""
        if len(history) <= 2 or self.runtime is None:
            return
        try:
            transcript = "\n".join(f"{m.get('role', '').upper()}: {(m.get('content') or '')[:1000]}" for m in history)
            transcript = transcript[:20000]
            summary = self.runtime.request_compaction(self.session_key, transcript)
            if not summary:
                logger.warning("Compaction returned no summary (timeout or empty). History will not shrink via summary.")
                return
            old_count = len(history)
            if self._active_db is not None and self._active_conversation_id is not None:
                save_compaction_marker(self._active_db, self._active_conversation_id, summary)
                session = getattr(self.runtime, "sessions", {}).get(self.session_key)
                if session is not None:
                    session.has_compaction_checkpoint = True
            tail = [self._shrink_for_tail(m) for m in history[-2:]]
            history[:] = [
                {"role": "user", "content": f"[Conversation summary from earlier]\n{summary}"},
                {"role": "assistant", "content": "Understood - I have the earlier context."},
                *tail,
            ]
            if self.on_notice:
                self.on_notice(f"Compacted {old_count} messages.")
        except Exception as e:
            logger.debug("Compaction failed: %s", e, exc_info=True)

    def _shrink_for_tail(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Aggressively truncate any oversized message preserved through
        compaction. Without this, a huge ``role: tool`` result in the last
        two messages would survive compaction intact and the post-compact
        retry would overflow again."""
        content = msg.get("content")
        if not isinstance(content, str) or len(content) <= self.MAX_TOOL_RESULT_CHARS:
            return msg
        return {**msg, "content": _truncate_middle(content, self.MAX_TOOL_RESULT_CHARS)}

    def _over_budget_summary(self, cs, actor_id, history, new_messages, attachments, db, conversation_id) -> None:
        try:
            nudge = {"role": "user", "content": "You've hit the tool-call limit. Summarize what you have and stop calling tools."}
            response = self._invoke(self._messages([*history, nudge]), None, None, history)
            text = _clean(getattr(response, "content", "")) or self.OVER_BUDGET_MESSAGE
        except Exception:
            text = self.OVER_BUDGET_MESSAGE
        self._final_text = text
        self._absorb(cs.enact("send_text", text, actor_id), "send_text", text, history, new_messages, attachments, db, conversation_id)

    def _cancelled(self) -> bool:
        return self.cancelled or bool(self.cancel_event and self.cancel_event.is_set())

    def _record(self, msg, history, new_messages, db, conversation_id):
        history.append(msg)
        new_messages.append(msg)
        if db is not None and conversation_id is not None:
            save_history_message(db, conversation_id, msg)

    def _tool_started(self, action_type: str, content: Any):
        if action_type != "call_tool":
            return None
        name = (content or {}).get("name") or "unknown"
        call_id = (content or {}).get("_tool_call_id") or "tc_unknown"
        args = (content or {}).get("args") or {}
        if self.on_tool_start:
            try:
                self.on_tool_start(name, call_id, args)
            except TypeError:
                self.on_tool_start(name)
        return name, call_id

    def _tool_finished(self, started, result=None, error: str | None = None):
        if not started or not self.on_tool_result:
            return
        name, call_id = started
        try:
            self.on_tool_result(name, call_id, result, error)
        except TypeError:
            self.on_tool_result(name, (getattr(result, "data", None) or {}).get("result") if result else None)

    @staticmethod
    def _is_image(path: str) -> bool:
        return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
