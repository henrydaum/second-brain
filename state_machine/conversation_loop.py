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

from state_machine.persistence import save_history_message

logger = logging.getLogger("ConversationLoop")


def _clean(text: str | None) -> str:
    text = text or ""
    while "<think>" in text or "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>")
        if start >= 0 and end > start:
            text = text[:start] + text[end + len("</think>"):]
        else:
            text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


class ConversationLoop:
    """Drive a participant's turn until they end it.

    For an agent: ask the LLM, translate the response into typed actions
    (`send_text`, `call_tool`, `end_turn`), dispatch each through
    `cs.enact()`. Tool execution lives inside `CallTool` (via the shared
    `_CallableAction._run` path), so this loop never touches the registry
    directly — it only orchestrates.
    """

    OVER_BUDGET_MESSAGE = "I've made too many tool calls. Could you try a more specific question?"

    def __init__(
        self,
        llm,
        tool_registry,
        config: dict,
        system_prompt: str | Callable[[], str],
        on_tool_start=None,
        on_tool_result=None,
        on_notice=None,
    ):
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config
        self.system_prompt = system_prompt
        self.on_tool_start = on_tool_start
        self.on_tool_result = on_tool_result
        self.on_notice = on_notice
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
        image_paths: list[str] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]], list[str]]:
        """Run iterations of choose-action / enact / record until turn ends.

        `history` is the provider-shaped transcript and is mutated in place;
        `new_messages` is what was appended this turn (returned for adapters).
        """
        self.running = True
        self.cancelled = False
        self._tool_call_counts.clear()
        self._pending_tool_calls.clear()
        self._assistant_text_for_pending = None
        self._final_text = None

        new_messages: list[dict[str, Any]] = []
        attachments: list[str] = []
        images = list(image_paths or [])

        # Generous upper bound so multi-call rounds (k tool calls per LLM turn,
        # potentially several rounds) cannot infinite-loop.
        max_iterations = (self.max_tool_calls + 1) * 4

        try:
            for _ in range(max_iterations):
                if self.cancelled or cs.turn_priority != actor_id:
                    break

                action_type, content = self._next_action(cs, history, images)
                if not action_type:
                    break
                if images:
                    images = []  # Only the first LLM call sees attached images.

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
                # Used up the iteration budget; close the turn cleanly through
                # the same `enact()` site so events stay consistent.
                self._absorb(
                    cs.enact("send_text", self.OVER_BUDGET_MESSAGE, actor_id),
                    "send_text", self.OVER_BUDGET_MESSAGE,
                    history, new_messages, attachments, db, conversation_id,
                )
                cs.enact("end_turn", None, actor_id)

            return self._final_text, new_messages, attachments
        finally:
            self.running = False

    # ──────────────────────────────────────────────────────────────────────
    # Picking the next action (the LLM half of the loop)
    # ──────────────────────────────────────────────────────────────────────

    def _next_action(
        self,
        cs,
        history: list[dict[str, Any]],
        image_paths: list[str],
    ) -> tuple[str | None, Any]:
        """Return `(action_type, content)` for the agent's next move.

        Drains pending tool calls from the previous LLM response one at a time
        before issuing the next LLM request. When the LLM returns text-only,
        emits `send_text` first and then `end_turn` on the following iteration.
        """
        # 1) Still have pending tool calls? Issue one. The first call of a
        #    batch carries the assistant's accompanying text (if any).
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
        response = self._invoke(self._messages(history), self.tool_registry.get_all_schemas() or None, image_paths or None, history)

        if getattr(response, "has_tool_calls", False):
            self._pending_tool_calls = list(response.tool_calls)
            self._assistant_text_for_pending = getattr(response, "content", None)
            self._compact_if_needed(response, history)
            # Recurse to immediately return the first call as an action.
            return self._next_action(cs, history, [])

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

        # Apply call-budget bookkeeping (mirror previous behavior).
        tool = getattr(self.tool_registry, "tools", {}).get(name)
        if tool and self._tool_call_counts.get(name, 0) >= getattr(tool, "max_calls", 1):
            return json.dumps({"error": f"Tool '{name}' has reached its call limit ({tool.max_calls}). Try a different approach."}), []
        self._tool_call_counts[name] = self._tool_call_counts.get(name, 0) + 1

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
            return text, paths
        except (TypeError, ValueError) as e:
            return json.dumps({"error": f"Result serialization failed: {e}"}), []

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _messages(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = self.system_prompt() if callable(self.system_prompt) else self.system_prompt
        return [{"role": "system", "content": prompt}, *[m for m in history if m.get("role") != "system"]]

    def _invoke(self, messages, tools, image_paths=None, history=None):
        from plugins.services.llmService import is_context_limit_error

        try:
            response = self.llm.chat_with_tools(messages, tools, image_paths=image_paths)
        except Exception as e:
            if history is None or not is_context_limit_error(e):
                raise
            logger.warning("Context limit hit, compacting and retrying: %s", e)
            self._compact(history)
            try:
                response = self.llm.chat_with_tools(self._messages(history), tools, image_paths=None)
            except Exception as retry_error:
                if is_context_limit_error(retry_error):
                    raise RuntimeError("Context limit reached even after compacting. Use /new to start fresh.") from retry_error
                raise
        if getattr(response, "is_error", False):
            err = getattr(response, "error", None) or getattr(response, "content", None) or "LLM provider error."
            if history is not None and is_context_limit_error(err):
                logger.warning("Context limit hit (response error), compacting and retrying: %s", err)
                self._compact(history)
                response = self.llm.chat_with_tools(self._messages(history), tools, image_paths=None)
                if getattr(response, "is_error", False):
                    raise RuntimeError("Context limit reached even after compacting. Use /new to start fresh.")
            else:
                raise RuntimeError(err)
        return response

    def _compact_if_needed(self, response, history) -> None:
        # Proactive compaction: trigger before hitting the context limit when
        # the model's context_size is set. context_size == 0 disables proactive
        # compaction; reactive compaction in `_invoke` is the safety net.
        ctx, tok = getattr(self.llm, "context_size", 0), getattr(response, "prompt_tokens", 0)
        if not ctx or not tok or tok / ctx < 0.80 or len(history) <= 2:
            return
        self._compact(history)

    def _compact(self, history) -> None:
        """Summarize the head of `history` in place. Used by both proactive
        and reactive compaction."""
        if len(history) <= 2:
            return
        try:
            transcript = "\n".join(f"{m.get('role', '').upper()}: {(m.get('content') or '')[:1000]}" for m in history[:-2])
            summary_response = self.llm.chat_with_tools([
                {"role": "system", "content": "Summarize this Second Brain conversation so the assistant can continue with minimal loss."},
                {"role": "user", "content": transcript[:20000]},
            ], None)
            if getattr(summary_response, "is_error", False):
                logger.debug("Compaction summarization returned error: %s", getattr(summary_response, "error", None))
                return
            summary = _clean(getattr(summary_response, "content", ""))
            if not summary:
                return
            history[:] = [
                {"role": "user", "content": f"[Conversation summary from earlier]\n{summary}"},
                {"role": "assistant", "content": "Understood - I have the earlier context."},
                *history[-2:],
            ]
            if self.on_notice:
                self.on_notice(f"Compacted conversation into {len(summary)} chars.")
        except Exception as e:
            logger.debug("Compaction failed: %s", e, exc_info=True)

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
