"""
Agent.

The thin loop that connects a caller to the tool layer via LLM
function calling. Receives an LLM instance from the ServiceManager
and a ToolRegistry — owns no clients, no config, no state beyond
conversation history.

Usage:
    llm = service_manager.get("llm")        # OpenAILLM with chat_with_tools()
    tools = tool_registry                   # ToolRegistry with registered tools

    agent = Agent(llm, tools)
    answer = agent.chat("What files mention revenue?")

    # Multi-turn
    answer = agent.chat("Summarize the top result")
"""

import json
import logging
import time
from pathlib import Path

from Stage_1.registry import get_modality
from gui.token_stripper import strip_model_tokens

logger = logging.getLogger("Agent")


class Agent:
    def __init__(self, llm, tool_registry, config, system_prompt=None,
                 on_tool_result=None, on_message=None):
        """
        Args:
            llm:            A BaseLLM instance that implements chat_with_tools().
            tool_registry:  A ToolRegistry instance with registered tools.
            system_prompt:  A string, or a callable that returns a string.
                            If callable, it is re-evaluated on every chat() call
                            so the LLM always sees current state (e.g. after
                            plugins are added/removed via hot-reload).
            on_tool_result: Optional callback(tool_name: str, tool_result: ToolResult)
                            fired after each tool execution, for GUI rendering.
            on_message:     Optional callback(msg: dict) fired after each message
                            is added to history, for conversation persistence.
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.on_tool_result = on_tool_result
        self.on_message = on_message
        self._default_prompt = (
            "You are a helpful assistant with access to a local file database. "
            "Use the available tools to search and retrieve information from the user's files. "
            "Be concise and cite which files your answers come from."
        )
        self.system_prompt = system_prompt or self._default_prompt
        self.max_tool_calls = tool_registry.max_tool_calls
        self.history: list[dict] = []
        self._tool_call_counts: dict[str, int] = {}
        self.cancelled = False

    def chat(self, message: str) -> str:
        """
        Send a message and get a response. Handles tool calls automatically.
        Maintains conversation history across calls.

        Returns the assistant's final text response.
        """
        user_msg = {"role": "user", "content": message}
        self.history.append(user_msg)
        self._fire_on_message(user_msg)

        # Build full message list with system prompt (re-evaluated if callable)
        prompt = self.system_prompt() if callable(self.system_prompt) else self.system_prompt
        messages = [{"role": "system", "content": prompt}]
        messages.extend(self.history)

        tools = self.tool_registry.get_all_schemas() or None
        self._tool_call_counts.clear()

        compiled_image_paths = []
        _prev_tool_count = len(tools) if tools else 0

        for round_num in range(self.max_tool_calls):
            if self.cancelled:
                return None

            logger.debug(
                f"LLM call (round {round_num + 1}), history size: {len(self.history)} messages"
            )
            t0 = time.time()
            response = self.llm.chat_with_tools(messages, tools, image_paths=compiled_image_paths or None)
            logger.debug(f"LLM responded in {time.time() - t0:.2f}s")

            if not response.has_tool_calls:
                clean, _ = strip_thinking(response.content)
                assistant_msg = {"role": "assistant", "content": clean}
                self.history.append(assistant_msg)
                self._fire_on_message(assistant_msg)
                return response.content  # raw — GUI renders thinking dropdown

            # Build the assistant message with tool calls for the conversation
            tool_names = [tc["name"] for tc in response.tool_calls]
            logger.info(f"Agent requesting tool calls: {tool_names}")
            assistant_msg = {"role": "assistant", "content": response.content or None, "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in response.tool_calls
            ]}
            messages.append(assistant_msg)
            self.history.append(assistant_msg)
            self._fire_on_message(assistant_msg)

            # Execute each tool call and append results
            for tc in response.tool_calls:
                if self.cancelled:
                    break
                t_tool = time.time()
                result_str, tc_images = self._execute_tool_call(tc)
                if tc_images:
                    compiled_image_paths.extend(tc_images)
                logger.debug(f"Tool '{tc['name']}' completed in {time.time() - t_tool:.2f}s")
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": result_str}
                messages.append(tool_msg)
                self.history.append(tool_msg)
                self._fire_on_message(tool_msg)

            # Refresh schemas if tools were added/removed (e.g. by build_plugin)
            refreshed = self.tool_registry.get_all_schemas() or []
            new_count = len(refreshed)
            if new_count != _prev_tool_count:
                tools = refreshed or None
                _prev_tool_count = new_count
                logger.info(f"Tool schemas refreshed — now {new_count} tool(s)")

        # Exceeded max rounds
        logger.warning(f"Agent hit max tool rounds ({self.max_tool_calls})")
        fallback = "I've made too many tool calls. Could you try a more specific question?"
        fallback_msg = {"role": "assistant", "content": fallback}
        self.history.append(fallback_msg)
        self._fire_on_message(fallback_msg)
        return fallback

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def _fire_on_message(self, msg: dict):
        """Notify the on_message callback, if set."""
        if self.on_message:
            try:
                self.on_message(msg)
            except Exception as e:
                logger.debug(f"on_message callback error: {e}")

    def _execute_tool_call(self, tool_call: dict) -> tuple[str, list[str]]:
        """Execute a single tool call via the registry, return (result_string, image_paths)."""
        name = tool_call["name"]
        try:
            args = json.loads(tool_call["arguments"])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse arguments for tool '{name}': {e}")
            return json.dumps({"error": f"Invalid arguments: {e}"}), []

        # Enforce per-tool call limit
        tool = self.tool_registry.tools.get(name)
        if tool:
            count = self._tool_call_counts.get(name, 0)
            if count >= tool.max_calls:
                logger.warning(f"Tool '{name}' hit max calls ({tool.max_calls})")
                return json.dumps({"error": f"Tool '{name}' has reached its call limit ({tool.max_calls}). Try a different approach."}), []

        logger.info(f"Tool call: {name}({args})")

        result = self.tool_registry.call(name, **args)
        self._tool_call_counts[name] = self._tool_call_counts.get(name, 0) + 1

        if self.on_tool_result:
            try:
                self.on_tool_result(name, result)
            except Exception as e:
                logger.debug(f"on_tool_result callback error: {e}")

        if result.success:
            image_paths = []
            if result.gui_display_paths:
                image_paths = [p for p in result.gui_display_paths if get_modality(Path(p).suffix) == "image"]

            try:
                result_str = result.llm_summary or json.dumps(result.data, default=str)
                return result_str, image_paths
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize result from '{name}': {e}")
                return json.dumps({"error": f"Result serialization failed: {e}"}), []
        else:
            logger.warning(f"Tool '{name}' failed: {result.error}")
            return json.dumps({"error": result.error}), []