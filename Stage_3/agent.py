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

logger = logging.getLogger("Agent")


class Agent:
    def __init__(self, llm, tool_registry, config, system_prompt: str = None):
        """
        Args:
            llm:            A BaseLLM instance that implements chat_with_tools().
            tool_registry:  A ToolRegistry instance with registered tools.
            system_prompt:  Optional system message. Uses a sensible default if None.
        """
        self.llm = llm
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt or (
            "You are a helpful assistant with access to a local file database. "
            "Use the available tools to search and retrieve information from the user's files. "
            "Be concise and cite which files your answers come from."
        )
        self.MAX_TOOL_CALLS = config['max_tool_calls']
        self.history: list[dict] = []

    def chat(self, message: str) -> str:
        """
        Send a message and get a response. Handles tool calls automatically.
        Maintains conversation history across calls.

        Returns the assistant's final text response.
        """
        self.history.append({"role": "user", "content": message})

        # Build full message list with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)

        tools = self.tool_registry.get_all_schemas() or None

        for _ in range(self.MAX_TOOL_CALLS):
            response = self.llm.chat_with_tools(messages, tools)

            if not response.has_tool_calls:
                self.history.append({"role": "assistant", "content": response.content})
                return response.content

            # Build the assistant message with tool calls for the conversation
            assistant_msg = {"role": "assistant", "content": response.content or None, "tool_calls": [
                {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in response.tool_calls
            ]}
            messages.append(assistant_msg)
            self.history.append(assistant_msg)

            # Execute each tool call and append results
            for tc in response.tool_calls:
                result_str = self._execute_tool_call(tc)
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result_str}
                messages.append(tool_msg)
                self.history.append(tool_msg)

        # Exceeded max rounds
        logger.warning(f"Agent hit max tool rounds ({self.MAX_TOOL_CALLS})")
        fallback = "I've made too many tool calls. Could you try a more specific question?"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def _execute_tool_call(self, tool_call: dict) -> str:
        """Execute a single tool call via the registry, return result as string."""
        name = tool_call["name"]
        try:
            args = json.loads(tool_call["arguments"])
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse arguments for tool '{name}': {e}")
            return json.dumps({"error": f"Invalid arguments: {e}"})

        logger.info(f"Tool call: {name}({args})")

        result = self.tool_registry.call(name, **args)

        if result.success:
            try:
                return json.dumps(result.data, default=str)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize result from '{name}': {e}")
                return json.dumps({"error": f"Result serialization failed: {e}"})
        else:
            logger.warning(f"Tool '{name}' failed: {result.error}")
            return json.dumps({"error": result.error})