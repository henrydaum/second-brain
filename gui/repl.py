"""
REPL.

Simple command loop that maps user input to the shared CommandRegistry.
Runs on its own daemon thread so it never blocks the dispatch loop.
"""

import json
import logging
import threading
from pathlib import Path

from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from gui.commands import CommandEntry, CommandRegistry, register_core_commands
from gui.formatters import format_tool_result

logger = logging.getLogger("REPL")


# =================================================================
# REPL LOOP
# =================================================================

def run_repl(ctrl, shutdown_fn, shutdown_event: threading.Event,
             tool_registry, services, config, root_dir: Path):
    agent = None

    # --- Console-based approval for run_command (fallback; GUI overrides) ---
    def _repl_approve_command(command: str, justification: str) -> bool:
        print(f"\n--- Agent wants to run a command ---")
        print(f"  Command:  {command}")
        print(f"  Reason:   {justification}")
        try:
            response = input("  Allow? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        return response in ("y", "yes")

    if tool_registry.on_approve_command is None:
        tool_registry.on_approve_command = _repl_approve_command

    # --- Build command registry (shared + REPL-specific) ---
    registry = CommandRegistry()
    register_core_commands(registry, ctrl, services, tool_registry, root_dir)

    # --- REPL-specific commands ---

    def _call_handler(arg):
        if not arg:
            return ("Usage: /call <tool_name> {\"arg\": \"value\"}\n"
                    "Example: /call sql_query {\"sql\": \"SELECT * FROM files LIMIT 5\"}")

        parts = arg.split(maxsplit=1)
        tool_name = parts[0]
        raw_args = parts[1] if len(parts) > 1 else "{}"

        try:
            kwargs = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return f"Invalid JSON arguments: {e}\nExpected format: /call <tool_name> {{\"key\": \"value\"}}"

        return format_tool_result(ctrl.call_tool(tool_name, kwargs))

    def _chat_handler(_arg):
        nonlocal agent
        llm = services.get("llm")
        if llm is None or not llm.loaded:
            return "LLM service not loaded. Run '/load llm' first."

        prompt = build_system_prompt(ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services)
        agent = Agent(llm, tool_registry, config, system_prompt=prompt)
        logger.info("Agent initialized.")

        print("Entering chat mode. Type 'exit' to return to REPL.")
        print("---")

        while not shutdown_event.is_set():
            try:
                user_input = input("you> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "back"):
                break
            if user_input.lower() == "reset":
                agent.reset()
                print("(conversation history cleared)")
                continue

            try:
                response = agent.chat(user_input)
                print(f"\nassistant> {response}\n")
            except Exception as e:
                logger.error(f"Agent error: {e}")
                print(f"Error: {e}")

        print("---")
        print("Exited chat mode.")
        return None

    def _quit_handler(_arg):
        shutdown_fn()
        return None

    for entry in [
        CommandEntry("call", "Call a tool directly", "<tool> {json}",
                     handler=_call_handler,
                     arg_completions=lambda: list(tool_registry.tools.keys())),
        CommandEntry("chat",  "Enter interactive chat mode", handler=_chat_handler),
        CommandEntry("clear", "Clear chat conversation history",
                     handler=lambda _: (agent.reset() if agent else None) or "(conversation history cleared)"),
        CommandEntry("quit",  "Shutdown", handler=_quit_handler),
        CommandEntry("exit",  "Shutdown", handler=_quit_handler),
    ]:
        registry.register(entry)

    # --- Main loop ---

    while not shutdown_event.is_set():
        try:
            raw = input("\n> ").strip()
            if not raw:
                continue

            # Strip leading / if present (commands work with or without it)
            if raw.startswith("/"):
                raw = raw[1:]

            parts = raw.split(maxsplit=1)
            cmd_name = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            output = registry.dispatch(cmd_name, arg)
            if output:
                print(output)

        except (KeyboardInterrupt, EOFError):
            shutdown_fn()
            return
