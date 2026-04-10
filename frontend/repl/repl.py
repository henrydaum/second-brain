"""
REPL.

Simple command loop that maps user input to the shared CommandRegistry.
Runs on its own daemon thread so it never blocks the dispatch loop.

The ``/chat`` subcommand enters a natural-language chat mode that uses
:func:`gui.dispatch.route_input` — the same channel as the GUI and API.
Slash commands work inside chat mode too.
"""

import logging
import threading
from pathlib import Path

from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from frontend.shared.commands import CommandEntry, CommandRegistry, register_core_commands
from frontend.shared.dispatch import route_input

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

    # --- Build command registry (shared + REPL-specific) ---
    registry = CommandRegistry()
    register_core_commands(registry, ctrl, services, tool_registry, root_dir,
                           get_agent=lambda: agent)

    # --- REPL-specific commands ---

    def _chat_handler(_arg):
        nonlocal agent
        llm = services.get("llm")
        if llm is None or not llm.loaded:
            return "LLM service not loaded. Run '/load llm' first."

        agent = Agent(
            llm, tool_registry, config,
            system_prompt=lambda: build_system_prompt(
                ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services
            ),
            approve_command=_repl_approve_command,
        )
        logger.info("Agent initialized.")

        print("Entering chat mode. Type 'exit' to return to REPL.")
        print("Slash commands (e.g. /services) work here too.")
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

            try:
                result = route_input(user_input, registry, agent)
                if result.type == "chat":
                    print(f"\nassistant> {result.text}\n")
                elif result.text:
                    print(result.text)
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
        CommandEntry("chat",  "Enter interactive chat mode", handler=_chat_handler),
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
