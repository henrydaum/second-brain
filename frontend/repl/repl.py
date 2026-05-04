"""Terminal frontend built on the shared frontend runtime."""

import logging
import threading
from pathlib import Path

from plugins.frontends.helpers.command_registry import CommandEntry
from frontend.platforms.platform_repl import ReplPlatformAdapter
from frontend.runtime import FrontendRuntime
from frontend.types import FrontendEvent, FrontendSession

logger = logging.getLogger("REPL")


# =================================================================
# REPL LOOP
# =================================================================

def run_repl(ctrl, shutdown_fn, shutdown_event: threading.Event,
             tool_registry, services, config, root_dir: Path,
             runtime: FrontendRuntime | None = None,
             adapter: ReplPlatformAdapter | None = None):
    adapter = adapter or ReplPlatformAdapter(
        ctrl, shutdown_fn, shutdown_event, tool_registry, services, config, root_dir
    )
    runtime = runtime or FrontendRuntime(ctrl, services, config, tool_registry, root_dir)
    if adapter.runtime is None:
        runtime.register_adapter(adapter)
    session = adapter.default_session() or FrontendSession("repl", "local", "console")
    registry = runtime.create_registry(session)

    # --- REPL-specific commands ---

    def _chat_handler(_arg):
        llm = services.get("llm")
        if llm is None or not llm.loaded:
            return "LLM service is not loaded. Run /load llm to load it."

        runtime.reset_session(session)
        runtime.refresh_agent(session)
        logger.info("Agent initialized.")

        print("Entering chat mode. Type 'exit' to return to REPL.")
        print(f"Agent: {config.get("active_agent_profile", "default")}")
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
                result = runtime.handle_frontend_event(FrontendEvent(
                    type="slash_command" if user_input.startswith("/") else "chat_message",
                    session=session,
                    text=user_input,
                ), registry)
                if result.type == "chat":
                    print(f"\nassistant> {result.text}\n")
                    if result.attachments:
                        print(f"  [{len(result.attachments)} attachment(s)]")
                        for p in result.attachments:
                            print(f"    • {p}")
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

    def _allow_handler(_arg):
        return runtime.handle_frontend_event(FrontendEvent(
            type="approval_response",
            session=session,
            payload={"approved": True},
        ), registry).text

    def _deny_handler(_arg):
        return runtime.handle_frontend_event(FrontendEvent(
            type="approval_response",
            session=session,
            payload={"approved": False},
        ), registry).text

    for entry in [
        CommandEntry("chat",  "Enter interactive chat mode", handler=_chat_handler),
        CommandEntry("quit",  "Shutdown", handler=_quit_handler),
        CommandEntry("exit",  "Shutdown", handler=_quit_handler),
        CommandEntry("allow", "Approve a pending command", handler=_allow_handler, hide_from_help=True),
        CommandEntry("deny",  "Deny a pending command", handler=_deny_handler, hide_from_help=True),
    ]:
        registry.register(entry)

    # --- Main loop ---

    print("Second Brain REPL ready. Type /help for commands, /chat for agent mode, /quit to exit.")

    while not shutdown_event.is_set():
        try:
            raw = input("\n").strip()
            if not raw:
                continue

            result = runtime.handle_frontend_event(FrontendEvent(
                type="slash_command",
                session=session,
                text=raw,
            ), registry)
            if result.text:
                print(result.text)

        except KeyboardInterrupt:
            shutdown_fn()
            return
        except EOFError:
            logger.info("REPL stdin closed; stopping REPL without shutting down the app.")
            return
