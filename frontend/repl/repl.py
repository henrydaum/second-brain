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
from event_bus import bus
from event_channels import APPROVAL_REQUESTED, APPROVAL_RESOLVED

logger = logging.getLogger("REPL")


# =================================================================
# REPL LOOP
# =================================================================

def run_repl(ctrl, shutdown_fn, shutdown_event: threading.Event,
             tool_registry, services, config, root_dir: Path):
    agent = None
    conversation_ref = {"id": None}

    # --- Conversation persistence ---
    def _set_conversation_id(conv_id):
        conversation_ref["id"] = conv_id

    def _on_agent_message(msg: dict):
        """Persist conversation messages to DB (same pattern as GUI/Telegram)."""
        import json
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if conversation_ref["id"] is None:
            title = (content[:80].replace("\n", " ").strip()
                     if role == "user" else "New conversation")
            conversation_ref["id"] = ctrl.db.create_conversation(title)

        save_content = content
        if msg.get("tool_calls"):
            save_content = json.dumps({
                "content": content,
                "tool_calls": msg["tool_calls"],
            })

        ctrl.db.save_message(
            conversation_ref["id"], role, save_content,
            tool_call_id=msg.get("tool_call_id"),
            tool_name=msg.get("name"),
        )

        if role == "assistant" and not msg.get("tool_calls"):
            ctrl.maybe_generate_conversation_title_async(conversation_ref["id"])

    # --- Console-based approval (bus subscriber). GUI subscriber wins if present. ---
    _pending_approvals = []
    
    def _repl_approve_handler(req: 'ApprovalRequest'):
        if req.is_resolved:
            return  # another subscriber already answered
            
        print(f"\n\n--- Agent wants to run a command ---")
        print(f"  Command:  {req.command}")
        print(f"  Reason:   {req.reason}")
        print(f"  (Type '/allow' or '/deny' to respond)")
        print("> ", end="", flush=True)
        _pending_approvals.append(req)

    def _on_approval_resolved(req: 'ApprovalRequest'):
        if req in _pending_approvals:
            _pending_approvals.remove(req)
            print(f"\n(Request resolved via another frontend)\n> ", end="", flush=True)

    bus.subscribe(APPROVAL_REQUESTED, _repl_approve_handler)
    bus.subscribe(APPROVAL_RESOLVED, _on_approval_resolved)

    # --- Build command registry (shared + REPL-specific) ---
    registry = CommandRegistry()
    register_core_commands(registry, ctrl, services, tool_registry, root_dir,
                           get_agent=lambda: agent,
                           set_conversation_id=_set_conversation_id)

    # --- REPL-specific commands ---

    def _chat_handler(_arg):
        nonlocal agent
        llm = services.get("llm")
        if llm is None or not llm.loaded:
            return "LLM service not loaded. Run '/load llm' first."

        conversation_ref["id"] = None  # fresh conversation on /chat entry
        agent = Agent(
            llm, tool_registry, config,
            system_prompt=lambda: build_system_prompt(
                ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services
            ),
            on_message=_on_agent_message,
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
        while _pending_approvals and _pending_approvals[0].is_resolved:
            _pending_approvals.pop(0)
            
        if not _pending_approvals:
            return "No pending approvals."
            
        req = _pending_approvals.pop(0)
        req.resolve(True)
        return "Command allowed."

    def _deny_handler(_arg):
        while _pending_approvals and _pending_approvals[0].is_resolved:
            _pending_approvals.pop(0)
            
        if not _pending_approvals:
            return "No pending approvals."
            
        req = _pending_approvals.pop(0)
        req.resolve(False)
        return "Command denied."

    for entry in [
        CommandEntry("chat",  "Enter interactive chat mode", handler=_chat_handler),
        CommandEntry("quit",  "Shutdown", handler=_quit_handler),
        CommandEntry("exit",  "Shutdown", handler=_quit_handler),
        CommandEntry("allow", "Approve a pending command", handler=_allow_handler, hide_from_help=True),
        CommandEntry("deny",  "Deny a pending command", handler=_deny_handler, hide_from_help=True),
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
