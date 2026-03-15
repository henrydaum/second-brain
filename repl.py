"""
REPL.

Simple command loop that maps user input to Controller methods.
Runs on its own daemon thread so it never blocks the dispatch loop.
"""

import json
import logging
import threading
from pathlib import Path

from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt

logger = logging.getLogger("REPL")


# =================================================================
# FORMATTERS — turn structured Controller data into REPL strings
# =================================================================

def _truncate_cell(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_tool_result(result) -> str:
    if not result.success:
        return f"Error: {result.error}"

    data = result.data

    # Special handling for sql_query — render as a table
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        columns = data["columns"]
        rows = data["rows"]

        if not rows:
            return "(no results)"

        col_widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(_truncate_cell(str(val))))

        header = "  ".join(c.ljust(w) for c, w in zip(columns, col_widths))
        separator = "  ".join("-" * w for w in col_widths)
        lines = [header, separator]
        for row in rows:
            line = "  ".join(
                _truncate_cell(str(val)).ljust(w)
                for val, w in zip(row, col_widths)
            )
            lines.append(line)

        if data.get("truncated"):
            lines.append("  ... (results capped at 100 rows)")

        return "\n".join(lines)

    # Default: pretty-print as JSON
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)


def _format_services(services: list[dict]) -> str:
    if not services:
        return "No services registered."
    lines = []
    for s in services:
        status = "LOADED" if s["loaded"] else "unloaded"
        lines.append(f"  {s['name']:<20} {status:<10} {s['model_name']}")
    return "Services:\n" + "\n".join(lines)


def _format_tasks(tasks: list[dict]) -> str:
    if not tasks:
        return "No tasks registered."
    lines = []
    for t in tasks:
        c = t["counts"]
        paused = " [PAUSED]" if t["paused"] else ""
        svc = f"  needs: {t['requires_services']}" if t["requires_services"] else ""
        lines.append(
            f"  {t['name']:<22} "
            f"P:{c['PENDING']:<4} "
            f"R:{c['PROCESSING']:<4} "
            f"D:{c['DONE']:<4} "
            f"F:{c['FAILED']:<4}"
            f"{paused}{svc}"
        )
    return "Tasks:\n" + "\n".join(lines)


def _format_stats(stats: dict) -> str:
    lines = ["Files by modality:"]
    files = stats.get("files", {})
    if files:
        for mod, count in sorted(files.items()):
            lines.append(f"  {mod:<12} {count}")
    else:
        lines.append("  (none)")
    lines.append(f"  {'total':<12} {sum(files.values()) if files else 0}")
    lines.append("")
    lines.append("Task queue:")
    tasks = stats.get("tasks", {})
    if tasks:
        for name, counts in sorted(tasks.items()):
            paused = " [PAUSED]" if counts.get("paused") else ""
            lines.append(
                f"  {name:<22} "
                f"P:{counts['PENDING']:<4} "
                f"R:{counts['PROCESSING']:<4} "
                f"D:{counts['DONE']:<4} "
                f"F:{counts['FAILED']:<4}"
                f"{paused}"
            )
    else:
        lines.append("  (empty)")
    return "\n".join(lines)


def _format_tools(tools: list[dict]) -> str:
    if not tools:
        return "No tools registered."
    lines = []
    for t in tools:
        status = "" if t["agent_enabled"] else " [DISABLED]"
        svc = f"  needs: {t['requires_services']}" if t["requires_services"] else ""
        lines.append(f"  {t['name']}{status}{svc}")
        desc = t["description"]
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"    {desc}")
        params = t["parameters"].get("properties", {})
        required = set(t["parameters"].get("required", []))
        if params:
            parts = [f"{p}{'*' if p in required else ''}" for p in params]
            lines.append(f"    args: {', '.join(parts)}")
        lines.append("")
    return "Tools:\n" + "\n".join(lines)


def _format_help(commands: list[dict]) -> str:
    return "Commands:\n" + "\n".join(
        f"  {c['command']:<25} {c['description']}" for c in commands
    )


# =================================================================
# REPL LOOP
# =================================================================

def run_repl(ctrl, shutdown_fn, shutdown_event: threading.Event,
             tool_registry, services, config, root_dir: Path):
    agent = None

    # --- Command handlers (one per REPL command) ---

    def cmd_help(arg):
        print(_format_help(ctrl.help()))

    def cmd_pipeline(arg):
        print(ctrl.orchestrator.dependency_pipeline_graph())

    def cmd_services(arg):
        print(_format_services(ctrl.list_services()))

    def cmd_load(arg):
        print(ctrl.load_service(arg) if arg else "Usage: load <service_name>")

    def cmd_unload(arg):
        print(ctrl.unload_service(arg) if arg else "Usage: unload <service_name>")

    def cmd_tasks(arg):
        print(_format_tasks(ctrl.list_tasks()))

    def cmd_pause(arg):
        print(ctrl.pause_task(arg) if arg else "Usage: pause <task_name>")

    def cmd_unpause(arg):
        print(ctrl.unpause_task(arg) if arg else "Usage: unpause <task_name>")

    def cmd_reset(arg):
        print(ctrl.reset_task(arg) if arg else "Usage: reset <task_name>")

    def cmd_retry(arg):
        if arg.lower() == "all":
            print(ctrl.retry_all())
        elif arg:
            print(ctrl.retry_task(arg))
        else:
            print("Usage: retry <task_name> | retry all")

    def cmd_stats(arg):
        print(_format_stats(ctrl.stats()))

    def cmd_tools(arg):
        print(_format_tools(ctrl.list_tools()))

    def cmd_enable(arg):
        print(ctrl.enable_tool(arg) if arg else "Usage: enable <tool_name>")

    def cmd_disable(arg):
        print(ctrl.disable_tool(arg) if arg else "Usage: disable <tool_name>")

    def cmd_call(arg):
        if not arg:
            print("Usage: call <tool_name> {\"arg\": \"value\"}")
            print("Example: call sql_query {\"sql\": \"SELECT * FROM files LIMIT 5\"}")
            return

        call_parts = arg.split(maxsplit=1)
        tool_name = call_parts[0]
        raw_args = call_parts[1] if len(call_parts) > 1 else "{}"

        try:
            kwargs = json.loads(raw_args)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON arguments: {e}")
            print("Expected format: call <tool_name> {\"key\": \"value\"}")
            return

        print(_format_tool_result(ctrl.call_tool(tool_name, kwargs)))

    def cmd_reload(arg):
        print(ctrl.reload_plugins(root_dir))

    def cmd_chat(arg):
        nonlocal agent
        llm = services.get("llm")
        if llm is None or not llm.loaded:
            print("LLM service not loaded. Run 'load llm' first.")
            return

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

    # --- Command dispatch table ---

    commands = {
        "help": cmd_help, "services": cmd_services,
        "load": cmd_load, "unload": cmd_unload,
        "tasks": cmd_tasks, "pause": cmd_pause,
        "unpause": cmd_unpause, "reset": cmd_reset,
        "retry": cmd_retry, "stats": cmd_stats,
        "tools": cmd_tools, "call": cmd_call,
        "enable": cmd_enable, "disable": cmd_disable,
        "chat": cmd_chat, "pipeline": cmd_pipeline,
        "reload": cmd_reload,
    }

    # --- Main loop ---

    while not shutdown_event.is_set():
        try:
            raw = input("\n> ").strip()
            if not raw:
                continue

            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("quit", "exit"):
                shutdown_fn()
                return

            handler = commands.get(cmd)
            if handler:
                handler(arg)
            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")

        except (KeyboardInterrupt, EOFError):
            shutdown_fn()
            return
