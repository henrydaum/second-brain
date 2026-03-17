"""
Flet GUI for the Data Refinery.

A chat-style window that mirrors the REPL: command mode for system
commands, chat mode for LLM conversation with inline tool result
rendering via modality-aware widgets.
"""

import json
import logging
import threading
from pathlib import Path

import flet as ft

from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from gui.renderers import render_paths

logger = logging.getLogger("GUI")


# ===================================================================
# REPL FORMATTERS (reused from repl.py logic)
# ===================================================================

def _truncate_cell(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_tool_result(result) -> str:
    if not result.success:
        return f"Error: {result.error}"
    data = result.data
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
            line = "  ".join(_truncate_cell(str(val)).ljust(w) for val, w in zip(row, col_widths))
            lines.append(line)
        if data.get("truncated"):
            lines.append("  ... (results capped at 100 rows)")
        return "\n".join(lines)
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
            f"P:{c['PENDING']:<4} R:{c['PROCESSING']:<4} "
            f"D:{c['DONE']:<4} F:{c['FAILED']:<4}{paused}{svc}"
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
                f"P:{counts['PENDING']:<4} R:{counts['PROCESSING']:<4} "
                f"D:{counts['DONE']:<4} F:{counts['FAILED']:<4}{paused}"
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


# ===================================================================
# MESSAGE WIDGETS
# ===================================================================

def _system_message(text: str) -> ft.Container:
    """A monospace text block for command output."""
    return ft.Container(
        content=ft.Text(text, font_family="Consolas", size=12, selectable=True),
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        margin=ft.margin.only(bottom=4),
        border_radius=8,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


def _user_bubble(text: str) -> ft.Container:
    """Right-aligned user chat bubble."""
    return ft.Container(
        content=ft.Text(text, size=13, color=ft.Colors.ON_PRIMARY),
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        margin=ft.margin.only(left=80, bottom=4),
        border_radius=ft.border_radius.only(
            top_left=16, top_right=16, bottom_left=16, bottom_right=4,
        ),
        bgcolor=ft.Colors.PRIMARY,
        alignment=ft.alignment.center_right,
    )


def _assistant_bubble(text: str) -> ft.Container:
    """Left-aligned assistant chat bubble."""
    return ft.Container(
        content=ft.Markdown(
            value=text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        ) if any(m in text[:200] for m in ("# ", "**", "- ", "```", "| ")) else
        ft.Text(text, size=13, selectable=True),
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        margin=ft.margin.only(right=80, bottom=4),
        border_radius=ft.border_radius.only(
            top_left=16, top_right=16, bottom_left=4, bottom_right=16,
        ),
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
    )


def _tool_call_card(tool_name: str, success: bool) -> ft.Container:
    """Small inline card showing a tool was called."""
    icon = ft.Icons.CHECK_CIRCLE if success else ft.Icons.ERROR
    color = ft.Colors.PRIMARY if success else ft.Colors.ERROR
    return ft.Container(
        content=ft.Row(
            controls=[
                ft.Icon(icon, size=14, color=color),
                ft.Text(f"Tool: {tool_name}", size=11, italic=True),
            ],
            spacing=6,
        ),
        padding=ft.padding.symmetric(horizontal=10, vertical=4),
        margin=ft.margin.only(bottom=2),
        border_radius=6,
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
    )


# ===================================================================
# MAIN APP
# ===================================================================

def run_gui(ctrl, shutdown_fn, shutdown_event: threading.Event,
            tool_registry, services, config, root_dir: Path,
            on_page_ready=None):
    """
    Launch the Flet GUI. Blocks until the window is closed.

    on_page_ready: optional callback(page, close_app) called once the page is set up,
                   so the caller (main.pyw) can wire tray actions to the GUI.
    """

    def main_view(page: ft.Page):
        page.title = "The Data Refinery"
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 800
        page.window.height = 700
        page.window.min_width = 500
        page.window.min_height = 400

        # --- Minimize to tray on close (X button) ---
        page.window.prevent_close = True

        def on_window_event(e):
            if e.data == "close":
                # Hide to tray instead of closing
                page.window.visible = False
                page.update()

        page.window.on_event = on_window_event

        def close_app():
            """Actually close the window and exit Flet's event loop."""
            page.window.prevent_close = False
            page.window.close()
            page.update()

        # Expose page and close function to caller for tray integration
        if on_page_ready:
            on_page_ready(page, close_app)

        # --- State ---
        chat_mode = {"active": False}
        agent_ref = {"agent": None}
        processing = {"value": False}

        # --- Message list ---
        message_list = ft.ListView(
            expand=True,
            spacing=4,
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            auto_scroll=True,
        )

        # Welcome message
        message_list.controls.append(_system_message(
            "The Data Refinery\n"
            "Type 'help' for commands, 'chat' to talk to the LLM, and 'quit' to shutdown."
        ))

        # --- Input bar ---
        input_field = ft.TextField(
            hint_text="Enter command...",
            expand=True,
            border_radius=20,
            text_size=13,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=8),
            on_submit=lambda e: handle_input(e),
        )

        send_button = ft.IconButton(
            icon=ft.Icons.SEND,
            on_click=lambda e: handle_input(e),
            icon_size=20,
        )

        input_row = ft.Row(
            controls=[input_field, send_button],
            spacing=8,
        )

        # --- Command dispatch (mirrors REPL) ---
        def handle_command(cmd: str, arg: str):
            """Handle a REPL command, append output to message list."""

            if cmd in ("quit", "exit"):
                close_app()
                return

            if cmd == "chat":
                llm = services.get("llm")
                if llm is None or not llm.loaded:
                    message_list.controls.append(_system_message(
                        "LLM service not loaded. Run 'load llm' first."
                    ))
                    return

                chat_mode["active"] = True
                prompt = build_system_prompt(
                    ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services
                )
                agent_ref["agent"] = Agent(
                    llm, tool_registry, config,
                    system_prompt=prompt,
                    on_tool_result=on_tool_result,
                )
                input_field.hint_text = "Chat with the assistant... (type 'back' to exit)"
                message_list.controls.append(_system_message(
                    "Entering chat mode. Type 'back' to return."
                ))
                return

            # Map commands to controller methods
            handlers = {
                "help": lambda: _format_help(ctrl.help()),
                "services": lambda: _format_services(ctrl.list_services()),
                "tasks": lambda: _format_tasks(ctrl.list_tasks()),
                "stats": lambda: _format_stats(ctrl.stats()),
                "tools": lambda: _format_tools(ctrl.list_tools()),
                "pipeline": lambda: ctrl.orchestrator.dependency_pipeline_graph(),
                "load": lambda: ctrl.load_service(arg) if arg else "Usage: load <service_name>",
                "unload": lambda: ctrl.unload_service(arg) if arg else "Usage: unload <service_name>",
                "pause": lambda: ctrl.pause_task(arg) if arg else "Usage: pause <task_name>",
                "unpause": lambda: ctrl.unpause_task(arg) if arg else "Usage: unpause <task_name>",
                "reset": lambda: ctrl.reset_task(arg) if arg else "Usage: reset <task_name>",
                "retry": lambda: ctrl.retry_all() if arg and arg.lower() == "all"
                         else ctrl.retry_task(arg) if arg
                         else "Usage: retry <task_name> | retry all",
                "enable": lambda: ctrl.enable_tool(arg) if arg else "Usage: enable <tool_name>",
                "disable": lambda: ctrl.disable_tool(arg) if arg else "Usage: disable <tool_name>",
                "reload": lambda: ctrl.reload_plugins(root_dir),
            }

            if cmd == "call":
                if not arg:
                    message_list.controls.append(_system_message(
                        "Usage: call <tool_name> {\"arg\": \"value\"}"
                    ))
                    return
                call_parts = arg.split(maxsplit=1)
                tool_name = call_parts[0]
                raw_args = call_parts[1] if len(call_parts) > 1 else "{}"
                try:
                    kwargs = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    message_list.controls.append(_system_message(f"Invalid JSON: {e}"))
                    return

                result = ctrl.call_tool(tool_name, kwargs)
                message_list.controls.append(_tool_call_card(tool_name, result.success))
                message_list.controls.append(_system_message(_format_tool_result(result)))

                # Render result_paths if any
                if result.result_paths:
                    widget = render_paths(result.result_paths, page, config)
                    message_list.controls.append(widget)
                return

            handler = handlers.get(cmd)
            if handler:
                output = handler()
                if output:
                    message_list.controls.append(_system_message(str(output)))
            else:
                message_list.controls.append(_system_message(
                    f"Unknown command: '{cmd}'. Type 'help' for available commands."
                ))

        # --- Tool result callback (called from agent thread) ---
        def on_tool_result(tool_name: str, result):
            """Insert a tool card + rendered paths into the message list."""
            message_list.controls.append(_tool_call_card(tool_name, result.success))
            if result.result_paths:
                widget = render_paths(result.result_paths, page, config)
                message_list.controls.append(widget)
            page.update()

        # --- Chat handling (runs agent.chat in background) ---
        def send_chat(user_text: str):
            """Run agent.chat() in a background thread."""
            processing["value"] = True
            input_field.disabled = True
            send_button.disabled = True
            page.update()

            def run():
                try:
                    response = agent_ref["agent"].chat(user_text)
                    message_list.controls.append(_assistant_bubble(response))
                except Exception as e:
                    logger.error(f"Agent error: {e}")
                    message_list.controls.append(_system_message(f"Error: {e}"))
                finally:
                    processing["value"] = False
                    input_field.disabled = False
                    send_button.disabled = False
                    input_field.focus()
                    page.update()

            threading.Thread(target=run, daemon=True).start()

        # --- Input handler ---
        def handle_input(e):
            text = input_field.value.strip()
            if not text:
                return

            input_field.value = ""

            if chat_mode["active"]:
                # Chat mode
                if text.lower() in ("exit", "quit", "back"):
                    chat_mode["active"] = False
                    agent_ref["agent"] = None
                    input_field.hint_text = "Enter command..."
                    message_list.controls.append(_system_message("Exited chat mode."))
                    page.update()
                    return

                if text.lower() == "reset":
                    if agent_ref["agent"]:
                        agent_ref["agent"].reset()
                    message_list.controls.append(_system_message("(conversation history cleared)"))
                    page.update()
                    return

                message_list.controls.append(_user_bubble(text))
                page.update()
                send_chat(text)
            else:
                # Command mode
                message_list.controls.append(_system_message(f"> {text}"))
                parts = text.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""
                handle_command(cmd, arg)
                page.update()

            input_field.focus()

        # --- Layout ---
        page.add(
            ft.Column(
                controls=[
                    message_list,
                    ft.Container(
                        content=input_row,
                        padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    ),
                ],
                expand=True,
                spacing=0,
            )
        )

        input_field.focus()

    ft.app(target=main_view)
