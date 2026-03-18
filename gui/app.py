"""
Flet GUI for the Data Refinery.

A unified chat-first interface. Plain text goes to the LLM agent;
slash-prefixed commands (e.g. /services, /load llm) control the system.
An autocomplete popup appears when typing /.

Organisation
------------
The file has four top-level sections — REPL Formatters, Message Widgets,
Log Handler, and Main App.  Nearly all GUI state lives inside the
``run_gui`` → ``main_view`` closure so that inner functions can freely
read and mutate shared widgets without passing dozens of arguments.
"""

import collections
import json
import logging
import threading
from pathlib import Path

import flet as ft

from Stage_3.agent import Agent
from Stage_3.system_prompt import build_system_prompt
from gui.commands import CommandEntry, CommandRegistry
from gui.renderers import render_paths

logger = logging.getLogger("GUI")


# =====================================================================
# REPL FORMATTERS -- Plain-text formatting for command output
# =====================================================================

def _truncate_cell(text: str, max_len: int = 60) -> str:
    """Truncate *text* to *max_len*, appending '...' if clipped."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _format_tool_result(result) -> str:
    """Format a ToolResult for monospace display.

    Tabular data (columns + rows) is rendered as aligned columns;
    everything else falls back to pretty-printed JSON.
    """
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
    """Format the service list showing name, loaded/unloaded status, and model."""
    if not services:
        return "No services registered."
    lines = []
    for s in services:
        status = "LOADED" if s["loaded"] else "unloaded"
        lines.append(f"  {s['name']:<20} {status:<10} {s['model_name']}")
    return "Services:\n" + "\n".join(lines)


def _format_tasks(tasks: list[dict]) -> str:
    """Format task list with queue counts (Pending/Processing/Done/Failed)."""
    if not tasks:
        return "No tasks registered."
    lines = []
    for t in tasks:
        c = t["counts"]
        paused = " [PAUSED]" if t["paused"] else ""
        svc = f"  needs: {t['requires_services']}" if t["requires_services"] else ""
        lines.append(
            f"  {t['name']:<22} "
            f"P:{c['PENDING']:<8} R:{c['PROCESSING']:<8} "
            f"D:{c['DONE']:<8} F:{c['FAILED']:<8}{paused}{svc}"
        )
    return "Tasks:\n" + "\n".join(lines)


def _format_stats(stats: dict) -> str:
    """Format system overview: file counts by modality + task queue summaries."""
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
                f"P:{counts['PENDING']:<8} R:{counts['PROCESSING']:<8} "
                f"D:{counts['DONE']:<8} F:{counts['FAILED']:<8}{paused}"
            )
    else:
        lines.append("  (empty)")
    return "\n".join(lines)


def _format_tools(tools: list[dict]) -> str:
    """Format tool list with enabled/disabled status, descriptions, and parameters."""
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
    """Format the help output as an aligned two-column list."""
    return "Commands:\n" + "\n".join(
        f"  {c['command']:<25} {c['description']}" for c in commands
    )


# =====================================================================
# MESSAGE WIDGETS -- Flet container factories for chat bubbles & cards
# =====================================================================

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
    bubble = ft.Container(
        content=ft.Text(text, size=13, color=ft.Colors.ON_PRIMARY),
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        border_radius=ft.border_radius.only(
            top_left=16, top_right=16, bottom_left=16, bottom_right=4,
        ),
        bgcolor=ft.Colors.PRIMARY,
    )
    return ft.Container(
        content=bubble,
        alignment=ft.alignment.center_right,
        margin=ft.margin.only(left=80, bottom=4),
    )


def _assistant_bubble(text: str) -> ft.Container:
    """Left-aligned assistant chat bubble (auto-detects Markdown)."""
    # Heuristic: if the first 200 chars contain markdown indicators,
    # render as Markdown; otherwise use plain Text for speed.
    bubble = ft.Container(
        content=ft.Markdown(
            value=text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        ) if any(m in text[:200] for m in ("# ", "**", "- ", "```", "| ")) else
        ft.Text(text, size=13, selectable=True),
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        border_radius=ft.border_radius.only(
            top_left=16, top_right=16, bottom_left=4, bottom_right=16,
        ),
        bgcolor=ft.Colors.SECONDARY_CONTAINER,
    )
    return ft.Container(
        content=bubble,
        alignment=ft.alignment.center_left,
        margin=ft.margin.only(right=80, bottom=4),
    )


def _tool_call_card(tool_name: str, success: bool, content_control: ft.Control, initially_expanded: bool) -> ft.Container:
    """Expansion tile showing a tool was called and its result."""
    icon = ft.Icons.CHECK_CIRCLE if success else ft.Icons.ERROR
    color = ft.Colors.PRIMARY if success else ft.Colors.ERROR
    return ft.Container(
        content=ft.ExpansionTile(
            title=ft.Row(
                controls=[
                    ft.Icon(icon, size=14, color=color),
                    ft.Text(f"Tool Call: {tool_name}", size=14),
                ],
                spacing=6,
            ),
            initially_expanded=initially_expanded,
            controls=[
                ft.Container(
                    content=content_control,
                    padding=ft.padding.only(left=16, right=16, bottom=8, top=8)
                )
            ],
            dense=True
        ),
        margin=ft.margin.only(bottom=2),
        border_radius=0,
        border=None,
        bgcolor=None,
    )


# =====================================================================
# LOG HANDLER -- Captures log records into a bounded deque for GUI display
# =====================================================================

class GuiLogHandler(logging.Handler):
    """Thread-safe handler that stores formatted records and notifies the GUI."""

    def __init__(self, max_records=500):
        """Initialize with a bounded deque of *max_records* and no callback."""
        super().__init__()
        self._records = collections.deque(maxlen=max_records)
        self._on_record = None

    def set_callback(self, fn):
        """Register *fn(formatted_str, record)* to be invoked on each emit."""
        self._on_record = fn

    @property
    def records(self):
        return list(self._records)

    def emit(self, record):
        """Store the formatted record and notify the callback if set."""
        formatted = self.format(record)
        self._records.append((formatted, record))
        if self._on_record:
            try:
                self._on_record(formatted, record)
            except Exception:
                pass


# =====================================================================
# MAIN APP -- Entry point; run_gui() blocks until the window closes
# =====================================================================

def run_gui(ctrl, shutdown_fn, shutdown_event: threading.Event,
            tool_registry, services, config, root_dir: Path,
            on_page_ready=None, watcher=None):
    """
    Launch the Flet GUI. Blocks until the window is closed.

    on_page_ready: optional callback(page, close_app) called once the page is set up,
                   so the caller (main.pyw) can wire tray actions to the GUI.
    """

    def main_view(page: ft.Page):
        """Flet page builder -- all GUI state lives as closures within this function.

        Sub-sections (approximate line numbers):
        ─────────────────────────────────────────
        Page Setup & Window Config
        Close & Cleanup Handlers
        Log Handler Setup
        State & Message List
        Agent Lifecycle
        Tool Result Callback
        Command Registry
          ├ Slash command handlers
          ├ Dynamic tool form overlay
          └ Command registration table
        Autocomplete Overlay
        Input Field & Send Button
        Chat Handling
        Input Handler
        Layout (input bar, status bar, log dialog)
        Log Streaming
        Settings Overlay
        Final Assembly
        Startup (auto-load LLM)
        """

        # -----------------------------------------------------------------
        # PAGE SETUP & WINDOW CONFIG
        # -----------------------------------------------------------------
        page.title = "The Data Refinery"
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 800
        page.window.height = 700
        page.window.min_width = 500
        page.window.min_height = 400
        page.padding = 0
        page.window.center()

        # Minimize to tray on close (X button) instead of quitting
        page.window.prevent_close = True

        def on_window_event(e):
            """Intercept the close event and hide the window instead."""
            if e.data == "close":
                page.window.visible = False
                page.update()

        page.window.on_event = on_window_event

        # -----------------------------------------------------------------
        # CLOSE & CLEANUP
        # -----------------------------------------------------------------
        def close_app():
            """Actually close the window and exit Flet's event loop."""
            logging.getLogger().removeHandler(gui_handler)
            page.window.prevent_close = False
            page.window.close()
            page.update()

        # Expose page and close function to caller for tray integration
        if on_page_ready:
            on_page_ready(page, close_app)

        # -----------------------------------------------------------------
        # LOG HANDLER SETUP
        # -----------------------------------------------------------------
        gui_handler = GuiLogHandler()
        gui_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
            datefmt="%I:%M%p",
        ))
        logging.getLogger().addHandler(gui_handler)

        # -----------------------------------------------------------------
        # STATE & MESSAGE LIST
        # -----------------------------------------------------------------
        # Mutable containers (not bare variables) so that inner closures
        # can mutate shared state — a Python scoping requirement.
        agent_ref = {"agent": None}
        processing = {"value": False}

        message_list = ft.ListView(
            expand=True,
            spacing=4,
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            auto_scroll=True,
        )

        message_list.controls.append(_system_message(
            "The Data Refinery\n"
            "Type a message to chat, or / for commands.\n"
            "Loading LLM..."
        ))

        # -----------------------------------------------------------------
        # AGENT LIFECYCLE
        # -----------------------------------------------------------------
        def create_agent():
            """Build or rebuild the Agent from the currently loaded LLM."""
            llm = services.get("llm")
            if llm and llm.loaded:
                prompt = build_system_prompt(
                    ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services
                )
                agent_ref["agent"] = Agent(
                    llm, tool_registry, config,
                    system_prompt=prompt,
                    on_tool_result=on_tool_result,
                )

        # -----------------------------------------------------------------
        # TOOL RESULT CALLBACK (called from agent thread)
        # -----------------------------------------------------------------
        def on_tool_result(tool_name: str, result):
            """Insert a tool card + rendered paths into the message list."""
            if result.gui_display_paths:
                content = render_paths(result.gui_display_paths, page, config)
            elif result.llm_summary:
                content = _system_message(result.llm_summary)
            else:
                content = _system_message(_format_tool_result(result))

            message_list.controls.append(
                _tool_call_card(tool_name, result.success, content, initially_expanded=(tool_name == "render_files"))
            )
            page.update()

        # =============================================================
        # COMMAND REGISTRY -- Slash-command handlers and registration
        # =============================================================
        registry = CommandRegistry()

        # --- Slash command handlers ---

        def _help_handler(_arg):
            """Build the /help output from the command registry."""
            lines = ["Commands:"]
            for cmd in registry.all_commands():
                hint = f" {cmd.arg_hint}" if cmd.arg_hint else ""
                lines.append(f"  /{cmd.name}{hint:<20}  {cmd.description}")
            return "\n".join(lines)

        def _load_handler(arg):
            """Handle /load <service>. Side-effect: creates agent when LLM loads."""
            if not arg:
                return "Usage: /load <service_name>"
            result = ctrl.load_service(arg)
            if arg == "llm" and services.get("llm") and services["llm"].loaded:
                create_agent()
                return result + "\nAgent ready — you can chat now."
            return result

        def _unload_handler(arg):
            """Handle /unload <service>. Destroys agent if LLM is unloaded."""
            if not arg:
                return "Usage: /unload <service_name>"
            result = ctrl.unload_service(arg)
            if arg == "llm":
                agent_ref["agent"] = None
            return result

        def _clear_handler(_arg):
            """Handle /clear: reset the agent's conversation history."""
            if agent_ref["agent"]:
                agent_ref["agent"].reset()
            return "(conversation history cleared)"

        # --- Dynamic tool form overlay ---

        def _show_tool_form(tool_name: str, tool):
            """Build and show a glassmorphism overlay with dynamic form fields.

            Each parameter in the tool's JSON schema is mapped to an
            appropriate Flet widget (Dropdown for enums, Checkbox for
            booleans, TextField variants for numbers/strings/arrays/objects).
            """
            properties = tool.parameters.get("properties", {})
            required_set = set(tool.parameters.get("required", []))

            # Build form fields: {param_name: {"control": widget, "type": str}}
            fields = {}
            field_rows = []

            for param_name, prop in properties.items():
                schema_type = prop.get("type", "string")
                description = prop.get("description", "")
                default = prop.get("default")
                enum_values = prop.get("enum")
                is_required = param_name in required_set
                title_text = f"{param_name} *" if is_required else param_name

                # 1. Create the input widget based on JSON schema type
                if enum_values:
                    control = ft.Dropdown(
                        options=[ft.dropdown.Option(str(v)) for v in enum_values],
                        value=str(default) if default is not None else None,
                        dense=True,
                    )
                    fields[param_name] = {"control": control, "type": "enum"}
                elif schema_type == "boolean":
                    control = ft.Checkbox(
                        value=bool(default) if default is not None else False,
                    )
                    fields[param_name] = {"control": control, "type": "boolean"}
                elif schema_type == "integer":
                    control = ft.TextField(
                        input_filter=ft.NumbersOnlyInputFilter(),
                        value=str(default) if default is not None else "",
                        dense=True,
                    )
                    fields[param_name] = {"control": control, "type": "integer"}
                elif schema_type == "number":
                    control = ft.TextField(
                        input_filter=ft.InputFilter(regex_string=r"[0-9.\-]"),
                        value=str(default) if default is not None else "",
                        dense=True,
                    )
                    fields[param_name] = {"control": control, "type": "number"}
                elif schema_type == "array":
                    default_str = "\n".join(str(v) for v in default) if isinstance(default, list) else ""
                    control = ft.TextField(
                        value=default_str,
                        hint_text="One item per line, no quotations",
                        multiline=True,
                        min_lines=2,
                        max_lines=6,
                        dense=True,
                    )
                    fields[param_name] = {"control": control, "type": "array"}
                elif schema_type == "object":
                    default_str = json.dumps(default, indent=2) if default is not None else ""
                    control = ft.TextField(
                        value=default_str,
                        multiline=True,
                        min_lines=3,
                        hint_text="JSON object",
                        dense=True,
                    )
                    fields[param_name] = {"control": control, "type": "object"}
                else:  # string (default)
                    control = ft.TextField(
                        value=str(default) if default is not None else "",
                        dense=True,
                    )
                    fields[param_name] = {"control": control, "type": "string"}

                # 2. Construct the visual block: name → description → widget
                block_controls = [
                    ft.Text(title_text, size=13, weight=ft.FontWeight.W_500),
                ]
                if description:
                    block_controls.append(
                        ft.Text(description, size=11, color=ft.Colors.ON_SURFACE_VARIANT)
                    )
                block_controls.append(control)

                field_rows.append(
                    ft.Container(
                        content=ft.Column(controls=block_controls, spacing=2),
                        padding=ft.padding.only(bottom=10, left=20, right=20),
                    )
                )

            def _close(e=None):
                """Hide the tool form overlay."""
                overlay.visible = False
                page.update()

            def _execute(e):
                """Validate form fields, call the tool, and display results."""
                kwargs = {}
                has_error = False

                for param_name, info in fields.items():
                    control = info["control"]
                    schema_type = info["type"]
                    is_required = param_name in required_set

                    # Clear previous errors
                    if hasattr(control, "error_text"):
                        control.error_text = None

                    raw = control.value

                    if schema_type == "boolean":
                        kwargs[param_name] = bool(raw)
                        continue

                    if not raw or (isinstance(raw, str) and not raw.strip()):
                        if is_required:
                            if hasattr(control, "error_text"):
                                control.error_text = "Required"
                            has_error = True
                        continue  # skip empty optional fields

                    raw = raw.strip() if isinstance(raw, str) else raw

                    try:
                        if schema_type == "integer":
                            kwargs[param_name] = int(raw)
                        elif schema_type == "number":
                            kwargs[param_name] = float(raw)
                        elif schema_type == "array":
                            kwargs[param_name] = [s.strip() for s in raw.split("\n") if s.strip()]
                        elif schema_type == "object":
                            kwargs[param_name] = json.loads(raw)
                        else:  # string, enum
                            kwargs[param_name] = raw
                    except (ValueError, json.JSONDecodeError):
                        if hasattr(control, "error_text"):
                            control.error_text = "Invalid value"
                        has_error = True

                if has_error:
                    page.update()
                    return

                _close()

                result = ctrl.call_tool(tool_name, kwargs)
                if result.gui_display_paths:
                    content = render_paths(result.gui_display_paths, page, config)
                elif result.llm_summary:
                    content = _system_message(result.llm_summary)
                else:
                    content = _system_message(_format_tool_result(result))

                message_list.controls.append(
                    _tool_call_card(tool_name, result.success, content, initially_expanded=True)
                )
                page.update()

            # Fixed header controls (always visible)
            header_controls = [
                ft.Text(tool_name, size=18, weight=ft.FontWeight.BOLD),
            ]
            if tool.description:
                header_controls.append(ft.Container(height=4))
                header_controls.append(
                    ft.Text(tool.description, size=12, color=ft.Colors.ON_SURFACE_VARIANT)
                )
            header_controls.append(ft.Container(height=8))
            header_controls.append(ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT))
            header_controls.append(ft.Container(height=8))

            # Form card
            form_card = ft.Container(
                expand=True,
                bgcolor=ft.Colors.SURFACE,
                border_radius=12,
                padding=25,
                content=ft.Column(
                    expand=True,
                    spacing=0,
                    controls=[
                        # Fixed header + description
                        *header_controls,
                        # Scrollable fields only
                        ft.Column(
                            controls=field_rows,
                            scroll=ft.ScrollMode.AUTO,
                            spacing=0,
                            expand=True,
                        ),
                        # Footer buttons
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.TextButton("Cancel", on_click=_close),
                                ft.ElevatedButton("Execute", on_click=_execute),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ],
                ),
            )

            # Full-screen blurred backdrop
            overlay = ft.Container(
                expand=True,
                blur=(10, 10),
                bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                padding=40,
                content=form_card,
                visible=True,
            )

            page.overlay.append(overlay)
            page.update()

        def _call_handler(arg):
            """Handle /call <tool>: open the dynamic tool form overlay."""
            if not arg:
                return "Usage: /call <tool_name>"
            tool_name = arg.split()[0]
            tool = tool_registry.tools.get(tool_name)
            if tool is None:
                return f"Unknown tool: '{tool_name}'"
            _show_tool_form(tool_name, tool)
            return None  # output comes later via Execute

        def _quit_handler(_arg):
            """Handle /quit or /exit: close the app."""
            close_app()
            return None

        # --- Command registration table ---

        # Lambdas (not static lists) so completions reflect hot-reloaded plugins.
        _task_names = lambda: list(ctrl.orchestrator.tasks.keys())
        _service_names = lambda: list(services.keys())
        _tool_names = lambda: list(tool_registry.tools.keys())
        _retry_names = lambda: _task_names() + ["all"]

        for entry in [
            CommandEntry("help",     "Show available commands",               handler=_help_handler),
            CommandEntry("services", "List services and status",              handler=lambda _: _format_services(ctrl.list_services())),
            CommandEntry("load",     "Load a service",         "<service>",   handler=_load_handler,   arg_completions=_service_names),
            CommandEntry("unload",   "Unload a service",       "<service>",   handler=_unload_handler, arg_completions=_service_names),
            CommandEntry("tasks",    "List tasks with status counts",         handler=lambda _: _format_tasks(ctrl.list_tasks())),
            CommandEntry("pipeline", "Show task dependency graph",            handler=lambda _: ctrl.orchestrator.dependency_pipeline_graph()),
            CommandEntry("pause",    "Pause a task",           "<task>",      handler=lambda a: ctrl.pause_task(a) if a else "Usage: /pause <task_name>",       arg_completions=_task_names),
            CommandEntry("unpause",  "Unpause a task",         "<task>",      handler=lambda a: ctrl.unpause_task(a) if a else "Usage: /unpause <task_name>",   arg_completions=_task_names),
            CommandEntry("reset",    "Reset a task to PENDING", "<task>",     handler=lambda a: ctrl.reset_task(a) if a else "Usage: /reset <task_name>",       arg_completions=_task_names),
            CommandEntry("retry",    "Retry failed entries",   "<task>|all",  handler=lambda a: ctrl.retry_all() if a and a.lower() == "all" else ctrl.retry_task(a) if a else "Usage: /retry <task_name> | /retry all", arg_completions=_retry_names),
            CommandEntry("tools",    "List registered tools",                 handler=lambda _: _format_tools(ctrl.list_tools())),
            CommandEntry("enable",   "Enable a tool for agent use",  "<tool>", handler=lambda a: ctrl.enable_tool(a) if a else "Usage: /enable <tool_name>",   arg_completions=_tool_names),
            CommandEntry("disable",  "Disable a tool",         "<tool>",      handler=lambda a: ctrl.disable_tool(a) if a else "Usage: /disable <tool_name>",  arg_completions=_tool_names),
            CommandEntry("call",     "Call a tool directly",   "<tool>",        handler=_call_handler, arg_completions=_tool_names),
            CommandEntry("reload",   "Hot-reload tasks and tools",            handler=lambda _: ctrl.reload_plugins(root_dir)),
            CommandEntry("stats",    "System overview",                       handler=lambda _: _format_stats(ctrl.stats())),
            CommandEntry("clear",    "Clear chat conversation history",       handler=_clear_handler),
            CommandEntry("quit",     "Shutdown",                              handler=_quit_handler),
            CommandEntry("exit",     "Shutdown",                              handler=_quit_handler),
            CommandEntry("config",   "Open settings panel",                   handler=lambda _: _show_settings()),
            CommandEntry("settings", "Open settings panel",                   handler=lambda _: _show_settings()),
        ]:
            registry.register(entry)

        # =============================================================
        # AUTOCOMPLETE OVERLAY -- Popup that filters commands/args as
        # the user types after '/'
        # =============================================================
        autocomplete_list = ft.ListView(spacing=0, padding=0)

        autocomplete_overlay = ft.Container(
            content=autocomplete_list,
            bgcolor=ft.Colors.SURFACE,
            border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
            border_radius=8,
            padding=ft.padding.symmetric(vertical=4),
            shadow=ft.BoxShadow(
                spread_radius=0, blur_radius=8,
                color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                offset=ft.Offset(0, -2),
            ),
            # Height is set dynamically in _update_autocomplete()
            visible=False,
        )

        def _update_autocomplete(text: str):
            """Show/hide and populate the autocomplete popup based on input text.

            Two phases:
            - Phase 1 (no space yet): match command names by prefix.
            - Phase 2 (space present): match argument values for the command.
            """
            if not text.startswith("/"):
                autocomplete_overlay.visible = False
                return

            body = text[1:]  # strip leading /
            has_space = " " in body

            if not has_space:
                # Phase 1: complete command names
                matches = registry.get_completions(body)
                if not matches:
                    autocomplete_overlay.visible = False
                    return

                autocomplete_list.controls.clear()
                for cmd in matches:
                    hint = f" {cmd.arg_hint}" if cmd.arg_hint else ""
                    tile = ft.Container(
                        content=ft.Column(
                            controls=[
                                ft.Text(f"/{cmd.name}{hint}", size=13, weight=ft.FontWeight.W_500),
                                ft.Text(cmd.description, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                            ],
                            spacing=2,
                        ),
                        padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        on_click=lambda e, n=cmd.name: _select_command(n),
                        ink=True,
                        height=56,  # Fixed height for predictable layout math
                    )
                    autocomplete_list.controls.append(tile)

                # Each command tile is 56px; add 8px for container padding.
                # Cap at 272px (~4.8 tiles) to avoid overwhelming the input area.
                calculated_height = (len(matches) * 56) + 8
                autocomplete_overlay.height = min(calculated_height, 272)
                autocomplete_overlay.visible = True
            else:
                # Phase 2: complete argument values
                parts = body.split(maxsplit=1)
                cmd_name = parts[0].lower()
                partial_arg = parts[1] if len(parts) > 1 else ""

                entry = registry._commands.get(cmd_name)
                if not entry or not entry.arg_completions:
                    autocomplete_overlay.visible = False
                    return

                candidates = entry.arg_completions()
                if partial_arg:
                    candidates = [c for c in candidates if c.lower().startswith(partial_arg.lower())]

                if not candidates:
                    autocomplete_overlay.visible = False
                    return

                autocomplete_list.controls.clear()
                for name in candidates:
                    tile = ft.Container(
                        content=ft.Text(name, size=13, weight=ft.FontWeight.W_500),
                        padding=ft.padding.symmetric(horizontal=12, vertical=10),
                        on_click=lambda e, cmd=cmd_name, arg=name: _select_arg(cmd, arg),
                        ink=True,
                        height=40,  # Fixed height for predictable layout math
                    )
                    autocomplete_list.controls.append(tile)

                # Each arg tile is 40px; same 8px padding, same 272px cap.
                calculated_height = (len(candidates) * 40) + 8
                autocomplete_overlay.height = min(calculated_height, 272)
                autocomplete_overlay.visible = True

        def _select_command(name: str):
            """Fill the input with the selected command and update the popup."""
            input_field.value = f"/{name} "
            _update_autocomplete(input_field.value)
            input_field.focus()
            page.update()

        def _select_arg(cmd_name: str, arg_value: str):
            """Fill the input with command + selected arg and hide the popup."""
            input_field.value = f"/{cmd_name} {arg_value}"
            autocomplete_overlay.visible = False
            input_field.focus()
            page.update()

        def _on_input_change(e):
            """Called on every keystroke in the input field."""
            _update_autocomplete(input_field.value or "")
            page.update()

        # -----------------------------------------------------------------
        # INPUT FIELD & SEND BUTTON
        # -----------------------------------------------------------------
        input_field = ft.TextField(
            label="Message the assistant, or type / for commands...",
            expand=True,
            border_radius=24,
            shift_enter=True,
            text_size=13,
            multiline=True,
            min_lines=1,
            max_lines=10,
            max_length=4096,
            focused_border_width=2,
            content_padding=ft.padding.symmetric(horizontal=16, vertical=8),
            on_submit=lambda e: handle_input(e),
            on_change=_on_input_change,
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

        # =============================================================
        # CHAT HANDLING -- Sends user text to agent in a background thread
        # =============================================================
        def send_chat(user_text: str):
            """Run agent.chat() in a background thread, re-enable input on completion."""
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

        # =============================================================
        # INPUT HANDLER -- Routes /commands to registry, plain text to agent
        # =============================================================
        def handle_input(e):
            """Unified input handler: routes /commands to registry, plain text to agent."""
            text = (input_field.value or "").strip()
            if not text:
                return

            input_field.value = ""
            autocomplete_overlay.visible = False

            if text.startswith("/"):
                # --- Slash command ---
                cmd_text = text[1:]  # strip leading /
                parts = cmd_text.split(maxsplit=1)
                cmd_name = parts[0].lower() if parts else ""
                arg = parts[1].strip() if len(parts) > 1 else ""

                message_list.controls.append(
                    _system_message(f"> /{cmd_name}" + (f" {arg}" if arg else ""))
                )

                output = registry.dispatch(cmd_name, arg)
                if output:
                    message_list.controls.append(_system_message(str(output)))
                page.update()
            else:
                # --- Chat message to LLM ---
                if not agent_ref["agent"]:
                    message_list.controls.append(_system_message(
                        "LLM is not loaded. Use /load llm to load it, "
                        "or /services to check status."
                    ))
                    page.update()
                    return

                message_list.controls.append(_user_bubble(text))
                page.update()
                send_chat(text)

            input_field.focus()

        # =============================================================
        # LAYOUT -- Widget assembly: input bar, status bar, log dialog
        # =============================================================

        # --- Input bar ---
        input_bar = ft.Container(
            content=input_row,
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
        )

        # --- Bottom status bar ---
        STATUS_BAR_HEIGHT = 24

        def _level_color(levelno):
            """Map log level to a Flet color: red for ERROR, amber for WARNING, grey otherwise."""
            if levelno >= logging.ERROR:
                return ft.Colors.ERROR
            if levelno >= logging.WARNING:
                return ft.Colors.AMBER
            return ft.Colors.OUTLINE

        latest_log_text = ft.Text(
            "",
            size=11,
            font_family="Consolas",
            color=ft.Colors.OUTLINE,
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )

        MAX_LOG_ENTRIES = 200

        log_list = ft.ListView(
            spacing=3,
            auto_scroll=True,
            width=9999,  # Forces fill of available horizontal space (Flet quirk)
        )

        # Position the autocomplete popup just above the input bar + status bar
        autocomplete_container = ft.Container(
            content=autocomplete_overlay,
            bottom=60 + STATUS_BAR_HEIGHT,
            left=12,
            right=12,
        )

        # --- Log dialog overlay (opened by clicking status bar) ---
        _log_overlay_ref = {"overlay": None}

        def _show_log_dialog(e=None):
            """Open (or re-show) the full-screen log viewer overlay."""
            if _log_overlay_ref["overlay"] is not None:
                _log_overlay_ref["overlay"].visible = True
                page.update()
                return

            def _close(e=None):
                _log_overlay_ref["overlay"].visible = False
                page.update()

            log_card = ft.Container(
                expand=True,
                bgcolor=ft.Colors.SURFACE,
                border_radius=12,
                padding=25,
                content=ft.Column(
                    expand=True,
                    spacing=0,
                    controls=[
                        ft.Row(
                            controls=[
                                ft.Text("Logs", size=18, weight=ft.FontWeight.BOLD),
                                ft.IconButton(
                                    icon=ft.Icons.CLOSE,
                                    icon_size=20,
                                    on_click=_close,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        ft.Container(height=8),
                        ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                        ft.Container(height=8),
                        ft.Container(
                            content=log_list,
                            expand=True,
                            padding=ft.padding.only(left=8, right=8),
                        ),
                    ],
                ),
            )

            overlay = ft.Container(
                expand=True,
                blur=(10, 10),
                bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                padding=40,
                content=log_card,
                visible=True,
            )

            _log_overlay_ref["overlay"] = overlay
            page.overlay.append(overlay)
            page.update()

        status_row = ft.Container(
            content=ft.Row(
                controls=[latest_log_text],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=STATUS_BAR_HEIGHT,
            padding=ft.padding.only(left=8, right=20),
            on_click=_show_log_dialog,
        )

        status_panel = ft.Container(
            content=status_row,
            height=STATUS_BAR_HEIGHT,
            bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        # =============================================================
        # LOG STREAMING -- Thread-safe buffered log display
        # =============================================================
        # Pattern: _on_log_record (called from ANY thread) just buffers.
        # _flush_log (called by a 250ms Timer) drains the buffer and does
        # a single page.update(), avoiding per-record UI thrashing.
        _prev_log = {"formatted": None, "record": None}
        _log_buffer = []
        _log_lock = threading.Lock()
        _log_flush_scheduled = {"value": False}

        def _flush_log():
            """Drain the log buffer, append entries to log_list, update status bar."""
            with _log_lock:
                _log_flush_scheduled["value"] = False
                batch = list(_log_buffer)
                _log_buffer.clear()
                latest_fmt = _prev_log["formatted"]
                latest_rec = _prev_log["record"]

            for fmt, rec in batch:
                log_list.controls.append(
                    ft.Text(
                        fmt, size=11, font_family="Consolas",
                        color=_level_color(rec.levelno),
                        selectable=True, no_wrap=True,
                    )
                )

            # Cap oldest entries
            overflow = len(log_list.controls) - MAX_LOG_ENTRIES
            if overflow > 0:
                del log_list.controls[:overflow]

            if latest_fmt:
                latest_log_text.value = latest_fmt.split("\n", 1)[0]
                latest_log_text.color = _level_color(latest_rec.levelno)

            try:
                page.update()
            except Exception:
                pass

        def _on_log_record(formatted, record):
            """Thread-safe callback: buffer the record and schedule a flush."""
            with _log_lock:
                _log_buffer.append((formatted, record))
                _prev_log["formatted"] = formatted
                _prev_log["record"] = record

                if not _log_flush_scheduled["value"]:
                    _log_flush_scheduled["value"] = True
                    threading.Timer(0.25, _flush_log).start()

        # Backfill: records captured between handler creation and callback
        # registration would otherwise be lost from the UI.
        records = gui_handler.records
        if records:
            for fmt, rec in records:
                log_list.controls.append(
                    ft.Text(
                        fmt,
                        size=11,
                        font_family="Consolas",
                        color=_level_color(rec.levelno),
                        selectable=True,
                        no_wrap=True,
                    )
                )
            last_fmt, last_rec = records[-1]
            _prev_log["formatted"] = last_fmt
            _prev_log["record"] = last_rec
            latest_log_text.value = last_fmt.split("\n", 1)[0]
            latest_log_text.color = _level_color(last_rec.levelno)

        gui_handler.set_callback(_on_log_record)

        # =============================================================
        # SETTINGS OVERLAY -- Config editor with save/cancel
        # =============================================================
        def _show_settings():
            """Build and display the settings editor overlay from SETTINGS_DATA."""
            import config_manager as cm
            from config_data import SETTINGS_DATA

            fields = {}
            field_rows = []

            for title, name, description, default, type_info in SETTINGS_DATA:
                if name not in config:
                    continue
                val = config[name]
                control_type = type_info.get("type", "text")

                if control_type == "bool":
                    control = ft.Checkbox(value=bool(val))
                elif control_type == "slider":
                    if type_info.get("is_float"):
                        control = ft.TextField(
                            value=str(val), dense=True,
                            input_filter=ft.InputFilter(regex_string=r"[0-9.\-]"),
                        )
                    else:
                        control = ft.TextField(
                            value=str(val), dense=True,
                            input_filter=ft.NumbersOnlyInputFilter(),
                        )
                elif control_type == "json_list":
                    control = ft.TextField(
                        value=json.dumps(val), dense=True,
                        multiline=True, min_lines=1,
                        hint_text="JSON array",
                    )
                else:  # "text"
                    control = ft.TextField(value=str(val), dense=True)

                fields[name] = {
                    "control": control,
                    "type": control_type,
                    "is_float": type_info.get("is_float", False),
                }

                field_rows.append(ft.Container(
                    content=ft.Column(controls=[
                        ft.Text(title, size=13, weight=ft.FontWeight.W_500),
                        ft.Text(description, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
                        control,
                    ], spacing=2),
                    padding=ft.padding.only(bottom=10, left=20, right=20),
                ))

            def _close(e=None):
                """Hide the settings overlay."""
                settings_overlay.visible = False
                page.update()

            def _save(e):
                """Validate and persist all settings, then rescan the watcher if present."""
                for key, info in fields.items():
                    raw = info["control"].value
                    t = info["type"]
                    try:
                        if t == "bool":
                            config[key] = bool(raw)
                        elif t == "slider":
                            config[key] = float(raw) if info["is_float"] else int(raw)
                        elif t == "json_list":
                            config[key] = json.loads(raw)
                        else:
                            config[key] = raw
                    except (ValueError, json.JSONDecodeError):
                        if hasattr(info["control"], "error_text"):
                            info["control"].error_text = "Invalid value"
                        page.update()
                        return

                cm.save(config)
                if watcher:
                    watcher.rescan()
                _close()

            settings_card = ft.Container(
                expand=True,
                bgcolor=ft.Colors.SURFACE,
                border_radius=12,
                padding=25,
                content=ft.Column(
                    expand=True,
                    spacing=0,
                    controls=[
                        ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
                        ft.Container(height=8),
                        ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                        ft.Container(height=8),
                        ft.Column(
                            controls=field_rows,
                            scroll=ft.ScrollMode.AUTO,
                            spacing=0,
                            expand=True,
                        ),
                        ft.Container(height=8),
                        ft.Row(
                            controls=[
                                ft.TextButton("Cancel", on_click=_close),
                                ft.ElevatedButton("Save", on_click=_save),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ],
                ),
            )

            settings_overlay = ft.Container(
                expand=True,
                blur=(10, 10),
                bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                padding=40,
                content=settings_card,
                visible=True,
            )

            page.overlay.append(settings_overlay)
            page.update()

        # =============================================================
        # FINAL ASSEMBLY -- Stack layout: chat column + autocomplete popup
        # =============================================================
        page.add(
            ft.Stack(
                controls=[
                    # Main chat layout
                    ft.Column(
                        controls=[message_list, input_bar, status_panel],
                        expand=True,
                        spacing=0,
                    ),
                    # Autocomplete popup (positioned above input bar + status bar)
                    autocomplete_container,
                ],
                expand=True,
            )
        )

        input_field.focus()

        # =============================================================
        # STARTUP -- Auto-load LLM in background thread
        # =============================================================
        def auto_load():
            """Attempt to load the LLM service and create the agent on success."""
            result = ctrl.load_service("llm")
            llm = services.get("llm")
            if llm and llm.loaded:
                create_agent()
                message_list.controls.append(_system_message(
                    "LLM loaded. You can start chatting."
                ))
            else:
                message_list.controls.append(_system_message(
                    f"LLM not available: {result}\n"
                    "You can still use /commands. Use /load llm to try again."
                ))
            page.update()

        threading.Thread(target=auto_load, daemon=True).start()

    # ft.app() blocks until the Flet window is closed.
    ft.app(target=main_view)
