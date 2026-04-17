"""
Flet GUI for Second Brain.

A unified chat-first interface. Plain text goes to the LLM agent;
slash-prefixed commands (e.g. /services, /load llm) control the system.
An autocomplete popup appears when typing /.

The GUI talks directly to the Agent, ToolRegistry, and Controller
**in-process** via ``route_input()``.  Flet must own the main thread,
and its rendering model relies on synchronous ``page.update()`` calls
from callbacks that fire during tool execution.

Organisation
------------
Formatters, message widgets, the log handler, and the settings overlay
have been extracted into frontend/shared/formatters.py,
frontend/gui/widgets.py, frontend/gui/log_handler.py, and
frontend/gui/settings.py respectively.

Nearly all GUI state lives inside the ``run_gui`` → ``main_view``
closure so that inner functions can freely read and mutate shared
widgets without passing dozens of arguments.
"""

import json
import logging
import os
import threading
from pathlib import Path

import flet as ft

from Stage_3.agent import Agent
from event_bus import bus
from event_channels import APPROVAL_REQUESTED, APPROVAL_RESOLVED
from Stage_3.system_prompt import build_system_prompt
from frontend.shared.commands import CommandEntry, CommandRegistry, register_core_commands
from frontend.shared.dispatch import route_input
from frontend.shared.formatters import format_tool_result
from frontend.gui.history import build_history_drawer
from frontend.gui.log_handler import GuiLogHandler
from frontend.gui.renderers import render_paths
from frontend.gui.widgets import system_message, user_bubble, assistant_message, tool_call_card
from paths import DATA_DIR, open_file

logger = logging.getLogger("GUI")



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
        page.title = "Second Brain"
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
            page.window.destroy()
            # Trigger the full shutdown sequence (saves config, unloads
            # services, etc.).  shutdown() is idempotent — safe if already
            # called from another thread.
            shutdown_fn()

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
        conversation_ref = {"id": None}  # current conversation DB id

        # Keyboard navigation state
        _ac_index = {"value": -1}           # highlighted autocomplete item (-1 = none)
        _input_history = []                  # submitted inputs, newest last
        _history_cursor = {"value": -1}      # -1 = not browsing; 0..N = position
        _history_stash = {"value": ""}       # saves draft when entering history mode

        message_list = ft.ListView(
            expand=True,
            spacing=4,
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            auto_scroll=True,
        )

        # -----------------------------------------------------------------
        # CONVERSATION PERSISTENCE
        # -----------------------------------------------------------------
        def _on_agent_message(msg: dict):
            """Callback from Agent — save each message to the DB."""
            role = msg.get("role", "")
            content = msg.get("content") or ""
            tool_call_id = msg.get("tool_call_id")
            tool_name = msg.get("name")  # present on tool result messages

            # Lazy creation: first message in a new chat creates the DB row
            if conversation_ref["id"] is None:
                title = content[:80].replace("\n", " ").strip() if role == "user" else "New conversation"
                conversation_ref["id"] = ctrl.db.create_conversation(title)

            conv_id = conversation_ref["id"]

            # For assistant messages with tool_calls, serialize the tool_calls
            if msg.get("tool_calls"):
                content = json.dumps({
                    "content": content,
                    "tool_calls": msg["tool_calls"],
                })

            ctrl.db.save_message(conv_id, role, content,
                                 tool_call_id=tool_call_id, tool_name=tool_name)

            if role == "assistant" and not msg.get("tool_calls"):
                ctrl.maybe_generate_conversation_title_async(conv_id)

        def _start_new_conversation():
            """Reset the UI for a fresh conversation. DB row is created lazily
            on the first message so that empty conversations never pile up."""
            conversation_ref["id"] = None
            if agent_ref["agent"]:
                agent_ref["agent"].reset()
            message_list.controls.clear()
            message_list.controls.append(system_message(
                "Second Brain\n"
                "Type a message to chat, or / for commands."
            ))

        def _load_conversation(conversation_id):
            """Load a past conversation into the chat view."""
            messages = ctrl.db.get_conversation_messages(conversation_id)

            conversation_ref["id"] = conversation_id
            message_list.controls.clear()

            if not messages:
                # Empty conversation — just switch to it with a fresh view
                if agent_ref["agent"]:
                    agent_ref["agent"].reset()
                message_list.controls.append(system_message(
                    "Second Brain\n"
                    "Type a message to chat, or / for commands."
                ))
                page.update()
                return

            message_list.controls.append(system_message(
                "Second Brain\n"
                "(Loaded from history)"
            ))

            # Rebuild the visual message list and agent history
            agent_history = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"] or ""

                if role == "user":
                    message_list.controls.append(user_bubble(content))
                    agent_history.append({"role": "user", "content": content})

                elif role == "assistant":
                    # Check if content is a serialized tool_calls message
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "tool_calls" in parsed:
                            agent_history.append({
                                "role": "assistant",
                                "content": parsed.get("content"),
                                "tool_calls": parsed["tool_calls"],
                            })
                            continue  # tool call assistants are shown via tool_call_card
                    except (json.JSONDecodeError, TypeError):
                        pass
                    message_list.controls.append(assistant_message(content))
                    agent_history.append({"role": "assistant", "content": content})

                elif role == "tool":
                    agent_history.append({
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id"),
                        "content": content,
                    })
                    # Show a minimal tool card for history
                    tool_name = msg.get("tool_name") or "tool"
                    message_list.controls.append(
                        tool_call_card(tool_name, True,
                                       system_message(content[:200] + ("..." if len(content) > 200 else "")),
                                       initially_expanded=False)
                    )

            # Restore agent conversation state
            if agent_ref["agent"]:
                agent_ref["agent"].history = agent_history

            page.update()

        # Start the first conversation
        _start_new_conversation()

        # -----------------------------------------------------------------
        # AGENT LIFECYCLE
        # -----------------------------------------------------------------
        def create_agent():
            """Build or rebuild the Agent from the currently loaded LLM."""
            llm = services.get("llm")
            if llm and llm.loaded:
                agent_ref["agent"] = Agent(
                    llm, tool_registry, config,
                    system_prompt=lambda: build_system_prompt(
                        ctrl.db, ctrl.orchestrator, ctrl.tool_registry, ctrl.services
                    ),
                    on_tool_result=on_tool_result,
                    on_message=_on_agent_message,
                )

        # -----------------------------------------------------------------
        # TOOL RESULT CALLBACK (called from agent thread)
        # -----------------------------------------------------------------
        def on_tool_result(tool_name: str, result):
            """Insert tool results into the message list.

            render_files → inline preview cards (no tool wrapper).
            All other tools → collapsed expansion tile.
            """
            if tool_name == "render_files" and result.gui_display_paths:
                # Inline: preview cards go directly into the message flow
                control = render_paths(result.gui_display_paths, page, config)
                message_list.controls.append(control)
            else:
                # Wrapped in a collapsed tool card
                if result.gui_display_paths:
                    content = render_paths(result.gui_display_paths, page, config)
                elif result.llm_summary:
                    content = system_message(result.llm_summary)
                else:
                    content = system_message(format_tool_result(result))
                card = tool_call_card(tool_name, result.success, content, initially_expanded=False)
                message_list.controls.append(card)
            try:
                page.update()
            except Exception as e:
                logger.error(f"page.update() failed in on_tool_result: {e}")
                if message_list.controls:
                    message_list.controls.pop()

        # -----------------------------------------------------------------
        # COMMAND APPROVAL DIALOG (bus subscriber — APPROVAL_REQUESTED)
        # -----------------------------------------------------------------
        
        # Track active dialogs by req.id so they can be closed remotely
        _active_overlays = {}

        def _on_approval_requested(req: 'ApprovalRequest'):
            """Show a confirmation dialog. Non-blocking: the click handlers
            call req.resolve() which signals the producer thread."""
            if req.is_resolved:
                return

            def _cleanup():
                if req.id in _active_overlays:
                    ov = _active_overlays.pop(req.id)
                    ov.visible = False
                    if ov in page.overlay:
                        page.overlay.remove(ov)
                    page.update()

            def _allow(e):
                req.resolve(True)
                _cleanup()

            def _deny(e):
                req.resolve(False)
                _cleanup()

            overlay = ft.Container(
                expand=True,
                blur=(10, 10),
                bgcolor=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                padding=40,
                content=ft.Container(
                    bgcolor=ft.Colors.SURFACE,
                    border_radius=12,
                    padding=25,
                    width=500,
                    content=ft.Column([
                        ft.Text("Agent Requests Approval", size=18, weight=ft.FontWeight.BOLD),
                        ft.Container(height=8),
                        ft.Container(
                            bgcolor=ft.Colors.SURFACE,
                            border_radius=8,
                            padding=12,
                            content=ft.Text(req.command, font_family="Consolas", size=13, selectable=True),
                        ),
                        ft.Container(height=8),
                        ft.Container(
                            bgcolor=ft.Colors.SURFACE,
                            border_radius=8,
                            padding=12,
                            content=ft.Text(req.reason, font_family="Consolas", size=12,
                                            selectable=True, max_lines=20,
                                            overflow=ft.TextOverflow.ELLIPSIS),
                        ),
                        ft.Container(height=8),
                        ft.Text(
                            "Only approve actions you understand.",
                            size=11, color=ft.Colors.ERROR, italic=True,
                        ),
                        ft.Container(height=12),
                        ft.Row([
                            ft.TextButton("Deny", on_click=_deny),
                            ft.ElevatedButton("Allow", on_click=_allow),
                        ], alignment=ft.MainAxisAlignment.END),
                    ], tight=True),
                ),
                alignment=ft.alignment.center,
            )
            _active_overlays[req.id] = overlay
            page.overlay.append(overlay)
            page.update()
            
        def _on_approval_resolved(req: 'ApprovalRequest'):
            if req.id in _active_overlays:
                ov = _active_overlays.pop(req.id)
                ov.visible = False
                if ov in page.overlay:
                    page.overlay.remove(ov)
                page.update()

        bus.subscribe(APPROVAL_REQUESTED, _on_approval_requested)
        bus.subscribe(APPROVAL_RESOLVED, _on_approval_resolved)



        # =============================================================
        # COMMAND REGISTRY -- Shared core + GUI-specific overrides
        # =============================================================
        registry = CommandRegistry()
        register_core_commands(registry, ctrl, services, tool_registry, root_dir,
                               get_agent=lambda: agent_ref["agent"],
                               set_conversation_id=lambda cid: conversation_ref.__setitem__("id", cid))

        # --- GUI-specific command handlers (override or extend core) ---

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

        def _new_handler(_arg):
            """Handle /new: start a fresh conversation."""
            _start_new_conversation()
            page.update()
            return "(new conversation started)"

        def _history_handler(_arg):
            """Handle /history: open the conversation history drawer."""
            refresh_history()
            page.drawer = history_drawer
            history_drawer.open = True
            page.update()
            return None  # no text output needed


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
                if tool_name == "render_files" and result.gui_display_paths:
                    message_list.controls.append(
                        render_paths(result.gui_display_paths, page, config)
                    )
                else:
                    if result.gui_display_paths:
                        content = render_paths(result.gui_display_paths, page, config)
                    elif result.llm_summary:
                        content = system_message(result.llm_summary)
                    else:
                        content = system_message(format_tool_result(result))
                    message_list.controls.append(
                        tool_call_card(tool_name, result.success, content, initially_expanded=False)
                    )
                page.update()

            # Allow Enter to submit from single-line TextFields
            for info in fields.values():
                c = info["control"]
                if isinstance(c, ft.TextField) and not c.multiline:
                    c.on_submit = _execute

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

        def _open_folder(path):
            """Open a folder in the system file manager, creating it if needed."""
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            open_file(str(p))
            return f"Opened {p}"

        # --- GUI-specific command registration (overrides + additions) ---

        _service_names = lambda: list(services.keys())
        _tool_names = lambda: list(tool_registry.tools.keys())

        for entry in [
            # Override /load and /unload to manage agent lifecycle
            CommandEntry("load",      "Load a service",         "<service>",
                         handler=_load_handler, arg_completions=_service_names),
            CommandEntry("unload",    "Unload a service",       "<service>",
                         handler=_unload_handler, arg_completions=_service_names),
            # GUI-only commands
            CommandEntry("new",       "Start a new conversation",    handler=_new_handler),
            CommandEntry("call",      "Call a tool directly",   "<tool>",
                         handler=_call_handler, arg_completions=_tool_names),
            CommandEntry("quit",      "Shutdown",               handler=_quit_handler),
            CommandEntry("exit",      "Shutdown",               handler=_quit_handler),
            CommandEntry("config",    "Open settings panel",    handler=lambda _: _show_settings()),
            CommandEntry("settings",  "Open settings panel",    handler=lambda _: _show_settings()),
            CommandEntry("history",   "Open conversation history",   handler=_history_handler),
            CommandEntry("open_data", "Open the data folder in Explorer",
                         handler=lambda _: _open_folder(DATA_DIR)),
            CommandEntry("open_root", "Open the project root in Explorer",
                         handler=lambda _: _open_folder(root_dir)),
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
            _ac_index["value"] = -1  # reset highlight on every content change

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
                for i, cmd in enumerate(matches):
                    hint = f" {cmd.arg_hint}" if cmd.arg_hint else ""
                    tile = ft.Container(
                        key=str(i),
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
                for i, name in enumerate(candidates):
                    tile = ft.Container(
                        key=str(i),
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

        def _highlight_ac_item(index: int):
            """Apply visual highlight to the tile at *index*, clear all others."""
            for i, tile in enumerate(autocomplete_list.controls):
                tile.bgcolor = (
                    ft.Colors.with_opacity(0.12, ft.Colors.PRIMARY)
                    if i == index
                    else None
                )
            # Scroll so the highlighted tile is centered in the visible area.
            # Skip scrolling near the bottom to avoid a jarring jump.
            count = len(autocomplete_list.controls)
            tile_h = autocomplete_list.controls[index].height
            total_h = count * tile_h
            visible_h = autocomplete_overlay.height or 272
            max_offset = total_h - visible_h
            item_center = (index * tile_h) + (tile_h / 2)
            offset = item_center - visible_h / 2
            if offset < 0 or offset >= max_offset:
                page.update()
                return  # near top or bottom — don't scroll
            autocomplete_list.scroll_to(offset=offset, duration=0)
            page.update()

        def _select_ac_current() -> bool:
            """Select the currently highlighted autocomplete item.

            Returns True if a selection was made.
            """
            idx = _ac_index["value"]
            controls = autocomplete_list.controls
            if idx < 0 or idx >= len(controls):
                return False

            text = input_field.value or ""
            body = text[1:]  # strip leading /
            has_space = " " in body

            if not has_space:
                # Phase 1 — command names
                matches = registry.get_completions(body)
                if idx < len(matches):
                    _select_command(matches[idx].name)
                    return True
            else:
                # Phase 2 — argument values
                parts = body.split(maxsplit=1)
                cmd_name = parts[0].lower()
                partial_arg = parts[1] if len(parts) > 1 else ""
                entry = registry._commands.get(cmd_name)
                if entry and entry.arg_completions:
                    candidates = entry.arg_completions()
                    if partial_arg:
                        candidates = [c for c in candidates
                                      if c.lower().startswith(partial_arg.lower())]
                    if idx < len(candidates):
                        _select_arg(cmd_name, candidates[idx])
                        return True
            return False

        def _on_input_change(e):
            """Called on every keystroke in the input field."""
            _history_cursor["value"] = -1  # exit history mode on manual edit
            _update_autocomplete(input_field.value or "")
            try:
                page.update()
            except AssertionError:
                pass  # Flet race: rapid keystrokes can leave new controls without UIDs

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

        _spinner = ft.ProgressRing(width=20, height=20, stroke_width=2)
        _stop_icon = ft.Icon(ft.Icons.STOP, size=20)

        def _on_cancel_hover(e):
            cancel_button.content = _stop_icon if e.data == "true" else _spinner
            cancel_button.tooltip = "Cancel" if e.data == "true" else None
            page.update()

        def _on_cancel_click(e):
            agent = agent_ref.get("agent")
            if agent:
                agent.cancelled = True
                message_list.controls.append(system_message("Cancelling request..."))
                page.update()

        cancel_button = ft.IconButton(
            content=_spinner,
            on_click=_on_cancel_click,
            on_hover=_on_cancel_hover,
            icon_size=20,
            visible=False,
        )

        # -----------------------------------------------------------------
        # CONVERSATION HISTORY DRAWER
        # -----------------------------------------------------------------
        def _on_select_conversation(conversation_id):
            _load_conversation(conversation_id)

        def _on_new_chat():
            _start_new_conversation()
            page.update()

        def _on_delete_conversation(conversation_id):
            ctrl.db.delete_conversation(conversation_id)
            # If the deleted conversation is the current one, clear the view
            if conversation_ref["id"] == conversation_id:
                _start_new_conversation()
                page.update()
            # If it was an unsaved conversation (id is None), nothing to do

        history_drawer, refresh_history = build_history_drawer(
            page, ctrl.db,
            on_select_conversation=_on_select_conversation,
            on_new_chat=_on_new_chat,
            on_delete_conversation=_on_delete_conversation,
        )

        def _open_history(e):
            refresh_history()
            page.drawer = history_drawer
            history_drawer.open = True
            page.update()

        history_button = ft.IconButton(
            icon=ft.Icons.HISTORY,
            on_click=_open_history,
            icon_size=20,
            tooltip="Conversation history",
        )

        input_row = ft.Row(
            controls=[history_button, input_field, send_button, cancel_button],
            spacing=8,
        )

        # =============================================================
        # CHAT HANDLING -- Sends user text to agent in a background thread
        # =============================================================
        def send_chat(user_text: str):
            """Run route_input() in a background thread, re-enable input on completion."""
            processing["value"] = True
            input_field.disabled = True
            send_button.visible = False
            cancel_button.content = _spinner  # reset to spinner
            cancel_button.visible = True
            page.update()

            def run():
                try:
                    result = route_input(user_text, registry, agent_ref["agent"])
                    if result.text:
                        message_list.controls.append(assistant_message(result.text))
                except Exception as e:
                    logger.error(f"Agent error: {e}")
                    message_list.controls.append(system_message(f"Error: {e}"))
                finally:
                    processing["value"] = False
                    input_field.disabled = False
                    cancel_button.visible = False
                    send_button.visible = True
                    agent_ref["agent"].cancelled = False
                    input_field.focus()
                    page.update()

            threading.Thread(target=run, daemon=True).start()

        # =============================================================
        # INPUT HANDLER -- Routes /commands to registry, plain text to agent
        # =============================================================
        def handle_input(e):
            """Unified input handler: routes /commands to registry, plain text to agent."""
            # Guard: if autocomplete is visible with a highlighted item,
            # Enter selects the item instead of submitting input.
            if autocomplete_overlay.visible and _ac_index["value"] >= 0:
                _select_ac_current()
                _ac_index["value"] = -1
                return

            text = (input_field.value or "").strip()
            if not text:
                return

            # Record input history and reset navigation state
            _input_history.append(text)
            _history_cursor["value"] = -1
            _history_stash["value"] = ""

            input_field.value = ""
            autocomplete_overlay.visible = False

            if text.startswith("/"):
                # --- Slash command (synchronous) ---
                message_list.controls.append(system_message(f"> {text}"))
                result = route_input(text, registry, agent_ref["agent"])
                if result.text:
                    message_list.controls.append(system_message(result.text))
                page.update()
            else:
                # --- Chat message to LLM (threaded) ---
                if not agent_ref["agent"]:
                    message_list.controls.append(system_message(
                        "LLM is not loaded. Use /load llm to load it, "
                        "or /services to check status."
                    ))
                    page.update()
                    return

                message_list.controls.append(user_bubble(text))
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
        _log_at_bottom = {"value": True}

        def _on_log_scroll(e: ft.OnScrollEvent):
            _log_at_bottom["value"] = e.pixels >= e.max_scroll_extent - 20

        log_list = ft.ListView(
            spacing=3,
            auto_scroll=False,
            width=9999,  # Forces fill of available horizontal space (Flet quirk)
            on_scroll=_on_log_scroll,
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
                if _log_at_bottom["value"]:
                    log_list.scroll_to(offset=-1, duration=0)
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
            """Build and display the settings editor overlay."""
            from frontend.gui.settings import show_settings
            show_settings(
                page, config, services, ctrl, watcher,
                message_list, agent_ref, create_agent, root_dir,
            )

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

        # =============================================================
        # KEYBOARD EVENT HANDLER -- Arrow keys & Escape for autocomplete
        # navigation and input history recall
        # =============================================================
        def _on_key(e: ft.KeyboardEvent):
            key = e.key

            # --- Autocomplete navigation (takes priority) ---
            if autocomplete_overlay.visible:
                count = len(autocomplete_list.controls)
                if count == 0:
                    return

                if key == "Arrow Down":
                    _ac_index["value"] = (_ac_index["value"] + 1) % count
                    _highlight_ac_item(_ac_index["value"])
                elif key == "Arrow Up":
                    _ac_index["value"] = (_ac_index["value"] - 1) % count
                    _highlight_ac_item(_ac_index["value"])
                elif key == "Escape":
                    autocomplete_overlay.visible = False
                    _ac_index["value"] = -1
                    page.update()
                elif key == "Tab":
                    if _ac_index["value"] < 0:
                        _ac_index["value"] = 0
                        _highlight_ac_item(0)
                    _select_ac_current()
                    _ac_index["value"] = -1
                    page.update()
                # Enter is handled by the guard in handle_input
                return

            # --- Escape closes the topmost visible overlay ---
            if key == "Escape":
                for overlay in reversed(page.overlay):
                    if hasattr(overlay, "visible") and overlay.visible:
                        overlay.visible = False
                        page.update()
                        return

            # --- Input history navigation ---
            # Only activate when input has no newlines (single logical line)
            current = input_field.value or ""
            if "\n" in current:
                return  # let arrow keys do normal cursor movement

            if not _input_history:
                return

            if key == "Arrow Up":
                if _history_cursor["value"] == -1:
                    # Entering history mode: stash current input
                    _history_stash["value"] = current
                    _history_cursor["value"] = len(_input_history) - 1
                elif _history_cursor["value"] > 0:
                    _history_cursor["value"] -= 1

                input_field.value = _input_history[_history_cursor["value"]]
                page.update()

            elif key == "Arrow Down":
                if _history_cursor["value"] == -1:
                    return  # not in history mode
                elif _history_cursor["value"] < len(_input_history) - 1:
                    _history_cursor["value"] += 1
                    input_field.value = _input_history[_history_cursor["value"]]
                else:
                    # Past the end: restore stashed input
                    _history_cursor["value"] = -1
                    input_field.value = _history_stash["value"]
                page.update()

        page.on_keyboard_event = _on_key

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
                message_list.controls.append(system_message(
                    "LLM loaded. You can start chatting."
                ))
            else:
                message_list.controls.append(system_message(
                    f"LLM not available: {result}\n"
                    "You can still use /commands. Use /load llm to try again."
                ))
            page.update()

        threading.Thread(target=auto_load, daemon=True).start()

    # ft.app() blocks until the Flet window is closed.
    ft.app(target=main_view)
