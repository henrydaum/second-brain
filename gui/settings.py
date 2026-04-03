"""
Settings overlay — builds and displays the settings editor from SETTINGS_DATA.

Extracted from app.py to keep the main GUI module focused on chat/command flow.
"""

import json
import logging

import flet as ft

from gui.widgets import system_message

logger = logging.getLogger("GUI")


def _build_field(title, name, description, default, type_info, config):
    """Build a single settings field control. Returns (field_info, container) or None."""
    if name not in config:
        return None
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

    field_info = {
        "control": control,
        "type": control_type,
        "is_float": type_info.get("is_float", False),
    }

    container = ft.Container(
        content=ft.Column(controls=[
            ft.Text(title, size=13, weight=ft.FontWeight.W_500),
            ft.Text(description, size=11, color=ft.Colors.ON_SURFACE_VARIANT),
            control,
        ], spacing=2),
        padding=ft.padding.only(bottom=10, left=20, right=20),
    )
    return field_info, container


def show_settings(page, config, services, ctrl, watcher,
                  message_list, agent_ref, create_agent_fn, root_dir):
    """Build and display the settings editor overlay from SETTINGS_DATA + plugin settings."""
    import config_manager as cm
    from config_data import SETTINGS_DATA
    from plugin_discovery import get_plugin_settings

    fields = {}          # name -> field_info (for both core + plugin)
    plugin_fields = {}   # name -> field_info (plugin only, for separate save)
    field_rows = []

    # --- Core settings ---
    for title, name, description, default, type_info in SETTINGS_DATA:
        result = _build_field(title, name, description, default, type_info, config)
        if result is None:
            continue
        field_info, container = result
        fields[name] = field_info
        field_rows.append(container)

    # --- Plugin settings ---
    plugin_settings = get_plugin_settings()
    if plugin_settings:
        field_rows.append(ft.Container(
            content=ft.Column(controls=[
                ft.Container(height=4),
                ft.Divider(height=1, color=ft.Colors.OUTLINE_VARIANT),
                ft.Container(height=4),
                ft.Text("Plugin Settings", size=15, weight=ft.FontWeight.W_600),
            ], spacing=2),
            padding=ft.padding.only(bottom=10, left=20, right=20),
        ))

        for title, name, description, default, type_info in plugin_settings:
            # Ensure the value exists in config (use default if missing)
            if name not in config:
                config[name] = default
            result = _build_field(title, name, description, default, type_info, config)
            if result is None:
                continue
            field_info, container = result
            fields[name] = field_info
            plugin_fields[name] = field_info
            field_rows.append(container)

    def _close(e=None):
        """Hide the settings overlay."""
        settings_overlay.visible = False
        page.update()

    def _save(e):
        """Validate, persist, and live-reload affected subsystems."""
        # -- Change detection: snapshot values that need special handling --
        ORCH_KEYS = {"max_workers", "poll_interval"}
        all_plugin_keys = {entry[1] for entry in plugin_settings}
        old = {k: config.get(k) for k in all_plugin_keys | ORCH_KEYS | {"db_path"}}

        # -- Validate and write new values into the shared config dict --
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

        # -- Save plugin config separately (merge with existing to preserve unshown keys) --
        if plugin_fields:
            existing = cm.load_plugin_config()
            existing.update({k: config[k] for k in plugin_fields if k in config})
            cm.save_plugin_config(existing)

        # -- Detect what changed --
        changed = {k for k, v in old.items() if config.get(k) != v}
        feedback = []

        # -- Group A: Service settings -> targeted rebuild --
        svc_feedback = ctrl.reload_services_for_settings(changed, root_dir)
        feedback.extend(svc_feedback)

        # Recreate agent if LLM was reloaded
        if any("'llm'" in f for f in svc_feedback):
            llm = services.get("llm")
            if llm and llm.loaded:
                create_agent_fn()
            else:
                agent_ref["agent"] = None

        # -- Group B: Orchestrator settings --
        if "poll_interval" in changed:
            ctrl.orchestrator.poll_interval = config["poll_interval"]
            feedback.append(f"Poll interval -> {config['poll_interval']}s")

        if "max_workers" in changed:
            from concurrent.futures import ThreadPoolExecutor
            old_executor = ctrl.orchestrator.executor
            ctrl.orchestrator.max_workers = config["max_workers"]
            ctrl.orchestrator.executor = ThreadPoolExecutor(
                max_workers=config["max_workers"],
                thread_name_prefix="Worker",
            )
            old_executor.shutdown(wait=False)
            feedback.append(f"Worker pool -> {config['max_workers']} threads")

        # -- Group C: db_path (restart required) --
        if "db_path" in changed:
            feedback.append("Database path changed — restart required.")

        # -- Watcher rescan (handles sync dirs, ignored extensions, etc.) --
        if watcher:
            watcher.rescan()

        _close()

        # -- Show feedback to user --
        if feedback:
            message_list.controls.append(
                system_message(
                    "Settings saved.\n" + "\n".join(f"  • {f}" for f in feedback)
                )
            )
            page.update()

    # Allow Enter to submit from single-line TextFields
    for info in fields.values():
        c = info["control"]
        if isinstance(c, ft.TextField) and not c.multiline:
            c.on_submit = _save

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
