"""
Flet container factories for chat bubbles, command output, and tool cards.
"""

import flet as ft


def system_message(text: str) -> ft.Container:
    """A monospace text block for command output."""
    return ft.Container(
        content=ft.Text(text, font_family="Consolas", size=12, selectable=True),
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        margin=ft.margin.only(bottom=4),
        border_radius=8,
        bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
    )


def user_bubble(text: str) -> ft.Container:
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


def assistant_bubble(text: str) -> ft.Container:
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


def tool_call_card(tool_name: str, success: bool, content_control: ft.Control, initially_expanded: bool) -> ft.Container:
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
