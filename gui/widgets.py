"""
Flet container factories for chat bubbles, command output, tool cards,
and the unified preview card shell.
"""

import os
from pathlib import Path

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


def assistant_message(text: str) -> ft.Container:
    """Left-aligned assistant message, no bubble — flush on page background."""
    content = ft.Markdown(
        value=text,
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
    ) if any(m in text[:200] for m in ("# ", "**", "- ", "```", "| ")) else \
        ft.Text(text, size=13, selectable=True)
    return ft.Container(
        content=content,
        padding=ft.padding.only(left=14, right=80, top=6, bottom=6),
        margin=ft.margin.only(bottom=4),
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


# ===================================================================
# UNIFIED PREVIEW CARD
# ===================================================================

def preview_card(
    filename: str,
    file_path: str,
    content: ft.Control,
    page: ft.Page,
    nav_strip: ft.Control | None = None,
) -> ft.Container:
    """
    Unified preview card for all rendered file modalities.

    Layout:
      ┌───────────────────────────────┐
      │ filename.ext            [ ⋮ ] │  title strip
      │ /path/to/file                 │  path (small, muted)
      ├───────────────────────────────┤
      │   [modality content]          │  content area
      ├───────────────────────────────┤
      │     ◀  1 of N  ▶             │  nav strip (optional)
      └───────────────────────────────┘
    """
    title_row = ft.Row(
        controls=[
            ft.Text(filename, size=14, weight=ft.FontWeight.BOLD, expand=True),
            ft.PopupMenuButton(
                icon=ft.Icons.MORE_VERT,
                icon_size=16,
                tooltip=filename,
                items=[
                    ft.PopupMenuItem(
                        text="Open File",
                        icon=ft.Icons.OPEN_IN_NEW_ROUNDED,
                        on_click=lambda _, p=file_path: os.startfile(p),
                    ),
                    ft.PopupMenuItem(
                        text="Copy Path",
                        icon=ft.Icons.COPY_ROUNDED,
                        on_click=lambda _, p=file_path: page.set_clipboard(p),
                    ),
                ],
            ),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

    path_text = ft.Text(
        file_path, size=10, color=ft.Colors.ON_SURFACE_VARIANT,
        max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
    )

    controls = [title_row, path_text, content]
    if nav_strip:
        controls.append(nav_strip)

    return ft.Container(
        content=ft.Column(controls=controls, spacing=4),
        padding=12,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        margin=ft.margin.only(bottom=8),
    )


def build_nav_strip(
    current_ref: dict,
    total: int,
    items: list,
    on_navigate,
    page: ft.Page,
) -> ft.Container:
    """
    Bottom navigation strip: ◀ 1 of N ▶. Used by all modalities.

    current_ref: mutable dict like {"idx": 0} to track position
    items: list of (ft.Control, filename, filepath) tuples for the stack
    on_navigate: callback(new_index) to update visibility and title
    """
    counter = ft.Text(f"1 of {total}", size=12, text_align=ft.TextAlign.CENTER)

    def _go(delta, _e):
        new = current_ref["idx"] + delta
        if 0 <= new < total:
            current_ref["idx"] = new
            counter.value = f"{new + 1} of {total}"
            left_btn.disabled = (new == 0)
            right_btn.disabled = (new >= total - 1)
            on_navigate(new)
            page.update()

    left_btn = ft.IconButton(
        ft.Icons.CHEVRON_LEFT, on_click=lambda e: _go(-1, e),
        icon_size=20, disabled=True,
    )
    right_btn = ft.IconButton(
        ft.Icons.CHEVRON_RIGHT, on_click=lambda e: _go(1, e),
        icon_size=20, disabled=(total <= 1),
    )

    return ft.Container(
        content=ft.Row(
            controls=[left_btn, counter, right_btn],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4,
        ),
        border=ft.border.only(top=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT)),
        padding=ft.padding.only(top=4),
    )
