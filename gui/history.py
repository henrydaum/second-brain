"""
Conversation history drawer — slide-out panel listing past conversations.

Extracted from app.py to keep the main GUI module focused on chat flow.
"""

import time
from datetime import datetime

import flet as ft


def _format_time(timestamp):
    """Format a Unix timestamp into a human-readable relative or absolute string."""
    if not timestamp:
        return ""
    dt = datetime.fromtimestamp(timestamp)
    now = datetime.now()
    diff = now - dt
    if diff.days == 0:
        return dt.strftime("%I:%M %p")
    elif diff.days == 1:
        return "Yesterday"
    elif diff.days < 7:
        return dt.strftime("%A")  # day name
    else:
        return dt.strftime("%b %d")


def _truncate(text, max_len=40):
    """Truncate text for display as a conversation title."""
    if not text:
        return "New conversation"
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def build_history_drawer(page, db, on_select_conversation, on_new_chat, on_delete_conversation):
    """
    Build a NavigationDrawer containing the conversation history list.

    Args:
        page:                    Flet page instance.
        db:                      Database instance with conversation CRUD methods.
        on_select_conversation:  Callback(conversation_id: int) when a conversation is clicked.
        on_new_chat:             Callback() when "New Chat" is clicked.
        on_delete_conversation:  Callback(conversation_id: int) when delete is clicked.

    Returns:
        (drawer, refresh_fn) — the NavigationDrawer control and a function to
        refresh its contents (call before opening).
    """
    conversation_list = ft.ListView(
        spacing=2,
        padding=ft.padding.symmetric(horizontal=8, vertical=4),
        expand=True,
    )

    def refresh():
        """Reload the conversation list from the database."""
        conversations = db.list_conversations(limit=100)
        conversation_list.controls.clear()

        if not conversations:
            conversation_list.controls.append(
                ft.Container(
                    content=ft.Text(
                        "No conversations yet.",
                        size=12, color=ft.Colors.ON_SURFACE_VARIANT,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=ft.padding.all(24),
                    alignment=ft.alignment.center,
                )
            )
        else:
            for conv in conversations:
                conv_id = conv["id"]
                title = _truncate(conv["title"])
                time_str = _format_time(conv.get("updated_at"))

                tile = ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Column(
                                controls=[
                                    ft.Text(title, size=13, weight=ft.FontWeight.W_500,
                                            max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                                    ft.Text(time_str, size=11,
                                            color=ft.Colors.ON_SURFACE_VARIANT),
                                ],
                                spacing=2,
                                expand=True,
                            ),
                            ft.IconButton(
                                icon=ft.Icons.DELETE_OUTLINE,
                                icon_size=16,
                                icon_color=ft.Colors.ON_SURFACE_VARIANT,
                                tooltip="Delete",
                                on_click=lambda e, cid=conv_id: _handle_delete(cid),
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    border_radius=8,
                    ink=True,
                    on_click=lambda e, cid=conv_id: _handle_select(cid),
                    on_hover=lambda e: _on_hover(e),
                )
                conversation_list.controls.append(tile)

        page.update()

    def _handle_select(conversation_id):
        drawer.open = False
        page.update()
        on_select_conversation(conversation_id)

    def _handle_delete(conversation_id):
        on_delete_conversation(conversation_id)
        refresh()

    def _on_hover(e):
        e.control.bgcolor = (
            ft.Colors.with_opacity(0.08, ft.Colors.ON_SURFACE)
            if e.data == "true" else None
        )
        e.control.update()

    def _handle_new_chat(e):
        drawer.open = False
        page.update()
        on_new_chat()

    header = ft.Container(
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.Icon(ft.Icons.HISTORY, size=20),
                        ft.Text("Conversations", size=16, weight=ft.FontWeight.BOLD),
                    ],
                    spacing=8,
                ),
                ft.Container(height=4),
                ft.FilledTonalButton(
                    text="New Chat",
                    icon=ft.Icons.ADD,
                    on_click=_handle_new_chat,
                    expand=True,
                ),
            ],
            spacing=4,
        ),
        padding=ft.padding.all(16),
    )

    drawer = ft.NavigationDrawer(
        controls=[
            header,
            ft.Divider(height=1),
            conversation_list,
        ],
        on_dismiss=lambda e: None,
    )

    return drawer, refresh
