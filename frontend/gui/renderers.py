"""
Modality renderers for the GUI.

Takes a list of file paths, parses each one, groups by modality,
and returns Flet controls wrapped in the unified preview card shell.

Design philosophy: rendered files are **previews**, not viewers.
Truncate aggressively, no scrolling inside cards. The "Open File"
button is the real action.
"""

import base64
import io
import logging
import os

from paths import open_file
from collections import defaultdict
from pathlib import Path

import flet as ft

from Stage_1.registry import parse, get_modality
from frontend.gui.widgets import preview_card, build_nav_strip, _truncate_path

logger = logging.getLogger("Renderers")


# ===================================================================
# CONSTANTS
# ===================================================================

_PREVIEW_MAX_TEXT_CHARS = 500    # ~15 lines of text
_PREVIEW_MAX_TEXT_LINES = 15
_PREVIEW_TABLE_ROWS = 5
_PREVIEW_TABLE_COLS = 6
_PREVIEW_TABLE_COL_WIDTH = 30
_PREVIEW_IMAGE_HEIGHT = 300

_CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp", ".java", ".cs", ".php", ".rb",
    ".go", ".rs", ".swift", ".kt", ".sql", ".sh", ".bat", ".ps1",
    ".r", ".m", ".scala", ".lua", ".json", ".yaml", ".yml", ".xml",
    ".ini", ".toml", ".cfg", ".env", ".log",
}
_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".rst", ".tex", ".rtf"}


# ===================================================================
# CONTENT RENDERERS
#
# Each returns just the content area control (the middle of the card).
# The preview_card() shell is applied by render_paths().
# ===================================================================

def _render_text_content(path: str, output) -> ft.Control:
    """Render truncated text preview — no scrolling."""
    text = str(output) if output else "(empty)"

    # Truncate by lines first, then by chars
    lines = text.split("\n")
    if len(lines) > _PREVIEW_MAX_TEXT_LINES:
        text = "\n".join(lines[:_PREVIEW_MAX_TEXT_LINES])
        remaining = sum(len(l) for l in lines[_PREVIEW_MAX_TEXT_LINES:])
        truncated = True
    elif len(text) > _PREVIEW_MAX_TEXT_CHARS:
        text = text[:_PREVIEW_MAX_TEXT_CHARS]
        remaining = len(str(output)) - _PREVIEW_MAX_TEXT_CHARS
        truncated = True
    else:
        remaining = 0
        truncated = False

    ext = Path(path).suffix.lower()

    if ext in _CODE_EXTENSIONS:
        content = ft.Text(
            value=text, selectable=True, font_family="Consolas", size=12,
        )
    elif ext in _MARKDOWN_EXTENSIONS:
        content = ft.Markdown(
            value=text, selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )
    else:
        md_markers = ("# ", "## ", "**", "- ", "```", "| ")
        if any(marker in text[:500] for marker in md_markers):
            content = ft.Markdown(
                value=text, selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            )
        else:
            content = ft.Text(
                value=text, selectable=True, font_family="Consolas", size=12,
            )

    controls = [content]
    if truncated:
        controls.append(ft.Text(
            f"truncated — {remaining:,} more characters",
            size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT,
        ))

    return ft.Column(controls=controls, spacing=4)


def _render_image_content(path: str, output) -> ft.Control:
    """Render image fitted to card width, no distortion."""
    images = output if isinstance(output, list) else [output]
    img = images[0]  # first image for single-item; multi handled by nav

    thumb = img.copy()
    thumb.thumbnail((400, _PREVIEW_IMAGE_HEIGHT))

    buf = io.BytesIO()
    fmt = "PNG" if thumb.mode == "RGBA" else "JPEG"
    thumb.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return ft.Image(
        src_base64=b64,
        width=400,
        height=_PREVIEW_IMAGE_HEIGHT,
        fit=ft.ImageFit.CONTAIN,
        border_radius=4,
    )


def _render_audio_content(path: str, output, page: ft.Page) -> ft.Control:
    """Render a play button for audio."""
    audio = ft.Audio(src=path, autoplay=False)
    page.overlay.append(audio)

    is_playing = {"value": False}

    def toggle_play(_):
        if is_playing["value"]:
            audio.pause()
        else:
            audio.resume()
        is_playing["value"] = not is_playing["value"]
        play_btn.icon = ft.Icons.PAUSE_CIRCLE if is_playing["value"] else ft.Icons.PLAY_CIRCLE
        page.update()

    play_btn = ft.IconButton(
        icon=ft.Icons.PLAY_CIRCLE, on_click=toggle_play, icon_size=32,
    )

    return ft.Row(
        controls=[play_btn, ft.Text(Path(path).name, size=12)],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _render_video_content(path: str, output) -> ft.Control:
    """Render video player fitted to card."""
    return ft.Video(
        playlist=[ft.VideoMedia(path)],
        width=480,
        height=270,
        autoplay=False,
        show_controls=True,
    )


def _df_to_markdown(df, max_rows: int = _PREVIEW_TABLE_ROWS,
                    max_col_width: int = _PREVIEW_TABLE_COL_WIDTH,
                    max_cols: int = _PREVIEW_TABLE_COLS) -> str:
    """Convert a DataFrame to a GitHub-Flavored Markdown table string (preview)."""
    total_cols = len(df.columns)
    total_rows = len(df)

    if total_cols > max_cols:
        display_df = df.iloc[:max_rows, :max_cols]
    else:
        display_df = df.head(max_rows)

    if display_df.columns.empty:
        return "*(empty table — no columns)*"

    headers = [str(h).replace("|", "\\|")[:max_col_width] for h in display_df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for _, row in display_df.iterrows():
        cells = [str(v).replace("|", "\\|")[:max_col_width] for v in row]
        lines.append("| " + " | ".join(cells) + " |")

    footer_parts = []
    if total_rows > max_rows:
        footer_parts.append(f"{max_rows} of {total_rows} rows")
    if total_cols > max_cols:
        footer_parts.append(f"{max_cols} of {total_cols} columns")
    if footer_parts:
        lines.append(f"\n*Showing {', '.join(footer_parts)}*")

    return "\n".join(lines)


def _render_tabular_content(path: str, output) -> ft.Control:
    """Render first sheet as a markdown table preview. No sheet navigation."""
    tables = output if isinstance(output, dict) else {"default": output}
    first_name = next(iter(tables))
    first_df = tables[first_name]

    md = _df_to_markdown(first_df)
    controls = [
        ft.Markdown(
            md,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            selectable=True,
        )
    ]

    if len(tables) > 1:
        controls.append(ft.Text(
            f"Sheet \"{first_name}\" — {len(tables)} sheets total, open file to view all",
            size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT,
        ))

    return ft.Column(controls=controls, spacing=4)


def _render_container_content(path: str, output) -> ft.Control:
    """Render a folder as an info row."""
    return ft.Row(
        controls=[
            ft.Icon(ft.Icons.FOLDER_OPEN, size=16, color=ft.Colors.PRIMARY),
            ft.Text(str(path), size=12),
        ],
        spacing=8,
    )


# ===================================================================
# RENDERER DISPATCH
# ===================================================================

# Maps modality → renderer function.
# Audio renderer needs page for overlay; others don't.
_RENDERERS = {
    "text": _render_text_content,
    "image": _render_image_content,
    "audio": _render_audio_content,  # special: needs page arg
    "video": _render_video_content,
    "tabular": _render_tabular_content,
    "container": _render_container_content,
}


def _call_renderer(modality: str, path: str, output, page: ft.Page) -> ft.Control:
    """Call the appropriate renderer, handling the audio special case."""
    renderer = _RENDERERS[modality]
    if modality == "audio":
        return renderer(path, output, page)
    return renderer(path, output)


def _render_error_content(error) -> ft.Control:
    """Render an error message as card content."""
    return ft.Text(f"Error: {error}", color=ft.Colors.ERROR, size=12, selectable=True)


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

def render_paths(paths: list[str], page: ft.Page, config: dict = None,
                 services: dict = None) -> ft.Control:
    """
    Main entry point. Takes file paths, parses each, groups by modality,
    and returns preview cards wrapped in the unified shell.
    """
    if not paths:
        return ft.Text("(no files)", italic=True, size=12)

    # Phase 1: Parse all files, group by modality
    grouped: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for path in paths:
        modality = get_modality(Path(path).suffix)
        result = parse(path, config=config, services=services)
        grouped[modality].append((path, result))

    # Phase 2: Render per modality group into preview cards
    cards: list[ft.Control] = []

    for modality, items in grouped.items():
        if len(items) == 1:
            # Single file — simple card, no nav strip
            path, result = items[0]
            filename = Path(path).name

            if not result.success:
                content = _render_error_content(result.error)
            elif modality not in _RENDERERS:
                content = _render_error_content(f"No renderer for: {modality}")
            else:
                try:
                    content = _call_renderer(modality, path, result.output, page)
                except Exception as e:
                    logger.error(f"Renderer failed for {path} ({modality}): {e}")
                    content = _render_error_content(e)

            cards.append(preview_card(filename, path, content, page))

        else:
            # Multiple files — card with nav strip
            rendered_items = []  # (content, filename, filepath)
            for path, result in items:
                filename = Path(path).name
                if not result.success:
                    c = _render_error_content(result.error)
                elif modality not in _RENDERERS:
                    c = _render_error_content(f"No renderer for: {modality}")
                else:
                    try:
                        c = _call_renderer(modality, path, result.output, page)
                    except Exception as e:
                        logger.error(f"Renderer failed for {path} ({modality}): {e}")
                        c = _render_error_content(e)
                rendered_items.append((c, filename, path))

            # Build stack with visibility toggling
            for i, (c, _, _) in enumerate(rendered_items):
                c.visible = (i == 0)

            stack = ft.Stack(controls=[c for c, _, _ in rendered_items])
            current_ref = {"idx": 0}

            # Title/path refs that update on nav
            title_text = ft.Text(
                Path(rendered_items[0][1]).stem, size=14,
                weight=ft.FontWeight.BOLD,
            )
            path_text = ft.Text(
                _truncate_path(rendered_items[0][2]), size=10, italic=True,
                color=ft.Colors.ON_SURFACE_VARIANT,
                max_lines=1, overflow=ft.TextOverflow.ELLIPSIS,
            )

            def _on_navigate(new_idx, _items=rendered_items, _stack=stack,
                             _title=title_text, _path=path_text):
                for i, (c, _, _) in enumerate(_items):
                    c.visible = (i == new_idx)
                _title.value = Path(_items[new_idx][1]).stem
                _path.value = _truncate_path(_items[new_idx][2])

            nav = build_nav_strip(current_ref, len(rendered_items),
                                  rendered_items, _on_navigate, page)

            # Build the card manually (since title/path update dynamically)
            menu_btn = ft.PopupMenuButton(
                icon=ft.Icons.MORE_VERT,
                icon_size=16,
                tooltip=rendered_items[0][1],
                items=[
                    ft.PopupMenuItem(
                        text="Open File",
                        icon=ft.Icons.OPEN_IN_NEW_ROUNDED,
                        on_click=lambda _, ref=current_ref, ri=rendered_items: open_file(ri[ref["idx"]][2]),
                    ),
                    ft.PopupMenuItem(
                        text="Open File Location",
                        icon=ft.Icons.FOLDER_OPEN_ROUNDED,
                        on_click=lambda _, ref=current_ref, ri=rendered_items: open_file(str(Path(ri[ref["idx"]][2]).parent)),
                    ),
                    ft.PopupMenuItem(
                        text="Copy Path",
                        icon=ft.Icons.COPY_ROUNDED,
                        on_click=lambda _, ref=current_ref, ri=rendered_items: page.set_clipboard(ri[ref["idx"]][2]),
                    ),
                ],
            )

            title_col = ft.Column(
                controls=[title_text, path_text],
                spacing=0,
                expand=True,
            )

            header_row = ft.Row(
                controls=[title_col, menu_btn],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

            card = ft.Container(
                content=ft.Column(
                    controls=[header_row, stack, nav],
                    spacing=4,
                ),
                padding=12,
                border_radius=8,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                margin=ft.margin.only(bottom=8),
            )
            cards.append(card)

    if len(cards) == 1:
        return cards[0]
    return ft.Column(controls=cards, spacing=4)
