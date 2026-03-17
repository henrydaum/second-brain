"""
Modality renderers for the GUI.

Takes a list of file paths, parses each one, groups by modality,
and returns Flet controls that display the content with an "Open File"
button for each file.
"""

import base64
import io
import logging
import os
from collections import defaultdict
from pathlib import Path

import flet as ft

from Stage_1.registry import parse, get_modality

logger = logging.getLogger("Renderers")


# ===================================================================
# INDIVIDUAL MODALITY RENDERERS
#
# Each takes (path, parse_result, page) and returns an ft.Control.
# The parse_result.output is already in the standard format for that
# modality (see ParseResult docstring).
# ===================================================================

def _open_file_button(path: str) -> ft.IconButton:
    """Small button that opens the file in the OS default application."""
    return ft.IconButton(
        icon=ft.Icons.OPEN_IN_NEW,
        tooltip=f"Open {Path(path).name}",
        on_click=lambda _: os.startfile(path),
        icon_size=16,
    )


def _file_header(path: str) -> ft.Row:
    """File name label + open button, used as a header for each rendered file."""
    return ft.Row(
        controls=[
            ft.Text(Path(path).name, size=12, weight=ft.FontWeight.BOLD, expand=True),
            _open_file_button(path),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def render_text(path: str, output, page: ft.Page) -> ft.Control:
    """Render a text string. Uses Markdown if content looks like it."""
    text = str(output) if output else "(empty)"

    # Truncate very long text for display
    max_chars = 5000
    truncated = len(text) > max_chars
    display_text = text[:max_chars] + "\n\n... (truncated)" if truncated else text

    # Simple heuristic: if it has markdown-ish markers, render as markdown
    md_markers = ("# ", "## ", "**", "- ", "```", "| ")
    looks_like_md = any(marker in display_text[:500] for marker in md_markers)

    if looks_like_md:
        content = ft.Markdown(
            value=display_text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )
    else:
        content = ft.Text(
            value=display_text,
            selectable=True,
            font_family="Consolas",
            size=12,
        )

    return ft.Column(
        controls=[
            _file_header(path),
            ft.Container(
                content=ft.Column(
                    controls=[content],
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=8,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                border_radius=4,
                height=300,
            ),
        ],
        spacing=4,
    )


def render_image(path: str, output, page: ft.Page) -> ft.Control:
    """Render a list of PIL Images as a thumbnail gallery."""
    images = output if isinstance(output, list) else [output]

    thumbnails = []
    for i, img in enumerate(images):
        try:
            # Resize for thumbnail
            thumb = img.copy()
            thumb.thumbnail((256, 256))

            buf = io.BytesIO()
            fmt = "PNG" if thumb.mode == "RGBA" else "JPEG"
            thumb.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode()

            thumbnails.append(
                ft.Image(
                    src_base64=b64,
                    width=256,
                    height=256,
                    fit=ft.ImageFit.CONTAIN,
                    border_radius=4,
                )
            )
        except Exception as e:
            logger.debug(f"Failed to render image {i} from {path}: {e}")
            thumbnails.append(ft.Text(f"(image {i} failed: {e})", italic=True, size=11))

    return ft.Column(
        controls=[
            _file_header(path),
            ft.Container(
                content=ft.Column(
                    controls=[ft.Row(controls=thumbnails, wrap=True, spacing=8)],
                    scroll=ft.ScrollMode.AUTO,
                ),
                height=280,
            ),
        ],
        spacing=4,
    )


def render_audio(path: str, output, page: ft.Page) -> ft.Control:
    """Render an audio player using the original file path."""
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
        icon=ft.Icons.PLAY_CIRCLE,
        on_click=toggle_play,
        icon_size=32,
    )

    return ft.Column(
        controls=[
            _file_header(path),
            ft.Row(
                controls=[play_btn, ft.Text(Path(path).name, size=12)],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ],
        spacing=4,
    )


def render_video(path: str, output, page: ft.Page) -> ft.Control:
    """Render a video player using the original file path."""
    video = ft.Video(
        playlist=[ft.VideoMedia(path)],
        width=480,
        height=270,
        autoplay=False,
        show_controls=True,
    )

    return ft.Column(
        controls=[
            _file_header(path),
            video,
        ],
        spacing=4,
    )


def render_tabular(path: str, output, page: ft.Page) -> ft.Control:
    """Render dict[sheet_name, DataFrame] as DataTables."""
    tables = output if isinstance(output, dict) else {"default": output}

    controls = [_file_header(path)]

    for sheet_name, df in tables.items():
        # Cap display rows
        max_rows = 50
        display_df = df.head(max_rows)
        truncated = len(df) > max_rows

        columns = [ft.DataColumn(ft.Text(str(col), size=11, weight=ft.FontWeight.BOLD)) for col in display_df.columns]
        rows = []
        for _, row in display_df.iterrows():
            cells = [ft.DataCell(ft.Text(str(val)[:80], size=11)) for val in row]
            rows.append(ft.DataRow(cells=cells))

        label = f"Sheet: {sheet_name}" if sheet_name != "default" else ""
        if label:
            controls.append(ft.Text(label, size=11, italic=True))

        controls.append(
            ft.Container(
                content=ft.Column(
                    controls=[ft.DataTable(
                        columns=columns,
                        rows=rows,
                        column_spacing=16,
                        data_row_max_height=32,
                        horizontal_lines=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT),
                    )],
                    scroll=ft.ScrollMode.AUTO,
                ),
                height=400,
            )
        )

        if truncated:
            controls.append(ft.Text(f"... showing {max_rows} of {len(df)} rows", size=11, italic=True))

    return ft.Column(controls=controls, spacing=4)


def render_container(path: str, output, page: ft.Page) -> ft.Control:
    """Render a list of child file paths as a clickable list."""
    child_paths = output if isinstance(output, list) else []

    items = []
    for child in child_paths[:100]:  # Cap display
        items.append(
            ft.ListTile(
                leading=ft.Icon(ft.Icons.INSERT_DRIVE_FILE, size=16),
                title=ft.Text(Path(child).name, size=12),
                subtitle=ft.Text(str(child), size=10),
                on_click=lambda _, p=child: os.startfile(p),
                dense=True,
            )
        )

    if len(child_paths) > 100:
        items.append(ft.Text(f"... and {len(child_paths) - 100} more files", size=11, italic=True))

    return ft.Column(
        controls=[
            _file_header(path),
            ft.ListView(controls=items, height=min(300, len(items) * 48)),
        ],
        spacing=4,
    )


# ===================================================================
# RENDERER DISPATCH
# ===================================================================

_RENDERERS = {
    "text": render_text,
    "image": render_image,
    "audio": render_audio,
    "video": render_video,
    "tabular": render_tabular,
    "container": render_container,
}


def _render_single(path: str, page: ft.Page, config: dict = None, services: dict = None) -> ft.Control:
    """Parse a single file and return the appropriate rendered widget."""
    result = parse(path, config=config, services=services)

    if not result.success:
        return ft.Column(controls=[
            _file_header(path),
            ft.Text(f"Parse error: {result.error}", color=ft.Colors.ERROR, size=12),
        ])

    renderer = _RENDERERS.get(result.modality)
    if renderer is None:
        return ft.Column(controls=[
            _file_header(path),
            ft.Text(f"No renderer for modality: {result.modality}", italic=True, size=12),
        ])

    try:
        return renderer(path, result.output, page)
    except Exception as e:
        logger.error(f"Renderer failed for {path} ({result.modality}): {e}")
        return ft.Column(controls=[
            _file_header(path),
            ft.Text(f"Render error: {e}", color=ft.Colors.ERROR, size=12),
        ])


def render_paths(paths: list[str], page: ft.Page, config: dict = None, services: dict = None) -> ft.Control:
    """
    Main entry point. Takes one or more file paths, parses each,
    groups by modality, and returns a single Flet control that
    displays them all.

    If all files share the same modality, no grouping header is shown
    (unless there are multiple files). If modalities differ, results
    are grouped under modality headers.
    """
    if not paths:
        return ft.Text("(no files)", italic=True, size=12)

    # Parse all and group by modality
    grouped = defaultdict(list)
    for path in paths:
        modality = get_modality(Path(path).suffix)
        grouped[modality].append(path)

    # Single modality, single file — render directly
    if len(grouped) == 1 and len(paths) == 1:
        return _render_single(paths[0], page, config, services)

    # Build grouped display
    sections = []
    for modality, mod_paths in grouped.items():
        if len(grouped) > 1:
            sections.append(
                ft.Text(
                    modality.upper(),
                    size=13,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.PRIMARY,
                )
            )

        for path in mod_paths:
            sections.append(_render_single(path, page, config, services))
        sections.append(ft.Divider(height=1))

    # Remove trailing divider
    if sections and isinstance(sections[-1], ft.Divider):
        sections.pop()

    return ft.Container(
        content=ft.Column(controls=sections, spacing=8),
        padding=8,
        border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
        border_radius=8,
        bgcolor=ft.Colors.SURFACE,
    )
