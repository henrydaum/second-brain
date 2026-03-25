"""
Modality renderers for the GUI.

Takes a list of file paths, parses each one, groups by modality,
and returns Flet controls that display the content.

Layout strategies:
  - Carousel (dot indicators + arrows) for image, audio, video
  - ExpansionTiles (collapsed by default) for text, tabular
  - Simple row for container
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
# CONSTANTS
# ===================================================================

_CAROUSEL_MODALITIES = {"image", "audio", "video"}
_CAROUSEL_HEIGHTS = {
    "image": 340,   # 300 image + label + nav
    "audio": 60,    # player row
    "video": 310,   # 270 video + label + nav
}
_MAX_EXPAND_HEIGHT = 350

_CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".htm", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp", ".java", ".cs", ".php", ".rb",
    ".go", ".rs", ".swift", ".kt", ".sql", ".sh", ".bat", ".ps1",
    ".r", ".m", ".scala", ".lua", ".json", ".yaml", ".yml", ".xml",
    ".ini", ".toml", ".cfg", ".env", ".log",
}
_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".rst", ".tex", ".rtf"}


# ===================================================================
# CAROUSEL
# ===================================================================

class Carousel:
    """
    Reusable carousel widget with left/right arrows and dot indicators.

    Shows one slide at a time. Dot indicators for <= 7 items,
    numeric counter for > 7 items.
    """

    def __init__(self, slides: list[ft.Control], page: ft.Page, height: int | None = None):
        self._slides = slides
        self._page = page
        self._height = height
        self._current = 0

    def build(self) -> ft.Control:
        if len(self._slides) <= 1:
            return self._slides[0] if self._slides else ft.Text("(empty)", italic=True, size=11)

        # Wrap each slide in a centered container, only active one visible
        for i, slide in enumerate(self._slides):
            slide.visible = (i == 0)

        self._stack = ft.Stack(
            controls=self._slides,
            height=self._height,
        )

        self._left_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT,
            on_click=self._go_left,
            icon_size=24,
            disabled=True,
        )
        self._right_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT,
            on_click=self._go_right,
            icon_size=24,
            disabled=(len(self._slides) <= 1),
        )
        self._indicator = self._build_indicator()

        nav_row = ft.Row(
            controls=[self._left_btn, self._indicator, self._right_btn],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4,
        )

        return ft.Column(controls=[self._stack, nav_row], spacing=4)

    def _build_indicator(self) -> ft.Text:
        total = len(self._slides)
        return ft.Text(
            f"{self._current + 1} / {total}",
            size=12,
            text_align=ft.TextAlign.CENTER,
        )

    def _update_view(self):
        for i, slide in enumerate(self._slides):
            slide.visible = (i == self._current)

        total = len(self._slides)
        self._indicator.value = f"{self._current + 1} / {total}"

        self._left_btn.disabled = (self._current == 0)
        self._right_btn.disabled = (self._current >= total - 1)
        self._page.update()

    def _go_left(self, _):
        if self._current > 0:
            self._current -= 1
            self._update_view()

    def _go_right(self, _):
        if self._current < len(self._slides) - 1:
            self._current += 1
            self._update_view()


# ===================================================================
# HELPERS
# ===================================================================

def _open_file_button(path: str) -> ft.IconButton:
    """Small button that opens the file in the OS default application."""
    return ft.IconButton(
        icon=ft.Icons.OPEN_IN_NEW,
        tooltip=f"Open {Path(path).name}",
        on_click=lambda _: os.startfile(path),
        icon_size=16,
    )


def _expansion_title_row(path: str) -> ft.Row:
    """Filename + open button, used as the title of an ExpansionTile."""
    return ft.Row(
        controls=[
            ft.Text(Path(path).name, size=12, weight=ft.FontWeight.BOLD, expand=True),
            _open_file_button(path),
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _modality_header(modality: str) -> ft.Container:
    """Centered '{MODALITY} RESULTS:' label."""
    return ft.Container(
        content=ft.Text(
            f"{modality.upper()} RESULTS:",
            size=13,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.PRIMARY,
            text_align=ft.TextAlign.CENTER,
        ),
        alignment=ft.alignment.center,
        padding=ft.padding.only(top=4, bottom=4),
    )


def _error_tile(path: str, error) -> ft.ExpansionTile:
    """Error displayed as an ExpansionTile matching text/tabular style."""
    return ft.ExpansionTile(
        title=_expansion_title_row(path),
        initially_expanded=False,
        tile_padding=ft.padding.symmetric(horizontal=8, vertical=0),
        controls=[
            ft.ListTile(
                subtitle=ft.Container(
                    content=ft.Text(
                        f"Error: {error}",
                        color=ft.Colors.ERROR,
                        selectable=True,
                        size=12,
                    ),
                    bgcolor=ft.Colors.SURFACE,
                    border=ft.border.all(0.5, ft.Colors.ERROR),
                    padding=5,
                    border_radius=7,
                ),
            ),
        ],
        dense=True,
    )


def _error_slide(path: str, error) -> ft.Container:
    """Error displayed as a carousel slide for media modalities."""
    return ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(Path(path).name, size=11, italic=True),
                ft.Text(f"Error: {error}", color=ft.Colors.ERROR, size=12),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4,
        ),
        alignment=ft.alignment.center,
    )


# ===================================================================
# INDIVIDUAL MODALITY RENDERERS
#
# Carousel modalities (image, audio, video) return list[ft.Control]
# (one slide per item). The dispatch layer aggregates these into a
# single Carousel.
#
# Expansion modalities (text, tabular) return ft.ExpansionTile.
# Container returns a simple ft.Row.
# ===================================================================

def render_text(path: str, output, page: ft.Page) -> ft.Control:
    """Render a text string inside a collapsed ExpansionTile."""
    text = str(output) if output else "(empty)"

    max_chars = 5000
    truncated = len(text) > max_chars
    display_text = text[:max_chars] + "\n\n... (truncated)" if truncated else text

    ext = Path(path).suffix.lower()

    if ext in _CODE_EXTENSIONS:
        content = ft.Text(
            value=display_text,
            selectable=True,
            font_family="Consolas",
            size=12,
        )
    elif ext in _MARKDOWN_EXTENSIONS:
        content = ft.Markdown(
            value=display_text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        )
    else:
        # Unknown extension, .txt, or document-extracted text: content heuristic
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

    return ft.ExpansionTile(
        title=_expansion_title_row(path),
        initially_expanded=False,
        tile_padding=ft.padding.symmetric(horizontal=8, vertical=0),
        controls=[
            ft.ListTile(
                subtitle=ft.Container(
                    content=ft.Column(
                        controls=[content],
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    bgcolor=ft.Colors.SURFACE,
                    border=ft.border.all(0.5, ft.Colors.ON_SURFACE),
                    padding=5,
                    border_radius=7,
                    height=_MAX_EXPAND_HEIGHT,
                ),
            ),
        ],
        dense=True,
    )


def render_image(path: str, output, page: ft.Page) -> list[ft.Control]:
    """Return a list of carousel slides, one per PIL image."""
    images = output if isinstance(output, list) else [output]
    slides = []

    for i, img in enumerate(images):
        try:
            thumb = img.copy()
            thumb.thumbnail((400, 300))

            buf = io.BytesIO()
            fmt = "PNG" if thumb.mode == "RGBA" else "JPEG"
            thumb.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode()

            slides.append(
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text(Path(path).name, size=11, italic=True),
                            ft.Image(
                                src_base64=b64,
                                width=400,
                                height=300,
                                fit=ft.ImageFit.CONTAIN,
                                border_radius=4,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    ),
                    alignment=ft.alignment.center,
                )
            )
        except Exception as e:
            logger.debug(f"Failed to render image {i} from {path}: {e}")
            slides.append(
                ft.Container(
                    content=ft.Text(f"({Path(path).name} image {i} failed: {e})", italic=True, size=11),
                    alignment=ft.alignment.center,
                )
            )

    return slides


def render_audio(path: str, output, page: ft.Page) -> list[ft.Control]:
    """Return a list with one carousel slide containing an audio player."""
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

    slide = ft.Container(
        content=ft.Row(
            controls=[play_btn, ft.Text(Path(path).name, size=12)],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        alignment=ft.alignment.center,
    )

    return [slide]


def render_video(path: str, output, page: ft.Page) -> list[ft.Control]:
    """Return a list with one carousel slide containing a video player."""
    video = ft.Video(
        playlist=[ft.VideoMedia(path)],
        width=480,
        height=270,
        autoplay=False,
        show_controls=True,
    )

    slide = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text(Path(path).name, size=11, italic=True),
                video,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=4,
        ),
        alignment=ft.alignment.center,
    )

    return [slide]


def _build_sheet_table(df, sheet_name: str, show_sheet_label: bool = True) -> ft.Control:
    """Build a single sheet's DataTable with horizontal + vertical scroll."""
    max_rows = 50
    display_df = df.head(max_rows)
    truncated = len(df) > max_rows

    columns = [
        ft.DataColumn(ft.Text(str(col), size=11, weight=ft.FontWeight.BOLD))
        for col in display_df.columns
    ]
    if not columns:
        return ft.Text("(empty table — no columns)", size=11, italic=True)
    rows = []
    for _, row in display_df.iterrows():
        cells = [ft.DataCell(ft.Text(str(val)[:80], size=11)) for val in row]
        rows.append(ft.DataRow(cells=cells))

    table = ft.DataTable(
        columns=columns,
        rows=rows,
        column_spacing=16,
        data_row_max_height=32,
        horizontal_lines=ft.BorderSide(1, ft.Colors.OUTLINE_VARIANT),
    )

    controls = []
    if show_sheet_label and sheet_name != "default":
        controls.append(ft.Text(f"Sheet: {sheet_name}", size=11, italic=True))

    # Horizontal scroll via Row, vertical scroll via Column
    controls.append(
        ft.Row(
            controls=[table],
            scroll=ft.ScrollMode.AUTO,
        )
    )

    if truncated:
        controls.append(ft.Text(f"... showing {max_rows} of {len(df)} rows", size=11, italic=True))

    return ft.Column(controls=controls, spacing=4)


def render_tabular(path: str, output, page: ft.Page) -> ft.Control:
    """Render DataFrames inside a collapsed ExpansionTile. Multi-sheet files get an inner carousel."""
    tables = output if isinstance(output, dict) else {"default": output}

    if len(tables) == 1:
        # Single sheet — render directly
        sheet_name, df = next(iter(tables.items()))
        inner = _build_sheet_table(df, sheet_name)
    else:
        # Multiple sheets — dedicated switcher with controls above table
        sheet_items = list(tables.items())
        sheet_controls = []
        for sname, df in sheet_items:
            sheet_controls.append(_build_sheet_table(df, sname, show_sheet_label=False))

        current = {"idx": 0}
        for i, ctrl in enumerate(sheet_controls):
            ctrl.visible = (i == 0)

        stack = ft.Stack(controls=sheet_controls)

        sheet_label = ft.Text(
            f"Sheet: {sheet_items[0][0]}",
            size=12, weight=ft.FontWeight.W_500, expand=True,
            text_align=ft.TextAlign.CENTER,
        )
        counter = ft.Text(
            f"1 / {len(sheet_items)}", size=12,
            text_align=ft.TextAlign.CENTER,
        )

        def _go(delta, _e):
            new = current["idx"] + delta
            if 0 <= new < len(sheet_controls):
                sheet_controls[current["idx"]].visible = False
                current["idx"] = new
                sheet_controls[new].visible = True
                sheet_label.value = f"Sheet: {sheet_items[new][0]}"
                counter.value = f"{new + 1} / {len(sheet_items)}"
                left_btn.disabled = (new == 0)
                right_btn.disabled = (new >= len(sheet_items) - 1)
                page.update()

        left_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_LEFT, on_click=lambda e: _go(-1, e),
            icon_size=20, disabled=True,
        )
        right_btn = ft.IconButton(
            icon=ft.Icons.CHEVRON_RIGHT, on_click=lambda e: _go(1, e),
            icon_size=20, disabled=(len(sheet_items) <= 1),
        )

        nav_row = ft.Container(
            content=ft.Row(
                controls=[left_btn, sheet_label, counter, right_btn],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=4,
            ),
            bgcolor=ft.Colors.SURFACE,
            border_radius=ft.border_radius.only(top_left=6, top_right=6),
            padding=ft.padding.symmetric(horizontal=8, vertical=2),
        )

        inner = ft.Column(controls=[nav_row, stack], spacing=0)

    return ft.ExpansionTile(
        title=_expansion_title_row(path),
        initially_expanded=False,
        tile_padding=ft.padding.symmetric(horizontal=8, vertical=0),
        controls=[
            ft.ListTile(
                subtitle=ft.Container(
                    content=ft.Column(
                        controls=[inner],
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    bgcolor=ft.Colors.SURFACE,
                    border=ft.border.all(0.5, ft.Colors.ON_SURFACE),
                    padding=5,
                    border_radius=7,
                    height=_MAX_EXPAND_HEIGHT,
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                ),
            ),
        ],
        dense=True,
    )


def render_container(path: str, output, page: ft.Page) -> ft.Control:
    """Render a container as a simple filepath row with open button."""
    return ft.Row(
        controls=[
            ft.Icon(ft.Icons.FOLDER_OPEN, size=16, color=ft.Colors.PRIMARY),
            ft.Text(str(path), size=12, expand=True),
            _open_file_button(path),
        ],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=8,
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


def render_paths(paths: list[str], page: ft.Page, config: dict = None, services: dict = None) -> ft.Control:
    """
    Main entry point. Takes one or more file paths, parses each,
    groups by modality, and returns a single Flet control.

    Carousel modalities (image, audio, video) are aggregated across
    files into a single carousel per modality. Expansion modalities
    (text, tabular) get individual ExpansionTiles per file.
    """
    if not paths:
        return ft.Text("(no files)", italic=True, size=12)

    # Phase 1: Parse all files, group by modality (even failures stay with their modality)
    grouped: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for path in paths:
        modality = get_modality(Path(path).suffix)
        result = parse(path, config=config, services=services)
        grouped[modality].append((path, result))

    # Phase 2: Render per modality group — each gets its own bordered box
    modality_boxes: list[ft.Control] = []

    for modality, items in grouped.items():
        group_controls: list[ft.Control] = [_modality_header(modality)]

        if modality in _CAROUSEL_MODALITIES:
            # Collect all slides across files into one carousel
            all_slides: list[ft.Control] = []
            for path, result in items:
                if not result.success:
                    all_slides.append(_error_slide(path, result.error))
                    continue
                renderer = _RENDERERS.get(modality)
                if renderer is None:
                    all_slides.append(_error_slide(path, f"No renderer for: {modality}"))
                    continue
                try:
                    slides = renderer(path, result.output, page)
                    all_slides.extend(slides)
                except Exception as e:
                    logger.error(f"Renderer failed for {path} ({modality}): {e}")
                    all_slides.append(_error_slide(path, e))

            if len(all_slides) == 1:
                group_controls.append(all_slides[0])
            elif all_slides:
                height = _CAROUSEL_HEIGHTS.get(modality)
                carousel = Carousel(all_slides, page, height=height)
                group_controls.append(carousel.build())
        else:
            # Per-file rendering (text, tabular, container)
            for path, result in items:
                if not result.success:
                    group_controls.append(_error_tile(path, result.error))
                    continue
                renderer = _RENDERERS.get(modality)
                if renderer is None:
                    group_controls.append(_error_tile(path, f"No renderer for: {modality}"))
                    continue
                try:
                    group_controls.append(renderer(path, result.output, page))
                except Exception as e:
                    logger.error(f"Renderer failed for {path} ({modality}): {e}")
                    group_controls.append(_error_tile(path, e))

        # Wrap this modality group in its own bordered box
        modality_boxes.append(
            ft.Container(
                content=ft.Column(controls=group_controls, spacing=6),
                padding=8,
                border=ft.border.all(1, ft.Colors.OUTLINE_VARIANT),
                border_radius=4,
                bgcolor=ft.Colors.SURFACE,
            )
        )

    # If only one modality, return just that box; otherwise stack them
    if len(modality_boxes) == 1:
        return modality_boxes[0]

    return ft.Column(controls=modality_boxes, spacing=8)
