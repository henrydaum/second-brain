"""Save/share the current generated fractal."""

from pathlib import Path

from config import config_manager
from paths import DATA_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.fractal_gallery import GALLERY_DIR, current, save_share


class SaveFractalArt(BaseTool):
    name = "save_fractal_art"
    description = "Share the currently displayed original fractal art. Ask the user for an optional title and name first; skipped values become untitled and anonymous. This tool always opens a yes/no permission dialog before saving."
    parameters = {"type": "object", "properties": {
        "title": {"type": "string", "description": "Optional title for the shared fractal."},
        "artist": {"type": "string", "description": "Optional creator name; use anonymous if skipped."},
    }}
    requires_services = []
    max_calls = 1

    def run(self, context, **kwargs) -> ToolResult:
        item = current(getattr(context, "session_key", None))
        src = Path(item["path"]).resolve() if item else None
        origin = (DATA_DIR / "fractals" / "mandelbrot").resolve()
        if not item or not item.get("original") or origin not in src.parents:
            return ToolResult.failed("Only a currently displayed original fractal generated in this session can be shared.")
        title, artist = kwargs.get("title") or "untitled", kwargs.get("artist") or "anonymous"
        if not getattr(context, "approve_command", None) or not context.approve_command("Share fractal art?", f'Save "{title}" by "{artist}" to the public shared fractal gallery?'):
            return ToolResult.failed("The fractal was not shared.")
        dest, meta = save_share(item["path"], title, artist)
        _sync(context, dest)
        return ToolResult(data=meta, llm_summary=f'Shared "{meta["title"]}" by {meta["artist"]}: {dest}', attachment_paths=[str(dest)])


def _sync(context, path: Path):
    dirs = [str(p) for p in (context.config.get("sync_directories") or [])]
    if str(GALLERY_DIR) not in dirs:
        context.config["sync_directories"] = dirs + [str(GALLERY_DIR)]
        config_manager.save(context.config)
    if getattr(context, "db", None):
        from plugins.services.helpers.parser_registry import get_modality
        context.db.upsert_file(str(path), path.name, path.suffix.lower(), get_modality(path.suffix.lower()), path.stat().st_mtime)
        if getattr(context, "orchestrator", None):
            context.orchestrator.on_file_discovered(str(path), path.suffix.lower(), get_modality(path.suffix.lower()))
