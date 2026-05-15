"""Find shared fractals visually similar to the current one."""

from array import array
from pathlib import Path

from PIL import Image

from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.fractal_gallery import GALLERY_DIR, cosine, current, image_vector, is_image, read_json


class FindSimilarImages(BaseTool):
    name = "find_similar_images"
    description = "Find shared fractal art visually similar to the currently displayed fractal, then display the closest shared match with its title and creator."
    parameters = {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Number of similar shared images to report."}}}
    requires_services = []
    max_calls = 1

    def run(self, context, **kwargs) -> ToolResult:
        item = current(getattr(context, "session_key", None))
        if not item:
            return ToolResult.failed("No current fractal image is displayed.")
        limit = max(1, min(5, int(kwargs.get("limit") or 3)))
        ranked = _embedded(context, item["path"]) or _visual(item["path"])
        ranked = [r for r in ranked if Path(r["path"]).resolve() != Path(item["path"]).resolve()][:limit]
        if not ranked:
            return ToolResult.failed("No shared fractal images are available yet.")
        lines = [f'{i+1}. "{r["title"]}" by {r["artist"]} ({r["score"]:.2f})' for i, r in enumerate(ranked)]
        return ToolResult(data={"matches": ranked}, llm_summary="Closest shared fractals:\n" + "\n".join(lines), attachment_paths=[ranked[0]["path"]])


def _visual(path):
    target = image_vector(path)
    return sorted((_row(p, cosine(target, image_vector(p))) for p in GALLERY_DIR.glob("*.png") if is_image(p)), key=lambda r: r["score"], reverse=True)


def _embedded(context, path):
    emb = context.services.get("image_embedder") if getattr(context, "services", None) else None
    if not emb or not getattr(emb, "loaded", False) or not getattr(context, "db", None):
        return []
    try:
        target = list(emb.encode([Image.open(path).convert("RGB")])[0])
        with context.db.lock:
            rows = context.db.conn.execute("SELECT path, embedding FROM image_embeddings WHERE path LIKE ?", (str(GALLERY_DIR) + "%",)).fetchall()
        out = []
        for r in rows:
            vec = array("f"); vec.frombytes(r["embedding"])
            if is_image(r["path"]):
                out.append(_row(r["path"], cosine(target, vec)))
        return sorted(out, key=lambda r: r["score"], reverse=True)
    except Exception:
        return []


def _row(path, score):
    meta = read_json(Path(path).with_suffix(".json"))
    return {"path": str(path), "title": meta.get("title") or "untitled", "artist": meta.get("artist") or "anonymous", "score": float(score)}
