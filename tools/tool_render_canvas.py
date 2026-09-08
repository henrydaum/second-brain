"""Render the canvas layer chain to a PNG and show it.

Calls ``canvas_render.py``, which walks the layers in order — background
first, then each filter/object — and writes the final PNG to
``workspace/canvas_renders/``. Returns the image as an attachment so the user
can see it.

The renderer uses a pool-hash cache: if the layer chain, dimensions, palette,
and seed are unchanged from a previous render, the cached PNG is returned
instantly. Changing any control, adding a layer, or passing
``force_new_seed=True`` busts the cache.

Call this after adding or editing layers. The tool automatically resolves the
session canvas, so you can omit ``canvas_id`` unless working with multiple
canvases.
"""

from guest.bases import BaseTool


class RenderCanvas(BaseTool):
    name = "render_canvas"
    requires_services = ["canvas"]
    dependencies_files = ["services/service_canvas.py", "scripts/canvas_render.py", "scripts/art_kit.py"]
    dependencies_pip = ["Pillow>=10.1", "numpy"]
    description = (
        "Render the canvas layer chain to a PNG image. Call this after "
        "adding or editing layers. Returns the rendered image so the user "
        "can see the result. Pass force_new_seed=True to get a fresh "
        "random seed for generative layers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "canvas_id": {
                "type": "string",
                "description": "Canvas to render. Omit to use the session's current canvas.",
            },
            "seed": {"type": "integer", "description": "Explicit reproducible seed (including zero)."},
            "out": {"type": "string", "description": "Optional PNG export path."},
            "force": {"type": "boolean", "description": "Recompute all steps, bypassing cached outputs."},
            "force_new_seed": {
                "type": "boolean",
                "description": "Mint a fresh random seed for generative layers. Default false.",
            },
            "narration": {
                "type": "string",
                "description": "A few words on what you are rendering, shown to the user.",
            },
        },
    }

    def run(self, sdk, canvas_id=None, force_new_seed=False, seed=None, out=None, force=False):
        # Resolve the canvas.
        if canvas_id:
            state = sdk.services.call("canvas", "get_state", canvas_id)
        else:
            state = sdk.services.call("canvas", "for_session")
        if state is None:
            return sdk.fail(
                "No canvas. Create one with "
                "sdk.services.call('canvas', 'get_or_create') or add a layer."
            )
        canvas_id = state["canvas_id"]

        layers = state.get("layers") or []
        # Render.
        result = sdk.scripts.run(
            "canvas_render.py",
            canvas_id=canvas_id,
            force_new_seed=bool(force_new_seed), seed=seed, out=out, force=force,
        )
        if not result.get("path"):
            return sdk.fail("Render produced no output path.")

        path = result["path"]
        attachment_path = result["attachment_path"]
        seed = result.get("seed", "?")
        cache_hit = result.get("cache_hit", False)
        cached = result.get("cached_layers", 0)
        total = result.get("total_layers", len(layers))
        hit_note = " (cache hit)" if cache_hit else ""
        summary = (
            f"Rendered {total} layer(s) "
            f"({result['width']}×{result['height']}, seed={seed}). "
            f"{cached}/{total} cached{hit_note}. "
            f"Output: {path}. Attached: {attachment_path}. SHA-256: {result['image_sha256']}"
        )

        return sdk.ok(
            {"path": path, "attachment_path": attachment_path, "image_sha256": result["image_sha256"],
             "seed": seed, "canvas_id": canvas_id,
             "pool_hash": result["pool_hash"], "cache_hit": cache_hit, "cached_layers": cached,
             "width": result["width"], "height": result["height"]},
            llm_summary=summary,
            attachments=[attachment_path],
        )
