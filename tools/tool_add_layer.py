"""Add a layer to a canvas.

A layer is either a background (produces an image from scratch), a filter
(transforms the prior layer's output), or an object (alpha-composites on top).
Each layer names a script from the scripts/ directory and optional controls
(keyword arguments passed to the script at render time).

The three layer kinds:
- ``background`` — produces a new image from nothing. Only layer 0 may be a
  background. Adding a second background replaces the first.
- ``filter`` — reads the prior layer's output PNG, returns a transformed PNG.
  Use for blurs, color grades, glitch effects, warp transforms.
- ``object`` — reads the prior layer's output, returns an RGBA image
  alpha-composited on top. Use for text, shapes, overlays.

The ``script`` parameter is the filename of a script in the scripts/ directory,
without the ``.py`` extension (e.g. ``"canvas_load_image"``, ``"canvas_blur"``).
The ``canvas_`` prefix is a convention for layer scripts — follow it, but the
renderer only appends ``".py"`` and calls ``sdk.scripts.run``. Controls are the
keyword arguments passed to that script's ``main()`` at render time.

A background is optional; filters and objects can start on transparency. Always call
``render_canvas`` after adding layers to see the result.
"""

from guest.bases import BaseTool


class AddLayer(BaseTool):
    name = "add_layer"
    requires_services = ["canvas"]
    dependencies_files = ["services/service_canvas.py"]
    description = (
        "Add a layer to a canvas. 'kind' is 'background' (produces a new "
        "image from scratch — only layer 0 may be a background), 'filter' "
        "(transforms the prior layer's output), or 'object' (alpha-composites "
        "on top of the prior layer). 'script' is the filename of a script in "
        "the scripts/ directory without the .py extension (e.g. "
        "'canvas_load_image', 'canvas_blur'). The 'canvas_' prefix is a "
        "convention for layer scripts. 'controls' are keyword arguments "
        "passed to that script at render time."
    )
    parameters = {
        "type": "object",
        "properties": {
            "canvas_id": {
                "type": "string",
                "description": "Canvas to add to. Omit to use the session's current canvas.",
            },
            "script": {
                "type": "string",
                "description": "Script filename without .py (e.g. 'canvas_load_image', 'canvas_blur'). The 'canvas_' prefix is a convention for layer scripts.",
            },
            "kind": {
                "type": "string",
                "enum": ["background", "filter", "object"],
                "description": "Layer kind.",
            },
            "controls": {
                "type": "object",
                "description": "Keyword arguments for the script (e.g. {'radius': 5.0}).",
            },
            "properties": {"type": "object", "description": "Layer name, visible, opacity, blend_mode, mask path, offset [x,y], dependencies file paths."},
            "narration": {
                "type": "string",
                "description": "A few words on what you are adding, shown to the user.",
            },
        },
        "required": ["script", "kind"],
    }

    def run(self, sdk, script, kind, canvas_id=None, controls=None, properties=None):
        # Resolve canvas.
        if canvas_id:
            state = sdk.services.call("canvas", "get_state", canvas_id)
        else:
            state = sdk.services.call("canvas", "for_session")
        if state is None and canvas_id:
            return sdk.fail(f"Unknown canvas: {canvas_id}")
        if state is None:
            # Auto-create.
            state = sdk.services.call("canvas", "get_or_create")
        canvas_id = state["canvas_id"]

        try:
            state = sdk.services.call(
                "canvas", "add_layer",
                canvas_id, script, kind,
                controls or {}, **(properties or {}),
            )
        except Exception as exc:
            return sdk.fail(str(exc))

        layers = state.get("layers") or []
        summary = (
            f"Added {kind} layer '{script}' at index "
            f"{0 if kind == 'background' else len(layers) - 1}. "
            f"Canvas {canvas_id} now has {len(layers)} layer(s)."
        )
        return sdk.ok(state, llm_summary=summary)