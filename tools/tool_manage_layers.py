"""Edit the canvas layer chain — delete, reorder, tweak, and undo.

All mutations snapshot before changing, so undo/redo work. Layer 0 is always
the background and cannot be moved; deleting it clears the entire canvas.

Actions:
- ``delete`` — remove the layer at chain_index. Deleting layer 0 (the
  background) clears the entire canvas.
- ``move`` — reorder from from_index to to_index. Layer 0 is anchored.
- ``set_control`` — update one control parameter on one layer. E.g. change a
  blur radius from 5.0 to 12.0 without removing and re-adding the layer.
- ``set_palette`` — change the canvas-wide palette. All layers that reference
  palette colors will use the new palette on the next render.
- ``set_dimensions`` — resize the canvas (width, height, clamped 16..8192).
- ``clear`` — wipe the entire layer chain. Palette and dimensions stay.
- ``undo`` / ``redo`` — step through the last 50 mutations.

Always call ``render_canvas`` after mutations to see the result.
"""

from guest.bases import BaseTool


class ManageLayers(BaseTool):
    name = "manage_layers"
    description = (
        "Edit the canvas layer chain. action=delete removes the layer at "
        "chain_index (0 is the background — deleting it clears the canvas). "
        "action=move reorders from from_index to to_index; layer 0 must stay "
        "a background. action=set_control updates one control on one layer "
        "(chain_index, name, value). action=set_palette changes the "
        "canvas-wide palette. action=set_dimensions resizes the canvas "
        "(width, height, clamped 16..8192). action=clear wipes the layer "
        "chain. action=undo/redo step through history."
    )
    parameters = {
        "type": "object",
        "properties": {
            "canvas_id": {
                "type": "string",
                "description": "Canvas to edit. Omit for the session canvas.",
            },
            "action": {
                "type": "string",
                "enum": [
                    "delete", "move", "set_control", "set_palette",
                    "set_dimensions", "clear", "undo", "redo",
                ],
            },
            "chain_index": {
                "type": "integer",
                "description": "Target layer index for delete or set_control.",
            },
            "from_index": {"type": "integer"},
            "to_index": {"type": "integer"},
            "name": {"type": "string", "description": "Control name for set_control."},
            "value": {"description": "New control value for set_control."},
            "palette_id": {"type": "string"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "narration": {
                "type": "string",
                "description": "A few words on what you are changing, shown to the user.",
            },
        },
        "required": ["action"],
    }

    def run(self, sdk, action, canvas_id=None, **kwargs):
        if canvas_id:
            state = sdk.services.call("canvas", "get_state", canvas_id)
        else:
            state = sdk.services.call("canvas", "for_session")
        if state is None:
            return sdk.fail("No canvas. Add a layer first.")
        cid = state["canvas_id"]

        try:
            if action == "delete":
                idx = int(kwargs.get("chain_index", -1))
                state = sdk.services.call("canvas", "remove_layer", cid, idx)
                return sdk.ok(state, llm_summary=f"Deleted layer {idx}.")

            if action == "move":
                fi = int(kwargs.get("from_index", -1))
                ti = int(kwargs.get("to_index", -1))
                state = sdk.services.call("canvas", "move_layer", cid, fi, ti)
                return sdk.ok(state, llm_summary=f"Moved layer {fi} → {ti}.")

            if action == "set_control":
                idx = int(kwargs.get("chain_index", -1))
                name = str(kwargs.get("name") or "")
                state = sdk.services.call(
                    "canvas", "set_control", cid, idx, name, kwargs.get("value"),
                )
                return sdk.ok(state, llm_summary=f"Set {name}={kwargs.get('value')!r} on layer {idx}.")

            if action == "set_palette":
                pid = str(kwargs.get("palette_id") or "")
                state = sdk.services.call("canvas", "set_palette", cid, pid)
                return sdk.ok(state, llm_summary=f"Palette set to {pid}.")

            if action == "set_dimensions":
                w = int(kwargs.get("width", 1024))
                h = int(kwargs.get("height", 1024))
                state = sdk.services.call("canvas", "set_dimensions", cid, w, h)
                return sdk.ok(state, llm_summary=f"Resized to {w}×{h}.")

            if action == "clear":
                state = sdk.services.call("canvas", "clear", cid)
                return sdk.ok(state, llm_summary="Cleared the canvas.")

            if action == "undo":
                state = sdk.services.call("canvas", "undo", cid)
                return sdk.ok(state, llm_summary="Undo.")

            if action == "redo":
                state = sdk.services.call("canvas", "redo", cid)
                return sdk.ok(state, llm_summary="Redo.")

            return sdk.fail(f"Unknown action: {action!r}")

        except Exception as exc:
            return sdk.fail(str(exc))