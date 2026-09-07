"""Inspect and mutate image recipes. All successful edits are undoable."""
from guest.bases import BaseTool


class ManageLayers(BaseTool):
    name = "manage_layers"
    requires_services = ["canvas"]
    dependencies_files = ["services/service_canvas.py"]
    description = (
        "Inspect or edit an image recipe. create makes and selects an empty transparent canvas; "
        "inspect lists canvas state, list lists canvases, select switches canvas_id. "
        "delete removes only one layer; move reorders (optional background stays first). "
        "update changes a layer's properties (script, controls, name, visible, opacity 0..1, "
        "blend_mode, mask path, offset [x,y], dependencies file paths). duplicate copies a layer. "
        "set_control changes one control. set_dimensions accepts any positive integer width/height "
        "and replays the recipe at that size. set_palette, clear, undo and redo are supported."
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
                    "create", "inspect", "list", "select", "update", "duplicate",
                    "delete", "move", "set_control", "set_palette",
                    "set_dimensions", "clear", "undo", "redo",
                ],
            },
            "properties": {"type": "object", "description": "Layer fields for update; controls replaces the control dictionary."},
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
        if action == "list":
            return sdk.services.call("canvas", "list_canvases")
        if action == "create":
            cid = sdk.services.call("canvas", "create", width=kwargs.get("width", 1024),
                                    height=kwargs.get("height", 1024),
                                    palette_id=kwargs.get("palette_id", "default"))
            return self._view(sdk.services.call("canvas", "for_session", cid))
        if action == "select":
            if not canvas_id:
                return sdk.fail("select requires canvas_id")
            return self._view(sdk.services.call("canvas", "for_session", canvas_id))
        if canvas_id:
            state = sdk.services.call("canvas", "get_state", canvas_id)
        else:
            state = sdk.services.call("canvas", "for_session")
        if state is None:
            return sdk.fail("No canvas. Add a layer first.")
        cid = state["canvas_id"]

        try:
            if action == "inspect":
                return {k: v for k, v in state.items() if k not in ("undo_stack", "redo_stack")}
            if action == "update":
                return self._view(sdk.services.call("canvas", "update_layer", cid,
                                         kwargs.get("chain_index"), **kwargs.get("properties", {})))
            if action == "duplicate":
                return self._view(sdk.services.call("canvas", "duplicate_layer", cid, kwargs.get("chain_index")))
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
                w = kwargs.get("width", state["width"])
                h = kwargs.get("height", state["height"])
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
    @staticmethod
    def _view(state):
        return {k: v for k, v in state.items() if k not in ("undo_stack", "redo_stack")}
