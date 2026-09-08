"""Inspect and mutate image recipes. All successful edits are undoable."""
from guest.bases import BaseTool


class ManageLayers(BaseTool):
    name = "manage_layers"
    requires_services = ["canvas"]
    dependencies_files = ["services/service_canvas.py", "scripts/canvas_catalog.py"]
    description = (
        "Inspect or edit an image recipe without rendering or attaching images. create makes and selects an empty transparent canvas; "
        "inspect lists canvas state, list lists canvases, select switches canvas_id. "
        "cached resolves a pool_hash and seed to a PNG and saved recipe; remix opens that recipe "
        "as a new selected canvas. Cached pixels can be reused as image inputs or mask paths. "
        "delete removes only one layer; move reorders (optional background stays first). "
        "update changes a layer's properties (script, controls, name, visible, opacity 0..1, "
        "blend_mode, mask path, offset [x,y], dependencies file paths). Masks work on all techniques; "
        "set properties={'mask':'<same-size PNG>','opacity':0.55}. These are layer properties, "
        "not controls. Filters only affect earlier layers. duplicate copies a layer. "
        "set_control changes one control. set_dimensions accepts any positive integer width/height "
        "and replays the recipe at that size. Use technique_resize to resample pixels. "
        "controls shows current values and their specifications; set_controls patches several "
        "controls in one undo step. layer_id can replace chain_index for stable targeting. "
        "palettes lists presets; set_palette accepts colors (role-to-hex overrides). "
        "clear, undo and redo are supported."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pool_hash": {"type": "string", "description": "Cached recipe hash returned by render_canvas."},
            "seed": {"type": "integer", "description": "Seed returned by render_canvas, for cached/remix."},
            "canvas_id": {
                "type": "string",
                "description": "Canvas to edit. Omit for the session canvas.",
            },
            "action": {
                "type": "string",
                "enum": ["cached", "remix",
                    "create", "inspect", "list", "select", "update", "duplicate", "controls", "set_controls", "palettes",
                    "delete", "move", "set_control", "set_palette",
                    "set_dimensions", "clear", "undo", "redo",
                ],
            },
            "properties": {"type": "object", "description": "Layer fields for update; controls replaces the control dictionary."},
            "controls": {"type": "object", "description": "Control patch for set_controls; unmentioned values stay unchanged."},
            "layer_id": {"type": "string", "description": "Stable target instead of chain_index (or from_index for move)."},
            "colors": {"type": "object", "description": "Palette role-to-hex overrides. Replaces overrides; {} restores preset colors."},
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
        if action in ("cached", "remix"):
            try:
                return sdk.services.call("canvas", "cached_render" if action == "cached" else "remix",
                                         kwargs.get("pool_hash"), kwargs.get("seed"))
            except ValueError as exc:
                return sdk.fail(str(exc))
        if action == "palettes":
            return sdk.services.call("canvas", "list_palettes")
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
        if kwargs.get("layer_id"):
            indices = [i for i, layer in enumerate(state["layers"]) if layer["id"] == kwargs["layer_id"]]
            if not indices:
                return sdk.fail("Unknown layer_id; inspect the canvas again.")
            kwargs["from_index" if action == "move" else "chain_index"] = indices[0]

        try:
            if action == "inspect":
                return {k: v for k, v in state.items() if k not in ("undo_stack", "redo_stack")}
            if action == "update":
                properties = dict(kwargs.get("properties", {}))
                if "controls" in properties or "script" in properties:
                    layer = self._layer(state, kwargs.get("chain_index"))
                    prepared = self._prepare(sdk, properties.get("script", layer["script"]),
                                             properties.get("controls", layer["controls"]), layer["kind"])
                    properties["controls"] = prepared["controls"]
                return self._view(sdk.services.call("canvas", "update_layer", cid,
                                         kwargs.get("chain_index"), **properties))
            if action == "controls":
                layer = self._layer(state, kwargs.get("chain_index"))
                spec = sdk.scripts.run("canvas_catalog.py", script=layer["script"])
                return {"layer_id": layer["id"], "current": layer["controls"], "technique": spec}
            if action == "set_controls":
                layer = self._layer(state, kwargs.get("chain_index"))
                patch = kwargs.get("controls")
                if not isinstance(patch, dict):
                    return sdk.fail("set_controls requires a controls object")
                prepared = self._prepare(sdk, layer["script"], {**layer["controls"], **patch}, layer["kind"])
                return self._view(sdk.services.call("canvas", "update_layer", cid, kwargs.get("chain_index"),
                                                   controls=prepared["controls"]))
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
                idx = kwargs.get("chain_index")
                layer = self._layer(state, idx)
                name = str(kwargs.get("name") or "")
                prepared = self._prepare(sdk, layer["script"], {**layer["controls"], name: kwargs.get("value")}, layer["kind"])
                state = sdk.services.call(
                    "canvas", "update_layer", cid, idx, controls=prepared["controls"],
                )
                return sdk.ok(state, llm_summary=f"Set {name}={kwargs.get('value')!r} on layer {idx}.")

            if action == "set_palette":
                pid = kwargs.get("palette_id") or state["palette_id"]
                state = sdk.services.call("canvas", "set_palette", cid, pid, colors=kwargs.get("colors"))
                return self._view(state)

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

    @staticmethod
    def _layer(state, index):
        if type(index) is not int or not 0 <= index < len(state["layers"]):
            raise ValueError("valid chain_index or layer_id is required")
        return state["layers"][index]

    @staticmethod
    def _prepare(sdk, script, controls, kind):
        prepared = sdk.scripts.run("canvas_catalog.py", action="prepare", script=script,
                                   controls=controls, kind=kind)
        for path in prepared["dependencies"]:
            if not sdk.fs.exists(path):
                raise ValueError(f"Input file does not exist: {path}")
        return prepared
