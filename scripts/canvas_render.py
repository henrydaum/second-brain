"""Render an ordered image recipe over a transparent RGBA canvas.

Layer main(sdk, kind, input_path, output_path, width, height, seed, palette,
controls) writes PNG through sdk.fs. Background gets no input; filters return
full replacement images; objects return overlays (never the flattened input).
Helpers use box="image_editing" and relative imports. Declare external image,
font and other file inputs in layer.dependencies; masks are tracked automatically.
Only validated, complete PNGs enter the prefix cache. Force bypasses all cache.
"""
import ast
import hashlib
import json
import secrets
from io import BytesIO
from .art_kit import read_image, write_png, composite
from .canvas_catalog import prepare

box = "image_editing"
dependencies_files = ["scripts/art_kit.py", "scripts/canvas_catalog.py"]
timeout = 600


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _resolve(sdk, script):
    name = script if script.endswith(".py") else script + ".py"
    if "/" in name or "\\" in name:
        raise ValueError("layer script must be a filename, without directories")
    for root in ("workspace", "installed", "bundled"):
        path = sdk.path.join(sdk.paths.get(root), "scripts", name)
        if sdk.fs.exists(path):
            return path
    raise ValueError(f"missing layer script: {name}")


def _source_hash(sdk, path, seen=None):
    # Follow relative helper imports, including art_kit. Cycles are harmless.
    seen = set() if seen is None else seen
    if path in seen:
        return "cycle"
    seen.add(path)
    source = sdk.fs.read(path)
    helpers = {}
    parent = sdk.path.parent(path)
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.level != 1 or not node.module:
                raise ValueError("cached scripts require explicit sibling helper imports")
            helper = sdk.path.join(parent, node.module.replace(".", "/") + ".py")
            if not sdk.fs.exists(helper):
                helper = _resolve(sdk, node.module)
            helpers[helper] = _source_hash(sdk, helper, seen)
    return _digest([source, helpers])


def _valid(sdk, path, size):
    if not sdk.fs.exists(path):
        return False
    from PIL import Image
    try:
        with Image.open(BytesIO(sdk.fs.read_bytes(path))) as im:
            if im.format != "PNG" or (size is not None and im.size != size) or im.mode != "RGBA":
                return False
            im.load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def main(sdk, canvas_id, out=None, seed=None, force_new_seed=False, force=False):
    from PIL import Image
    state = sdk.services.call("canvas", "get_state", canvas_id)
    if state is None:
        raise ValueError(f"unknown canvas: {canvas_id!r}")
    size = (state["width"], state["height"])
    if any(type(v) is not int or v < 1 for v in size):
        raise ValueError("dimensions must be positive integers")
    if seed is not None and force_new_seed:
        raise ValueError("choose seed or force_new_seed, not both")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")
    seed = seed if seed is not None else (None if force_new_seed else state.get("render_seed"))
    if seed is None:
        seed = secrets.randbelow(2147483647)
    palettes = sdk.services.call("canvas", "list_palettes")
    palette = next((p for p in palettes if p["id"] == state["palette_id"]), None)
    if palette is None:
        raise ValueError("unknown palette")
    palette = dict(palette, colors={**palette["colors"], **state.get("palette_colors", {})})
    layers = state.get("layers", [])
    root = sdk.path.join(sdk.paths.get("workspace"), "canvas_renders")
    # Renderer and library changes invalidate even the empty-canvas cache.
    renderer = _resolve(sdk, "canvas_render")
    key = _digest(["rgba-recipe-v3", size, palette, seed, _source_hash(sdk, renderer)])
    paths, scripts, prepared = [], [], []
    paths.append(sdk.path.join(root, key + ".png"))
    for index, layer in enumerate(layers):
        kind = layer["kind"]
        if kind not in ("background", "filter", "object") or (kind == "background" and index != 0):
            raise ValueError("background may appear only at index zero")
        script = None
        controls = layer.get("controls", {})
        if layer.get("visible", True):
            script = _resolve(sdk, layer["script"])
            technique = prepare(layer["script"], controls, kind)
            controls = technique["controls"]
            inputs = list(layer.get("dependencies", [])) + technique["dependencies"]
            if layer.get("mask"):
                inputs.append(layer["mask"])
            files = {p: hashlib.sha256(sdk.fs.read_bytes(p)).hexdigest() for p in inputs}
            pixels = {k: v for k, v in layer.items() if k not in ("id", "name")}
            pixels["controls"] = controls
            key = _digest([key, pixels, _source_hash(sdk, script), files])
        scripts.append(script)
        prepared.append(controls)
        paths.append(sdk.path.join(root, key + ".png"))
    cached = -1
    if not force:
        for count in range(len(layers), -1, -1):
            if _valid(sdk, paths[count], size if count == 0 else None):
                cached = count
                break
    cache_hit = cached == len(layers)
    if cached < 0:
        write_png(sdk, paths[0], Image.new("RGBA", size, (0, 0, 0, 0)))
        cached = 0
    start = cached
    for idx in range(start, len(layers)):
        layer = layers[idx]
        if scripts[idx] is None:
            continue
        temp = sdk.fs.temp(suffix=".png")
        try:
            base = read_image(sdk, paths[idx])
            sdk.scripts.run(scripts[idx], kind=layer["kind"],
                            input_path=None if layer["kind"] == "background" else paths[idx],
                            output_path=temp, width=base.width, height=base.height, seed=seed,
                            palette=palette, controls=prepared[idx])
            with Image.open(BytesIO(sdk.fs.read_bytes(temp))) as image:
                if image.format != "PNG":
                    raise ValueError("layer output must be PNG")
            rendered = read_image(sdk, temp)
            mask = read_image(sdk, layer["mask"], base.size) if layer.get("mask") else None
            if layer["kind"] != "object" and rendered.size != base.size:
                if (layer.get("opacity", 1) != 1 or mask is not None or
                        layer.get("blend_mode", "normal") != "normal" or
                        tuple(layer.get("offset", (0, 0))) != (0, 0)):
                    raise ValueError("dimension-changing steps need opacity=1, no mask, normal blend and zero offset")
                result = rendered
            else:
                result = composite(base, rendered, opacity=layer.get("opacity", 1),
                                   blend_mode=layer.get("blend_mode", "normal"), mask=mask,
                                   offset=layer.get("offset", (0, 0)),
                                   replace=layer["kind"] in ("background", "filter"))
            # Encode into unique scratch first; publish only a completed image.
            write_png(sdk, temp, result)
            sdk.fs.move(temp, paths[idx + 1])
        finally:
            if sdk.fs.exists(temp):
                sdk.fs.delete(temp)
    final_path = paths[-1]
    final_size = read_image(sdk, final_path).size
    if out and out != final_path:
        sdk.fs.write_bytes(out, sdk.fs.read_bytes(final_path))
    sdk.services.call("canvas", "set_render_seed", canvas_id, seed)
    return {"path": out or final_path, "seed": seed, "pool_hash": key,
            "cache_hit": cache_hit,
            "cached_layers": start, "total_layers": len(layers),
            "width": final_size[0], "height": final_size[1]}
