"""Render a canvas layer chain to a PNG file with deterministic caching.

This is the renderer that ``tool_render_canvas.py`` calls. It walks the layer
chain from the canvas service, calling each layer's script via
``sdk.scripts.run``. Each layer gets the previous layer's output PNG as input
(except backgrounds, which start from nothing).

Cache
    The pool-hash identifies a unique chain: layer scripts, kinds, controls,
    dimensions, palette, and seed. A full match returns the cached PNG
    instantly. Partial matches find the longest cached prefix and only
    re-render from that point. Cached renders live in
    ``workspace/canvas_renders/<pool_hash>/<seed>.png``.

Layer contract
    Each layer script must export a ``main(sdk, kind, input_path, output_path,
    width, height, seed, palette, controls)`` function.

    - ``kind`` — ``"background"``, ``"filter"``, or ``"object"``
    - ``input_path`` — path to the previous layer's PNG (None for backgrounds)
    - ``output_path`` — where to write the PNG result
    - ``width``, ``height`` — canvas dimensions in pixels
    - ``seed`` — deterministic random seed for reproducible noise
    - ``palette`` — the canvas palette dict (``{"id": ..., "colors": {...}}``)
    - ``controls`` — the layer's own kwargs dict

    The script must write a valid PNG to ``output_path``. Layer scripts can
    ``from art_kit import ...`` for free — the sandbox adds the scripts
    directory to its import path.

This script is called by ``tool_render_canvas.py`` via ``sdk.scripts.run``.
Do not call it directly from the agent toolset; use the ``render_canvas`` tool
instead.
"""

import hashlib
import json
import time

POOL_HASH_LEN = 16


def main(sdk, canvas_id, out=None, seed=None, force_new_seed=False):
    state = sdk.services.call("canvas", "get_state", canvas_id)
    if state is None:
        raise ValueError(f"unknown canvas: {canvas_id!r}")

    layers = state.get("layers") or []
    if not layers:
        raise ValueError("canvas has no layers — nothing to render")
    if layers[0]["kind"] != "background":
        raise ValueError("layer 0 must be a background")
    if any(layer["kind"] != "filter" for layer in layers[1:]):
        raise ValueError("only filters after the background are supported until object compositing is implemented")

    width = state["width"]
    height = state["height"]
    palette_id = state.get("palette_id", "default")

    palettes = sdk.services.call("canvas", "list_palettes")
    palette = _find_palette(palettes, palette_id)

    renders_dir = sdk.path.join(sdk.paths.get("workspace"), "canvas_renders")

    if seed is not None:
        seed_val = int(seed)
    elif force_new_seed:
        import random
        seed_val = random.randint(1, 2_147_483_647)
    else:
        seed_val = state.get("render_seed") or _mint_seed()

    # Full-chain cache check.
    full_hash = _pool_hash(layers, width, height, palette_id)
    full_path = _cache_path(renders_dir, full_hash, seed_val)

    if not force_new_seed and sdk.fs.exists(full_path):
        sdk.services.call("canvas", "set_render_seed", canvas_id, seed_val)
        sdk.log(f"canvas_render: cache HIT pool={full_hash} seed={seed_val}")
        if out and out != full_path:
            sdk.fs.write_bytes(out, sdk.fs.read_bytes(full_path))
            return {"path": out, "seed": seed_val, "pool_hash": full_hash,
                    "cache_hit": True, "cached_layers": len(layers),
                    "total_layers": len(layers)}
        return {"path": full_path, "seed": seed_val, "pool_hash": full_hash,
                "cache_hit": True, "cached_layers": len(layers),
                "total_layers": len(layers)}

    if out is None:
        out = sdk.path.join(renders_dir, f"canvas_{int(time.time() * 1000)}.png")

    # Find longest cached prefix.
    start_idx = 0
    current_input = None
    for count in range(len(layers) - 1, 0, -1):
        prefix_hash = _pool_hash(layers[:count], width, height, palette_id)
        prefix_path = _cache_path(renders_dir, prefix_hash, seed_val)
        if sdk.fs.exists(prefix_path):
            start_idx = count
            current_input = prefix_path
            sdk.log(f"canvas_render: prefix cache HIT layers=0..{count - 1}")
            break

    sdk.log(f"canvas_render: rendering {len(layers)} layer(s) "
            f"from idx={start_idx}, seed={seed_val}")

    # Walk the chain.
    for idx, layer in enumerate(layers[start_idx:], start=start_idx):
        script_name = layer["script"]
        kind = layer["kind"]
        controls = dict(layer.get("controls") or {})

        step_hash = _pool_hash(layers[:idx + 1], width, height, palette_id)
        step_out = _cache_path(renders_dir, step_hash, seed_val)

        sdk.log(f"canvas_render: layer {idx} ({kind!r}) "
                f"script={script_name!r}")

        sdk.scripts.run(
            script_name + ".py",
            kind=kind,
            input_path=current_input,
            output_path=step_out,
            width=width,
            height=height,
            seed=seed_val,
            palette=palette,
            controls=controls,
        )
        current_input = step_out

    if current_input and out != current_input:
        sdk.fs.write_bytes(out, sdk.fs.read_bytes(current_input))

    sdk.services.call("canvas", "set_render_seed", canvas_id, seed_val)
    sdk.log(f"canvas_render: done -> {out}")
    return {
        "path": out, "seed": seed_val, "pool_hash": full_hash,
        "cache_hit": False, "cached_layers": start_idx,
        "total_layers": len(layers),
    }


def _pool_hash(layers, width, height, palette_id):
    payload = {
        "layers": [
            {
                "script": str(layer.get("script") or ""),
                "kind": str(layer.get("kind") or ""),
                "controls": dict(sorted(
                    (layer.get("controls") or {}).items()
                )),
            }
            for layer in layers
        ],
        "width": int(width),
        "height": int(height),
        "palette_id": str(palette_id),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:POOL_HASH_LEN]


def _cache_path(renders_dir, pool_hash, seed):
    return renders_dir + "/" + pool_hash + "/" + str(int(seed)) + ".png"


def _find_palette(palettes, palette_id):
    for p in palettes:
        if p.get("id") == palette_id:
            return p
    return palettes[0] if palettes else {"id": "default", "colors": {}}


def _mint_seed():
    import random
    return random.randint(1, 2_147_483_647)