"""Mandelbrot fractal renderer for the web demo."""

from __future__ import annotations

import colorsys
import json
import math
import random
import time
from pathlib import Path

from PIL import Image

from paths import DATA_DIR
from plugins.BaseTool import BaseTool, ToolResult
from plugins.tools.helpers.color_theory import beautify_image, palette_color
from plugins.tools.helpers.fractal_gallery import image_stats, mark_original, set_current


PRESETS = {
    "overview": (-0.5, 0.0, 3.1),
    "seahorse_valley": (-0.743643887, 0.131825904, 0.018),
    "elephant_valley": (0.285, 0.01, 0.018),
    "triple_spiral": (-0.088, 0.654, 0.026),
    "mini_brot": (-1.25066, 0.02012, 0.006),
    "deep_fire": (-0.748, 0.102, 0.012),
}
PALETTES = {"aurora", "inferno", "electric", "nebula", "gold"}
try:
    import numpy as np
except Exception:
    np = None


class RenderMandelbrot(BaseTool):
    """Render a beautiful bounded Mandelbrot image."""

    name = "render_mandelbrot"
    description = (
        "Create a vivid Mandelbrot fractal PNG and display it in the web demo. "
        "Prefer named presets unless the user asks for a custom location. "
        "Use palette/detail/magnification for creative variety; avoid repeated calls unless the user asks for another version."
    )
    parameters = {
        "type": "object",
        "properties": {
            "preset": {"type": "string", "enum": list(PRESETS), "description": "Fractal region to render."},
            "palette": {"type": "string", "enum": list(PALETTES), "description": "Color mood."},
            "width": {"type": "integer", "minimum": 640, "maximum": 1600, "description": "Image width."},
            "height": {"type": "integer", "minimum": 480, "maximum": 1200, "description": "Image height."},
            "detail": {"type": "integer", "minimum": 80, "maximum": 450, "description": "Iteration limit. Higher is slower and more detailed."},
            "magnification": {"type": "number", "minimum": 0.5, "maximum": 8.0, "description": "Zoom multiplier applied to the preset."},
            "seed": {"type": "integer", "description": "Optional color seed. Leave blank for a unique random render."},
            "center_x": {"type": "number", "minimum": -2.0, "maximum": 1.0, "description": "Optional custom real center."},
            "center_y": {"type": "number", "minimum": -1.5, "maximum": 1.5, "description": "Optional custom imaginary center."},
            "scale": {"type": "number", "minimum": 0.0005, "maximum": 3.2, "description": "Optional custom complex-plane width."},
        },
    }
    requires_services = []
    max_calls = 2

    def run(self, context, **kwargs) -> ToolResult:
        preset = _choice(kwargs.get("preset"), PRESETS, "seahorse_valley")
        palette = _choice(kwargs.get("palette"), PALETTES, "aurora")
        width = _clamp_int(kwargs.get("width"), 1200, 640, 1600)
        height = _clamp_int(kwargs.get("height"), 900, 480, 1200)
        detail = _clamp_int(kwargs.get("detail"), 220, 80, 450)
        magnification = _clamp_float(kwargs.get("magnification"), 1.0, 0.5, 8.0)
        seed = _clamp_int(kwargs.get("seed"), random.randint(1, 2_147_483_647), 1, 2_147_483_647)
        cx, cy, scale = PRESETS[preset]
        cx = _clamp_float(kwargs.get("center_x"), cx, -2.0, 1.0)
        cy = _clamp_float(kwargs.get("center_y"), cy, -1.5, 1.5)
        scale = _clamp_float(kwargs.get("scale"), scale / magnification, 0.0005, 3.2)

        out_dir = DATA_DIR / "fractals" / "mandelbrot"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"mandelbrot-{preset}-{palette}-{seed}-{stamp}.png"
        _render(path, width, height, detail, cx, cy, scale, palette, seed)
        beautify_image(Image.open(path), seed, palette, "mandelbrot").save(path, "PNG", optimize=True)
        meta = {"preset": preset, "palette": palette, "seed": seed, "width": width, "height": height, "detail": detail, "center_x": cx, "center_y": cy, "scale": scale, "path": str(path), "stats": image_stats(path)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mark_original(path, meta)
        set_current(getattr(context, "session_key", None), path, True, meta)
        return ToolResult(
            data=meta,
            llm_summary=f"Rendered a unique {palette} Mandelbrot seed {seed}: beauty {meta['stats']['beauty_score']}, brightness {meta['stats']['brightness']}, contrast {meta['stats']['contrast']}, detail {meta['stats']['detail']}, guidance={meta['stats']['guidance']}. Sharing is handled by the website button.",
            attachment_paths=[str(path)],
        )


def _render(path: Path, w: int, h: int, max_iter: int, cx: float, cy: float, scale: float, palette: str, seed: int) -> None:
    from PIL import Image

    fast = _render_np(w, h, max_iter, cx, cy, scale, palette, seed)
    if fast is not None:
        fast.save(path, "PNG", optimize=True)
        return
    rnd = random.Random(seed)
    shift, contrast, glow = rnd.random(), rnd.uniform(0.82, 1.28), rnd.uniform(0.85, 1.18)
    img = Image.new("RGB", (w, h))
    px = img.load()
    aspect = h / w
    for y in range(h):
        im = cy + (y / (h - 1) - 0.5) * scale * aspect
        for x in range(w):
            re = cx + (x / (w - 1) - 0.5) * scale
            zr = zi = 0.0
            i = 0
            while zr * zr + zi * zi <= 4.0 and i < max_iter:
                zr, zi = zr * zr - zi * zi + re, 2.0 * zr * zi + im
                i += 1
            if i >= max_iter:
                px[x, y] = (3, 4, 9)
                continue
            mag = max(zr * zr + zi * zi, 1.000001)
            smooth = i + 1 - math.log(math.log(mag, 2), 2)
            px[x, y] = _color((smooth / max_iter * contrast + shift) % 1.0, palette, glow)
    img.save(path, "PNG", optimize=True)


def _render_np(w: int, h: int, max_iter: int, cx: float, cy: float, scale: float, palette: str, seed: int):
    if np is None:
        return None
    from PIL import Image

    rnd = random.Random(seed)
    x = np.linspace(cx - scale / 2, cx + scale / 2, w, dtype=np.float32)
    y = np.linspace(cy - scale * h / w / 2, cy + scale * h / w / 2, h, dtype=np.float32)
    c = x[None, :] + 1j * y[:, None]
    z = np.zeros(c.shape, dtype=np.complex64)
    count = np.zeros(c.shape, dtype=np.float32)
    live = np.ones(c.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(1, max_iter + 1):
            z[live] = z[live] * z[live] + c[live]
            escaped = live & (np.abs(z) > 4)
            count[escaped] = i
            live[escaped] = False
            if not live.any():
                break
    t = (count / max_iter * rnd.uniform(0.82, 1.28) + rnd.random()) % 1
    arr = np.zeros((*count.shape, 3), dtype=np.uint8)
    base = {"inferno": .08, "electric": .56, "nebula": .74, "gold": .12, "aurora": .42}.get(palette, .42) * 6.283
    v = np.clip(.12 + .9 * t, 0, 1) * rnd.uniform(.85, 1.18)
    arr[..., 0] = (255 * v * (.5 + .5 * np.sin(base + t * 6.0))).clip(0, 255).astype("uint8")
    arr[..., 1] = (255 * v * (.5 + .5 * np.sin(base + 2.1 + t * 7.2))).clip(0, 255).astype("uint8")
    arr[..., 2] = (255 * v * (.5 + .5 * np.sin(base + 4.2 + t * 5.1))).clip(0, 255).astype("uint8")
    arr[count == 0] = (3, 4, 9)
    return Image.fromarray(arr, "RGB")


def _color(t: float, palette: str, glow: float = 1.0) -> tuple[int, int, int]:
    return palette_color(t, palette, 1, glow)


def _choice(value, allowed, default):
    return value if value in allowed else default


def _clamp_int(value, default, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def _clamp_float(value, default, low, high):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default
