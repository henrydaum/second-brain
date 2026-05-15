"""Mandelbrot fractal renderer for the web demo."""

from __future__ import annotations

import colorsys
import json
import math
import time
from pathlib import Path

from paths import DATA_DIR
from plugins.BaseTool import BaseTool, ToolResult


PRESETS = {
    "overview": (-0.5, 0.0, 3.1),
    "seahorse_valley": (-0.743643887, 0.131825904, 0.018),
    "elephant_valley": (0.285, 0.01, 0.018),
    "triple_spiral": (-0.088, 0.654, 0.026),
    "mini_brot": (-1.25066, 0.02012, 0.006),
    "deep_fire": (-0.748, 0.102, 0.012),
}
PALETTES = {"aurora", "inferno", "electric", "nebula", "gold"}


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
        cx, cy, scale = PRESETS[preset]
        cx = _clamp_float(kwargs.get("center_x"), cx, -2.0, 1.0)
        cy = _clamp_float(kwargs.get("center_y"), cy, -1.5, 1.5)
        scale = _clamp_float(kwargs.get("scale"), scale / magnification, 0.0005, 3.2)

        out_dir = DATA_DIR / "fractals" / "mandelbrot"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out_dir / f"mandelbrot-{preset}-{palette}-{stamp}.png"
        _render(path, width, height, detail, cx, cy, scale, palette)
        meta = {"preset": preset, "palette": palette, "width": width, "height": height, "detail": detail, "center_x": cx, "center_y": cy, "scale": scale, "path": str(path)}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return ToolResult(
            data=meta,
            llm_summary=f"Rendered a {palette} Mandelbrot image at preset '{preset}' and displayed it in the showcase pane: {path}",
            attachment_paths=[str(path)],
        )


def _render(path: Path, w: int, h: int, max_iter: int, cx: float, cy: float, scale: float, palette: str) -> None:
    from PIL import Image

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
            px[x, y] = _color(smooth / max_iter, palette)
    img.save(path, "PNG", optimize=True)


def _color(t: float, palette: str) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if palette == "inferno":
        h, s, v = 0.03 + 0.12 * t, 0.95, 0.18 + 0.82 * t
    elif palette == "electric":
        h, s, v = 0.55 + 0.28 * math.sin(t * 5.2), 0.9, 0.2 + 0.8 * t
    elif palette == "nebula":
        h, s, v = 0.72 + 0.22 * t, 0.78, 0.16 + 0.84 * t
    elif palette == "gold":
        h, s, v = 0.10 + 0.05 * math.sin(t * 8), 0.82, 0.12 + 0.88 * t
    else:
        h, s, v = 0.44 + 0.35 * t, 0.82, 0.15 + 0.85 * t
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h % 1.0, s, v))


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
