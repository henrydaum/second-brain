"""Texture layer generators. Return RGBA with low alpha — they overlay forms."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw, ImageFilter

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


TEXTURE_STYLES = (
    "cross_hatch",
    "dot_grain",
    "brushwork",
    "fiber_weave",
    "static_noise",
    "scribble",
)

SCALE_MAP = {"fine": 0.6, "medium": 1.0, "coarse": 1.8}
STRENGTH_MAP = {"subtle": 50, "moderate": 100, "strong": 180}


def render(style: str, size: tuple[int, int], seed: int, scale: str = "medium", strength: str = "moderate", **_) -> Image.Image:
    fn = _DISPATCH.get(style, _static_noise)
    return fn(size, seed, SCALE_MAP.get(scale, 1.0), STRENGTH_MAP.get(strength, 100)).convert("RGBA")


def _blank(size):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _cross_hatch(size, seed, scale, alpha):
    rng = random.Random(seed)
    img = _blank(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    spacing = max(6, int(14 * scale))
    color = (20, 20, 20, alpha)
    offset = rng.randint(0, spacing)
    for k in range(-h, w + h, spacing):
        draw.line([(k - offset, 0), (k + h - offset, h)], fill=color, width=1)
    for k in range(-h, w + h, spacing):
        draw.line([(k + offset, 0), (k - h + offset, h)], fill=color, width=1)
    return img


def _dot_grain(size, seed, scale, alpha):
    if np is None:
        return _static_noise(size, seed, scale, alpha)
    rng = np.random.default_rng(seed)
    w, h = size
    n = int(w * h * 0.005 / scale)
    xs = rng.integers(0, w, n)
    ys = rng.integers(0, h, n)
    img = _blank(size)
    draw = ImageDraw.Draw(img, "RGBA")
    color = (15, 15, 15, alpha)
    radius = max(1, int(2 * scale))
    for x, y in zip(xs, ys):
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    return img


def _brushwork(size, seed, scale, alpha):
    rng = random.Random(seed)
    img = _blank(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    n = int(40 / max(scale, 0.5))
    for _ in range(n):
        x0 = rng.randint(0, w)
        y0 = rng.randint(0, h)
        length = rng.randint(40, 160)
        angle = rng.uniform(0, math.tau)
        x1 = int(x0 + math.cos(angle) * length)
        y1 = int(y0 + math.sin(angle) * length)
        gray = rng.randint(120, 200)
        a = max(20, alpha - 40)
        draw.line([(x0, y0), (x1, y1)], fill=(gray, gray, gray, a), width=max(2, int(4 * scale)))
    return img.filter(ImageFilter.GaussianBlur(1))


def _fiber_weave(size, seed, scale, alpha):
    if np is None:
        return _cross_hatch(size, seed, scale, alpha)
    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.standard_normal((h, w)).astype("float32")
    horizontal = np.tile(rng.standard_normal((1, w)).astype("float32") * 0.5, (h, 1))
    vertical = np.tile(rng.standard_normal((h, 1)).astype("float32") * 0.5, (1, w))
    field = base * 0.3 + horizontal + vertical
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    a = (field * alpha * 0.6).astype("uint8")
    gray = (field * 80 + 60).astype("uint8")
    rgba = np.stack([gray, gray, gray, a], axis=-1)
    img = Image.fromarray(rgba, "RGBA")
    return img


def _static_noise(size, seed, scale, alpha):
    if np is None:
        return _blank(size)
    rng = np.random.default_rng(seed)
    w, h = size
    noise = rng.standard_normal((h, w)).astype("float32")
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    a = (noise * alpha * 0.5).astype("uint8")
    rgb = (noise * 60 + 90).astype("uint8")
    rgba = np.stack([rgb, rgb, rgb, a], axis=-1)
    return Image.fromarray(rgba, "RGBA")


def _scribble(size, seed, scale, alpha):
    rng = random.Random(seed)
    img = _blank(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    n_lines = max(2, int(6 / max(scale, 0.5)))
    color = (30, 30, 30, alpha)
    for _ in range(n_lines):
        x, y = rng.randint(0, w), rng.randint(0, h)
        angle = rng.uniform(0, math.tau)
        pts = [(x, y)]
        for _ in range(rng.randint(30, 90)):
            angle += rng.uniform(-0.6, 0.6)
            step = rng.randint(6, 14)
            x = max(0, min(w, x + math.cos(angle) * step))
            y = max(0, min(h, y + math.sin(angle) * step))
            pts.append((x, y))
        draw.line(pts, fill=color, width=max(1, int(2 * scale)))
    return img


_DISPATCH = {
    "cross_hatch": _cross_hatch,
    "dot_grain": _dot_grain,
    "brushwork": _brushwork,
    "fiber_weave": _fiber_weave,
    "static_noise": _static_noise,
    "scribble": _scribble,
}
