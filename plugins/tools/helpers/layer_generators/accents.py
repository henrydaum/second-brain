"""Accent layer generators. Sparse focal marks. RGBA with mostly-transparent alpha."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

from plugins.tools.helpers.color_theory import harmony


ACCENT_STYLES = (
    "sparks",
    "drip_marks",
    "stars",
    "tally_marks",
    "arrows",
    "dots",
)

COUNT_MAP = {"few": 8, "several": 22, "many": 60}
SIZE_MAP = {"small": 0.6, "medium": 1.0, "large": 1.8}


def render(style: str, size: tuple[int, int], seed: int, count: str = "several", size_hint: str = "medium", **kwargs) -> Image.Image:
    # accept both `size_hint` and `size` keyword from callers; explicit named-arg precedence
    sz = kwargs.get("element_size") or size_hint
    fn = _DISPATCH.get(style, _sparks)
    return fn(size, seed, COUNT_MAP.get(count, 22), SIZE_MAP.get(sz, 1.0)).convert("RGBA")


def _blank(size):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _palette(seed):
    return harmony("electric", seed)


def _sparks(canvas_size, seed, n, scale):
    rng = random.Random(seed)
    pal = _palette(seed)
    img = _blank(canvas_size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = canvas_size
    for _ in range(n):
        cx = rng.randint(0, w); cy = rng.randint(0, h)
        r = int(rng.randint(2, 6) * scale)
        color = pal[rng.randint(2, 3)]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, 230))
        # cross hairs
        line_len = r * 4
        draw.line([(cx - line_len, cy), (cx + line_len, cy)], fill=(*color, 170), width=1)
        draw.line([(cx, cy - line_len), (cx, cy + line_len)], fill=(*color, 170), width=1)
    return img


def _drip_marks(canvas_size, seed, n, scale):
    rng = random.Random(seed)
    pal = _palette(seed)
    img = _blank(canvas_size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = canvas_size
    n = max(3, n // 3)
    for _ in range(n):
        x = rng.randint(0, w)
        y0 = rng.randint(0, int(h * 0.4))
        length = rng.randint(80, int(h * 0.7))
        width = max(2, int(rng.randint(3, 8) * scale))
        color = pal[rng.randint(1, 3)]
        draw.line([(x, y0), (x, y0 + length)], fill=(*color, 220), width=width)
        # drip blob at end
        draw.ellipse([x - width, y0 + length - width, x + width, y0 + length + width],
                     fill=(*color, 230))
    return img


def _stars(canvas_size, seed, n, scale):
    rng = random.Random(seed)
    img = _blank(canvas_size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = canvas_size
    for _ in range(n):
        cx = rng.randint(0, w); cy = rng.randint(0, h)
        r = int(rng.randint(3, 9) * scale)
        pts = []
        for k in range(10):
            theta = (math.tau / 10) * k - math.pi / 2
            rr = r if k % 2 == 0 else r * 0.45
            pts.append((cx + math.cos(theta) * rr, cy + math.sin(theta) * rr))
        draw.polygon(pts, fill=(255, 240, 200, 235))
    return img


def _tally_marks(canvas_size, seed, n, scale):
    rng = random.Random(seed)
    img = _blank(canvas_size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = canvas_size
    color = (15, 15, 15, 235)
    for _ in range(n):
        x = rng.randint(0, w); y = rng.randint(0, h)
        length = int(rng.randint(14, 28) * scale)
        angle = rng.uniform(-0.3, 0.3) + math.pi / 2
        x2 = x + math.cos(angle) * length
        y2 = y + math.sin(angle) * length
        draw.line([(x, y), (x2, y2)], fill=color, width=max(1, int(2 * scale)))
    return img


def _arrows(canvas_size, seed, n, scale):
    rng = random.Random(seed)
    pal = _palette(seed)
    img = _blank(canvas_size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = canvas_size
    for _ in range(n):
        x = rng.randint(0, w); y = rng.randint(0, h)
        length = int(rng.randint(20, 50) * scale)
        angle = rng.uniform(0, math.tau)
        x2 = x + math.cos(angle) * length
        y2 = y + math.sin(angle) * length
        color = pal[rng.randint(2, 3)]
        draw.line([(x, y), (x2, y2)], fill=(*color, 230), width=max(2, int(2 * scale)))
        # arrowhead
        head_len = max(4, int(8 * scale))
        for da in (-0.5, 0.5):
            hx = x2 - math.cos(angle + da) * head_len
            hy = y2 - math.sin(angle + da) * head_len
            draw.line([(x2, y2), (hx, hy)], fill=(*color, 230), width=max(2, int(2 * scale)))
    return img


def _dots(canvas_size, seed, n, scale):
    rng = random.Random(seed)
    pal = _palette(seed)
    img = _blank(canvas_size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = canvas_size
    for _ in range(n):
        cx = rng.randint(0, w); cy = rng.randint(0, h)
        r = int(rng.randint(5, 12) * scale)
        color = pal[rng.randint(1, 3)]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, 220))
    return img


_DISPATCH = {
    "sparks": _sparks,
    "drip_marks": _drip_marks,
    "stars": _stars,
    "tally_marks": _tally_marks,
    "arrows": _arrows,
    "dots": _dots,
}
