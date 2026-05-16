"""Background layer generators. Always return OPAQUE RGBA filling the full canvas."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFilter

from plugins.tools.helpers.color_theory import harmony

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


BACKGROUND_STYLES = (
    "flat_wash",
    "vertical_gradient",
    "radial_gradient",
    "cloud_noise",
    "paper_grain",
    "sky_band",
)

DENSITY_NOISE = {"minimal": 0.04, "moderate": 0.10, "rich": 0.18, "dense": 0.28}


def _pal(seed: int):
    return harmony("aurora", seed)


def render(style: str, size: tuple[int, int], seed: int, density: str = "moderate", **_) -> Image.Image:
    fn = _DISPATCH.get(style, _flat_wash)
    return fn(size, seed, density).convert("RGBA")


def _flat_wash(size, seed, density):
    pal = _pal(seed)
    color = pal[1] if random.Random(seed + 1).random() < 0.5 else pal[0]
    return Image.new("RGB", size, color)


def _vertical_gradient(size, seed, density):
    pal = _pal(seed)
    top, bottom = pal[2], pal[0]
    if np is not None:
        w, h = size
        t = np.linspace(0, 1, h, dtype="float32")[:, None]
        arr = np.zeros((h, w, 3), dtype="float32")
        for c in range(3):
            arr[..., c] = top[c] * (1 - t) + bottom[c] * t
        return Image.fromarray(arr.astype("uint8"), "RGB")
    img = Image.new("RGB", size, top)
    draw = ImageDraw.Draw(img)
    for y in range(size[1]):
        t = y / max(1, size[1] - 1)
        c = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line([(0, y), (size[0], y)], fill=c)
    return img


def _radial_gradient(size, seed, density):
    pal = _pal(seed)
    inner, outer = pal[2], pal[0]
    if np is None:
        return _vertical_gradient(size, seed, density)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    cy, cx = h / 2, w / 2
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    t = np.clip(r, 0, 1)[..., None]
    inner_a = np.array(inner, dtype="float32")
    outer_a = np.array(outer, dtype="float32")
    arr = inner_a * (1 - t) + outer_a * t
    return Image.fromarray(arr.astype("uint8"), "RGB")


def _cloud_noise(size, seed, density):
    if np is None:
        return _flat_wash(size, seed, density)
    pal = _pal(seed)
    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.standard_normal((h // 8, w // 8)).astype("float32")
    img = Image.fromarray(((base - base.min()) / (base.max() - base.min() + 1e-6) * 255).astype("uint8"))
    img = img.resize(size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(8))
    arr = np.asarray(img, dtype="float32") / 255.0
    amp = DENSITY_NOISE.get(density, 0.1)
    lo = np.array(pal[0], dtype="float32")
    hi = np.array(pal[2], dtype="float32")
    out = lo * (1 - arr[..., None]) + hi * arr[..., None]
    # gently lighten with noise amplitude factor
    out = np.clip(out + (arr[..., None] - 0.5) * amp * 80, 0, 255)
    return Image.fromarray(out.astype("uint8"), "RGB")


def _paper_grain(size, seed, density):
    pal = _pal(seed)
    base_color = (max(pal[2][0], 230), max(pal[2][1], 225), max(pal[2][2], 215))
    img = Image.new("RGB", size, base_color)
    if np is None:
        return img
    rng = np.random.default_rng(seed + 7)
    arr = np.asarray(img, dtype="float32") / 255.0
    noise = rng.standard_normal((size[1], size[0])).astype("float32") * (0.04 + DENSITY_NOISE.get(density, 0.1) * 0.2)
    arr = np.clip(arr + noise[..., None], 0, 1)
    return Image.fromarray((arr * 255).astype("uint8"), "RGB")


def _sky_band(size, seed, density):
    pal = _pal(seed)
    w, h = size
    horizon = h * random.Random(seed + 3).uniform(0.45, 0.7)
    if np is None:
        img = Image.new("RGB", size, pal[2])
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, int(horizon), w, h], fill=pal[0])
        return img
    yy = np.arange(h, dtype="float32")
    top_t = np.clip(yy / max(1, horizon), 0, 1)[:, None]
    bot_t = np.clip((yy - horizon) / max(1, h - horizon), 0, 1)[:, None]
    sky_top = np.array(pal[3], dtype="float32")
    sky_bot = np.array(pal[2], dtype="float32")
    ground_top = np.array(pal[1], dtype="float32")
    ground_bot = np.array(pal[0], dtype="float32")
    above = sky_top * (1 - top_t) + sky_bot * top_t
    below = ground_top * (1 - bot_t) + ground_bot * bot_t
    mask = (yy[:, None] < horizon)[..., None]
    rows = np.where(mask, above, below)
    arr = np.broadcast_to(rows[:, None, :], (h, w, 3)).copy()
    return Image.fromarray(arr.astype("uint8"), "RGB")


_DISPATCH = {
    "flat_wash": _flat_wash,
    "vertical_gradient": _vertical_gradient,
    "radial_gradient": _radial_gradient,
    "cloud_noise": _cloud_noise,
    "paper_grain": _paper_grain,
    "sky_band": _sky_band,
}
