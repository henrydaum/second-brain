"""Compose the layer stack into a single image. Palette + atmosphere are passes."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from plugins.tools.helpers import layered_canvas as lc
from plugins.tools.helpers.color_theory import harmony, visual_stats

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _open_rgba(path: str | Path, size: tuple[int, int]) -> Image.Image | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    img = Image.open(p).convert("RGBA")
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def _blank(size: tuple[int, int]) -> Image.Image:
    # Plain mid-grey opaque background fallback if no background layer is set.
    return Image.new("RGBA", size, (40, 40, 44, 255))


def _assert_transparent(img: Image.Image, slot: str) -> None:
    """Non-background layers must have at least some transparency."""
    if np is None:
        return
    alpha = np.asarray(img.split()[-1])
    if alpha.min() == 255:
        raise ValueError(f"layer '{slot}' has no transparent pixels — generator forgot to set alpha")


def composite(state: dict) -> Image.Image:
    """Build the final image from a canvas state dict."""
    size = tuple(state.get("size") or lc.DEFAULT_SIZE)
    layers = state.get("layers") or {}

    bg = _open_rgba((layers.get("background") or {}).get("path"), size) or _blank(size)
    base = bg.convert("RGBA")

    for slot in ("form", "texture", "accent"):
        entry = layers.get(slot)
        if not entry:
            continue
        img = _open_rgba(entry.get("path"), size)
        if img is None:
            continue
        try:
            _assert_transparent(img, slot)
        except ValueError:
            # Don't crash recompose. Treat fully-opaque as a mistake and skip.
            continue
        base = Image.alpha_composite(base, img)

    palette_pass = state.get("palette")
    if palette_pass:
        base = _apply_palette(base, **(palette_pass.get("params") or {}))

    atmosphere_pass = state.get("atmosphere")
    if atmosphere_pass:
        base = _apply_atmosphere(base, **(atmosphere_pass.get("params") or {}))

    return base.convert("RGB")


# --- palette pass ----------------------------------------------------------

_STRENGTH_MAP = {"subtle": 0.35, "moderate": 0.6, "strong": 0.85, "none": 0.0}
_INTENSITY_MAP = _STRENGTH_MAP


def _palette_lut(scheme: str, seed: int) -> "np.ndarray":
    """Build a 256-entry RGB lookup from luminance to palette ramp."""
    pal = harmony(scheme, seed)
    pal_arr = np.array(pal, dtype="float32") / 255.0  # (4, 3)
    lut = np.zeros((256, 3), dtype="float32")
    for i in range(256):
        t = i / 255.0
        x = t * (len(pal_arr) - 1)
        lo = int(x); hi = min(len(pal_arr) - 1, lo + 1); f = x - lo
        lut[i] = pal_arr[lo] * (1 - f) + pal_arr[hi] * f
    return lut


def _apply_palette(img: Image.Image, scheme: str = "rothko_warm", intensity: str | float = "moderate", seed: int = 1, **_) -> Image.Image:
    if np is None:
        # PIL fallback: gentle hue shift via colorize on luminance.
        return img
    mix = _INTENSITY_MAP.get(str(intensity), float(intensity) if isinstance(intensity, (int, float)) else 0.6)
    if mix <= 0:
        return img
    arr = np.asarray(img.convert("RGBA"), dtype="float32") / 255.0
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    # Stretch luminance to fill the LUT so flat-color forms don't collapse.
    lo, hi = float(np.percentile(lum, 2)), float(np.percentile(lum, 98))
    span = max(hi - lo, 1e-3)
    lum_n = np.clip((lum - lo) / span, 0.0, 1.0)
    idx = np.clip((lum_n * 255).astype("int32"), 0, 255)
    lut = _palette_lut(scheme, seed)
    mapped = lut[idx]
    out = rgb * (1 - mix) + mapped * mix
    out = np.concatenate([out, a], axis=-1)
    out = np.clip(out * 255.0, 0, 255).astype("uint8")
    return Image.fromarray(out, "RGBA")


# --- atmosphere pass -------------------------------------------------------

def _apply_atmosphere(img: Image.Image, style: str = "none", strength: str | float = "moderate", seed: int = 1, **_) -> Image.Image:
    if style in (None, "none"):
        return img
    s = _STRENGTH_MAP.get(str(strength), float(strength) if isinstance(strength, (int, float)) else 0.6)
    if s <= 0:
        return img
    fn = _ATMOSPHERE.get(style)
    if not fn:
        return img
    return fn(img.convert("RGB"), s, seed).convert("RGBA")


def _vignette(img: Image.Image, s: float, seed: int) -> Image.Image:
    if np is None:
        return img
    arr = np.asarray(img).astype("float32") / 255.0
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    cy, cx = h / 2, w / 2
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    falloff = np.clip(1.0 - (r * 0.85) ** 2.4 * s, 0.0, 1.0)
    arr = arr * falloff[..., None]
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"), "RGB")


def _golden_hour(img: Image.Image, s: float, seed: int) -> Image.Image:
    warm = Image.new("RGB", img.size, (255, 190, 110))
    out = Image.blend(img, ImageChops.multiply(img, warm), 0.4 * s)
    out = ImageEnhance.Color(out).enhance(1 + 0.25 * s)
    return out


def _bloom(img: Image.Image, s: float, seed: int) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(6 + 10 * s))
    glow = ImageEnhance.Brightness(blurred).enhance(0.65)
    return ImageChops.screen(img, glow)


def _film_grain(img: Image.Image, s: float, seed: int) -> Image.Image:
    if np is None:
        return img
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype("float32") / 255.0
    noise = rng.standard_normal(arr.shape[:2]).astype("float32") * (0.06 * s)
    arr = np.clip(arr + noise[..., None], 0, 1)
    return Image.fromarray((arr * 255).astype("uint8"), "RGB")


def _soft_haze(img: Image.Image, s: float, seed: int) -> Image.Image:
    blurred = img.filter(ImageFilter.GaussianBlur(2 + 4 * s))
    out = Image.blend(img, blurred, 0.4 * s)
    return ImageEnhance.Contrast(out).enhance(1 - 0.1 * s)


def _chromatic_drift(img: Image.Image, s: float, seed: int) -> Image.Image:
    r, g, b = img.split()
    shift = max(1, int(6 * s))
    r = ImageChops.offset(r, shift, 0)
    b = ImageChops.offset(b, -shift, 0)
    return Image.merge("RGB", (r, g, b))


def _noir(img: Image.Image, s: float, seed: int) -> Image.Image:
    gray = img.convert("L").convert("RGB")
    out = Image.blend(img, gray, 0.6 * s)
    out = ImageEnhance.Contrast(out).enhance(1 + 0.4 * s)
    return _vignette(out, 0.6 * s, seed)


_ATMOSPHERE = {
    "noir_vignette": _vignette,
    "golden_hour": _golden_hour,
    "bloom": _bloom,
    "film_grain": _film_grain,
    "soft_haze": _soft_haze,
    "chromatic_drift": _chromatic_drift,
    "noir": _noir,
}


def recompose(session_key: str) -> tuple[Path, dict]:
    """Recomposite the canvas for this session, write to disk, return (path, stats)."""
    state = lc.get_state(session_key)
    img = composite(state)
    out_path = lc.composite_path(session_key)
    img.save(out_path, "PNG", optimize=True)
    stats = visual_stats(img)
    lc.record_composite(session_key, out_path, stats)
    return out_path, stats
