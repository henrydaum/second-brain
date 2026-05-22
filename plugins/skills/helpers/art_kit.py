"""Curated helpers injected into the skill sandbox as `art_kit`.

Skills cannot import this module — the sandbox import gate blocks it. Instead,
the sandbox entry calls `build_namespace(canvas)` and exposes the returned
SimpleNamespace under the name `art_kit` in the skill's exec namespace.

Everything here is pure: no I/O, no global mutation. All randomness must take a
caller-supplied `random.Random` so skills stay deterministic from `canvas.seed`.
"""

from __future__ import annotations

import colorsys
import math
import random
from pathlib import Path
from types import SimpleNamespace

from PIL import ImageDraw, ImageFont

from plugins.tools.helpers.color_theory import oklch_to_rgb as _oklch_to_rgb


# ---------------------------------------------------------------------------
# Math primitives.
# ---------------------------------------------------------------------------

def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def smoothstep(t, edge0=0.0, edge1=1.0):
    x = clamp((t - edge0) / ((edge1 - edge0) or 1e-9), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def remap(x, in_lo, in_hi, out_lo, out_hi):
    t = (x - in_lo) / ((in_hi - in_lo) or 1e-9)
    return out_lo + (out_hi - out_lo) * t


# ---------------------------------------------------------------------------
# Color helpers.
# ---------------------------------------------------------------------------

def hex_to_rgb(h):
    h = str(h).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb):
    r, g, b = (int(round(clamp(c, 0, 255))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def mix_hex(a, b, t):
    """Linear RGB mix of two hex colors. t=0 -> a, t=1 -> b."""
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return rgb_to_hex((lerp(ar, br, t), lerp(ag, bg, t), lerp(ab, bb, t)))


def _luminance(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _palette_ramp(canvas_palette):
    """Return palette slots sorted by luminance, dark → bright. Cached on the namespace."""
    slots = ["background", "tertiary", "secondary", "primary", "accent"]
    pairs = []
    for slot in slots:
        hex_color = getattr(canvas_palette, slot, None)
        if hex_color is None:
            continue
        pairs.append((str(hex_color), _luminance(hex_to_rgb(hex_color))))
    pairs.sort(key=lambda p: p[1])
    return [p[0] for p in pairs] or ["#000000", "#ffffff"]


def _palette_color_fn(canvas_palette):
    ramp = _palette_ramp(canvas_palette)

    def palette_color(t, value=1.0):
        """Sample the luminance-sorted palette ramp at t∈[0,1]. Returns hex."""
        t = clamp(float(t), 0.0, 1.0)
        if len(ramp) == 1:
            return ramp[0]
        x = t * (len(ramp) - 1)
        i = int(x)
        if i >= len(ramp) - 1:
            return ramp[-1]
        f = x - i
        r1, g1, b1 = hex_to_rgb(ramp[i])
        r2, g2, b2 = hex_to_rgb(ramp[i + 1])
        v = clamp(float(value), 0.0, 1.5)
        return rgb_to_hex((lerp(r1, r2, f) * v, lerp(g1, g2, f) * v, lerp(b1, b2, f) * v))

    return palette_color


def oklch_to_rgb(l, c, h):
    """Re-export of color_theory.oklch_to_rgb. Hue h is in turns (0..1)."""
    return _oklch_to_rgb(l, c, h)


# ---------------------------------------------------------------------------
# Composition.
# ---------------------------------------------------------------------------

def rule_of_thirds(size):
    """Return the four rule-of-thirds anchor points and the guide lines.

    For a square canvas of `size` pixels, returns a SimpleNamespace with:
      .points    -> list of (x, y) intersections (4 points)
      .verticals -> (x1, x2) column guides
      .horizons  -> (y1, y2) row guides (use y1 as a sky-horizon)
    """
    s = int(size)
    a, b = s // 3, (2 * s) // 3
    return SimpleNamespace(
        points=[(a, a), (b, a), (a, b), (b, b)],
        verticals=(a, b),
        horizons=(a, b),
    )


def vogel_spiral(n, scale=1.0):
    """Sunflower-style point distribution using the golden angle.

    Yields (x, y) pairs in roughly [-scale, +scale]. Good for petals, seeds,
    star fields, or any radial "filled disc" arrangement.
    """
    n = max(0, int(n))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        r = math.sqrt((i + 0.5) / n) * scale
        theta = i * golden
        out.append((r * math.cos(theta), r * math.sin(theta)))
    return out


def jittered_grid(rng, cols, rows, jitter=0.4):
    """Centers of a cols×rows grid in [0,1]², each jittered within its cell."""
    out = []
    cw, ch = 1.0 / max(1, cols), 1.0 / max(1, rows)
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cw + (rng.random() - 0.5) * cw * jitter
            cy = (r + 0.5) * ch + (rng.random() - 0.5) * ch * jitter
            out.append((cx, cy))
    return out


# ---------------------------------------------------------------------------
# Noise.
# ---------------------------------------------------------------------------

def _hash01(rng_seed, ix, iy):
    # Deterministic value at lattice point (ix, iy). Uses a string seed for
    # Python 3.13 compatibility (tuple seeds were removed). Stable across runs.
    return random.Random(f"{rng_seed}:{int(ix)}:{int(iy)}").random()


def value_noise(seed, x, y):
    """Smooth 2D value noise in [0,1]. Inputs are continuous; integer steps are
    one lattice cell. Cheap drop-in alternative to perlin."""
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    sx, sy = smoothstep(fx), smoothstep(fy)
    n00 = _hash01(seed, x0, y0)
    n10 = _hash01(seed, x0 + 1, y0)
    n01 = _hash01(seed, x0, y0 + 1)
    n11 = _hash01(seed, x0 + 1, y0 + 1)
    return lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sy)


def fbm(seed, x, y, octaves=4, lacunarity=2.0, gain=0.5):
    """Fractional Brownian motion over value_noise. Returns ~[0,1]."""
    total = 0.0
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(int(octaves)):
        total += value_noise(seed, x * freq, y * freq) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / (norm or 1.0)


# ---------------------------------------------------------------------------
# Falloffs / masks. Returned as functions (call per pixel) to avoid numpy in
# the namespace surface — skills can still use numpy directly if they import it.
# ---------------------------------------------------------------------------

def radial_falloff(w, h, cx=None, cy=None):
    """Return a closure f(x, y) -> 1 at center, 0 at the canvas corner."""
    cx = (w / 2.0) if cx is None else float(cx)
    cy = (h / 2.0) if cy is None else float(cy)
    max_d = math.hypot(max(cx, w - cx), max(cy, h - cy)) or 1.0

    def f(x, y):
        return 1.0 - clamp(math.hypot(x - cx, y - cy) / max_d, 0.0, 1.0)

    return f


# ---------------------------------------------------------------------------
# Numpy primitives for transform skills.
#
# These let lens / warp / glitch transforms skip the most-repeated boilerplate:
# building a centered coordinate grid and resampling an array at fractional
# coordinates. Numpy is imported lazily so first-party skills that don't use
# these (creation skills, PIL-only transforms) pay no import cost.
# ---------------------------------------------------------------------------

def centered_grid(size):
    """Return (xx, yy, nx, ny) for an ``size x size`` canvas.

    xx, yy: float32 pixel coordinates (0..size-1).
    nx, ny: normalized to [-1, +1] from the canvas center — what every radial
            distortion (fisheye, CA, vignette) actually wants.

    Skills that want a custom center can compute their own normalization off
    of xx/yy; this helper covers the 95% case.
    """
    import numpy as _np
    s = int(size)
    yy, xx = _np.mgrid[0:s, 0:s].astype(_np.float32)
    c = (s - 1) / 2.0
    half = max(c, 1.0)
    nx = (xx - c) / half
    ny = (yy - c) / half
    return xx, yy, nx, ny


def bilinear_sample(arr, fx, fy):
    """Bilinear resample ``arr`` at fractional coordinates ``(fx, fy)``.

    ``arr`` may be 2D (H, W) for a single channel or 3D (H, W, C) for color.
    ``fx`` and ``fy`` are float arrays of the same shape as the output you
    want — typically the same shape as ``arr``'s first two dimensions.
    Coordinates outside the array are clamped to the edge.

    This is the workhorse for fisheye, polar coordinates, kaleidoscope,
    chromatic aberration, and any other warp transform.
    """
    import numpy as _np
    a = _np.asarray(arr)
    if a.ndim not in (2, 3):
        raise ValueError(f"bilinear_sample expects a 2D or 3D array, got shape {a.shape}")
    h, w = a.shape[:2]
    fx = _np.clip(_np.asarray(fx, dtype=_np.float32), 0, w - 1)
    fy = _np.clip(_np.asarray(fy, dtype=_np.float32), 0, h - 1)
    x0 = _np.floor(fx).astype(_np.int32)
    y0 = _np.floor(fy).astype(_np.int32)
    x1 = _np.clip(x0 + 1, 0, w - 1)
    y1 = _np.clip(y0 + 1, 0, h - 1)
    wx = fx - x0
    wy = fy - y0
    if a.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    p00 = a[y0, x0]
    p10 = a[y0, x1]
    p01 = a[y1, x0]
    p11 = a[y1, x1]
    top = p00 * (1.0 - wx) + p10 * wx
    bot = p01 * (1.0 - wx) + p11 * wx
    return top * (1.0 - wy) + bot * wy


# ---------------------------------------------------------------------------
# Voronoi.
# ---------------------------------------------------------------------------

def voronoi_nearest(points):
    """Return a closure f(x, y) -> (index, distance) of the nearest seed.

    Pure Python; skills that want a full per-pixel map and care about speed
    should implement Voronoi inline with numpy broadcasting. This form is
    useful for sparse sampling or for caller-driven decisions (e.g. shade
    edges when distance to second-nearest is small).
    """
    pts = [(float(px), float(py)) for (px, py) in points]

    def f(x, y):
        best_i, best_d2 = 0, float("inf")
        for i, (px, py) in enumerate(pts):
            dx, dy = x - px, y - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i, math.sqrt(best_d2)

    return f


# ---------------------------------------------------------------------------
# Flow fields.
# ---------------------------------------------------------------------------

def flow_field(seed, scale=0.005, octaves=4):
    """Return a closure f(x, y) -> angle_radians driven by fbm noise.

    Skills can advect particles or draw streamlines through it. `scale` controls
    how tightly the field swirls; smaller -> broader, smoother swirls.
    """

    def f(x, y):
        return fbm(seed, x * scale, y * scale, octaves=octaves) * math.tau

    return f


# ---------------------------------------------------------------------------
# L-systems.
# ---------------------------------------------------------------------------

def lindenmayer(axiom, rules, iterations):
    """Iteratively rewrite `axiom` using a {char: replacement} dict."""
    s = str(axiom)
    rules = dict(rules)
    for _ in range(int(iterations)):
        s = "".join(rules.get(ch, ch) for ch in s)
    return s


def turtle_segments(sentence, start=(0.0, 0.0), heading=None, step=10.0, turn=None):
    """Interpret an L-system string as turtle moves; return line segments.

    Symbols:
      F, G  forward by `step`, emit a segment.
      f     forward by `step` without emitting.
      +     turn right by `turn`.
      -     turn left by `turn`.
      [ ]   push / pop (position, heading).
    Other symbols are ignored (useful for non-drawing terminals like X, Y).

    Returns a list of (x1, y1, x2, y2) tuples for ImageDraw.line().
    """
    if heading is None:
        heading = -math.pi / 2.0
    if turn is None:
        turn = math.radians(25.0)
    x, y = float(start[0]), float(start[1])
    h = float(heading)
    step = float(step)
    turn = float(turn)
    stack = []
    out = []
    for ch in sentence:
        if ch == "F" or ch == "G":
            nx = x + math.cos(h) * step
            ny = y + math.sin(h) * step
            out.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch == "f":
            x += math.cos(h) * step
            y += math.sin(h) * step
        elif ch == "+":
            h += turn
        elif ch == "-":
            h -= turn
        elif ch == "[":
            stack.append((x, y, h))
        elif ch == "]":
            if stack:
                x, y, h = stack.pop()
    return out


# ---------------------------------------------------------------------------
# Wave interference.
# ---------------------------------------------------------------------------

def wave_field(sources):
    """Return a closure f(x, y) -> summed wave intensity.

    `sources` is a list of (cx, cy, wavelength, phase). Each contributes
    sin(2π * (dist/wavelength + phase)). The sum is roughly in [-N, +N] for N
    sources -- callers should normalize or pass through tanh / smoothstep.
    """
    srcs = [(float(cx), float(cy), float(wl) or 1e-9, float(ph)) for (cx, cy, wl, ph) in sources]

    def f(x, y):
        total = 0.0
        for cx, cy, wl, ph in srcs:
            d = math.hypot(x - cx, y - cy)
            total += math.sin(math.tau * (d / wl + ph))
        return total

    return f


# ---------------------------------------------------------------------------
# Strange attractors.
# ---------------------------------------------------------------------------

def attractor_points(name, n, seed, params=None):
    """Return n (x, y) points from a 2D strange attractor, normalized to ~[-1, 1].

    Supported names: "de_jong", "clifford". If `params` is None, the (a, b, c, d)
    constants are drawn from `seed` in a known-interesting range.
    """
    n = max(0, int(n))
    rng = random.Random(f"art_kit.attractor:{seed}")
    kind = str(name).lower()
    if params is None:
        if kind == "clifford":
            params = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0),
                      rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0))
        else:
            params = (rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0),
                      rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0))
    a, b, c, d = (float(p) for p in params)

    def step(x, y):
        if kind == "clifford":
            return (math.sin(a * y) + c * math.cos(a * x),
                    math.sin(b * x) + d * math.cos(b * y))
        return (math.sin(a * y) - math.cos(b * x),
                math.sin(c * x) - math.cos(d * y))

    x, y = 0.1, 0.1
    for _ in range(100):  # burn-in
        x, y = step(x, y)
    out = []
    for _ in range(n):
        x, y = step(x, y)
        out.append((x / 2.5, y / 2.5))
    return out


# ---------------------------------------------------------------------------
# Text rendering (Jost).
# ---------------------------------------------------------------------------

_FONTS_DIR = Path(__file__).resolve().parents[3] / "fonts"

_FONT_FILES = {
    ("light", False):   "Jost-300-Light.ttf",
    ("light", True):    "Jost-300-LightItalic.ttf",
    ("regular", False): "Jost-400-Book.ttf",
    ("regular", True):  "Jost-400-BookItalic.ttf",
    ("bold", False):    "Jost-700-Bold.ttf",
    ("bold", True):     "Jost-700-BoldItalic.ttf",
    ("black", False):   "Jost-900-Black.ttf",
    ("black", True):    "Jost-900-BlackItalic.ttf",
}

_FONT_CACHE: dict[tuple[str, bool, int], ImageFont.FreeTypeFont] = {}


def _load_font(weight: str, italic: bool, size: int) -> ImageFont.FreeTypeFont:
    key = (weight, bool(italic), int(size))
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    filename = _FONT_FILES.get((weight, bool(italic)))
    if filename is None:
        raise ValueError(
            f"unknown font variant: weight={weight!r}, italic={italic!r}. "
            f"weight must be one of: light, regular, bold, black"
        )
    font = ImageFont.truetype(str(_FONTS_DIR / filename), int(size))
    _FONT_CACHE[key] = font
    return font


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> list[str]:
    """Greedy word-wrap on whitespace. Preserves explicit \\n line breaks."""
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        words = paragraph.split(" ")
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            bbox = font.getbbox(trial)
            if (bbox[2] - bbox[0]) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def text(image, xy, content, size=48, weight="regular", italic=False,
         color=None, anchor="lt", align="left", max_width=None, line_spacing=1.15):
    """Draw `content` onto `image` in Jost at the given position.

    Args:
      image:       PIL Image (RGBA recommended). Drawn into in place.
      xy:          (x, y) anchor point. Meaning depends on `anchor`.
      content:     str. `\\n` introduces a hard line break.
      size:        Font size in pixels.
      weight:      "light" | "regular" | "bold" | "black".
      italic:      Bool.
      color:       Hex string or RGB(A) tuple. Defaults to black.
      anchor:      PIL text anchor (e.g. "lt", "mm", "rb"). See PIL docs.
      align:       "left" | "center" | "right". Only matters with multi-line.
      max_width:   If set, word-wrap to this pixel width.
      line_spacing: Multiplier on font size between lines.
    """
    font = _load_font(weight, italic, size)
    draw = ImageDraw.Draw(image)
    body = _wrap_text(content, font, max_width) if max_width else str(content).split("\n")
    rendered = "\n".join(body)
    draw.multiline_text(
        xy, rendered, font=font, fill=color or "#000000",
        anchor=anchor, align=align,
        spacing=int(size * (line_spacing - 1.0)),
    )


def text_bbox(content, size=48, weight="regular", italic=False,
              max_width=None, line_spacing=1.15):
    """Return (width, height) the text will occupy when drawn with the same args."""
    font = _load_font(weight, italic, size)
    body = _wrap_text(content, font, max_width) if max_width else str(content).split("\n")
    if not body:
        return (0, 0)
    widths = [font.getbbox(line)[2] - font.getbbox(line)[0] for line in body]
    line_h = int(size * line_spacing)
    return (max(widths), line_h * (len(body) - 1) + size)


# ---------------------------------------------------------------------------
# Namespace factory used by the sandbox.
# ---------------------------------------------------------------------------

def build_namespace(canvas_palette):
    """Construct the `art_kit` SimpleNamespace exposed to a running skill.

    `palette_color` is pre-bound to the canvas's palette so skills can call
    `art_kit.palette_color(t)` without re-passing it. Everything else takes its
    palette / rng / seed as an explicit argument.
    """
    return SimpleNamespace(
        # math
        lerp=lerp,
        clamp=clamp,
        smoothstep=smoothstep,
        remap=remap,
        # color
        hex_to_rgb=hex_to_rgb,
        rgb_to_hex=rgb_to_hex,
        mix_hex=mix_hex,
        palette_color=_palette_color_fn(canvas_palette),
        oklch_to_rgb=oklch_to_rgb,
        # composition
        rule_of_thirds=rule_of_thirds,
        vogel_spiral=vogel_spiral,
        jittered_grid=jittered_grid,
        # noise
        value_noise=value_noise,
        fbm=fbm,
        # masks
        radial_falloff=radial_falloff,
        # numpy primitives for transforms
        centered_grid=centered_grid,
        bilinear_sample=bilinear_sample,
        # voronoi
        voronoi_nearest=voronoi_nearest,
        # flow
        flow_field=flow_field,
        # l-systems
        lindenmayer=lindenmayer,
        turtle_segments=turtle_segments,
        # waves
        wave_field=wave_field,
        # attractors
        attractor_points=attractor_points,
        # text
        text=text,
        text_bbox=text_bbox,
        # stdlib re-exports for convenience
        pi=math.pi,
        tau=math.tau,
    )
