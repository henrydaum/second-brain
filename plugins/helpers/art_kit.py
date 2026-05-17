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
from types import SimpleNamespace

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
    # Deterministic value at lattice point (ix, iy) using a single Random.
    return random.Random((rng_seed, int(ix), int(iy))).random()


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
        # stdlib re-exports for convenience
        pi=math.pi,
        tau=math.tau,
    )
