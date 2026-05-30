"""Curated helpers injected into the technique sandbox as `art_kit`.

Techniques cannot import this module — the sandbox import gate blocks it. Instead,
the sandbox entry calls `build_namespace(canvas)` and exposes the returned
SimpleNamespace under the name `art_kit` in the technique's exec namespace.

Everything here is pure: no I/O, no global mutation. All randomness must take a
caller-supplied `random.Random` so techniques stay deterministic from `canvas.seed`.
"""

from __future__ import annotations

import colorsys
import functools
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
    """Linear interpolation: returns ``a`` at t=0, ``b`` at t=1, beyond either
    bound if ``t`` is outside [0, 1]. Works on scalars or numpy arrays."""
    return a + (b - a) * t


def clamp(x, lo=0.0, hi=1.0):
    """Clamp ``x`` to the [``lo``, ``hi``] range. Scalar only — for numpy
    arrays use ``numpy.clip``."""
    return lo if x < lo else hi if x > hi else x


def smoothstep(t, edge0=0.0, edge1=1.0):
    """Smooth Hermite ramp: 0 below ``edge0``, 1 above ``edge1``, with a
    smoothed S-curve in between. Use instead of ``lerp`` when you want
    transitions that ease in and out rather than snap linearly."""
    x = clamp((t - edge0) / ((edge1 - edge0) or 1e-9), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def remap(x, in_lo, in_hi, out_lo, out_hi):
    """Linearly re-range ``x`` from [in_lo, in_hi] to [out_lo, out_hi]. Does
    not clamp — values outside the input range extrapolate. Wrap with
    ``clamp`` if you need a bounded result."""
    t = (x - in_lo) / ((in_hi - in_lo) or 1e-9)
    return out_lo + (out_hi - out_lo) * t


# ---------------------------------------------------------------------------
# Color helpers.
# ---------------------------------------------------------------------------

def hex_to_rgb(h):
    """Parse a ``"#rrggbb"`` (or ``"rrggbb"``) string to a ``(r, g, b)``
    tuple of ints in [0, 255]. The inverse of ``rgb_to_hex``."""
    h = str(h).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb):
    """Format an ``(r, g, b)`` tuple as a ``"#rrggbb"`` string. Channels are
    clamped to [0, 255] and rounded, so float intermediates are safe."""
    r, g, b = (int(round(clamp(c, 0, 255))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def mix_hex(a, b, t):
    """Linear RGB mix of two hex colors: t=0 returns ``a``, t=1 returns ``b``.
    For palette-aware gradients, prefer ``palette_color(t)`` — it interpolates
    along the canvas's luminance-sorted ramp instead of two arbitrary hexes."""
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return rgb_to_hex((lerp(ar, br, t), lerp(ag, bg, t), lerp(ab, bb, t)))


def with_alpha(color, alpha):
    """Return an RGBA tuple for a hex/RGB/RGBA color with replaced alpha."""
    if isinstance(color, str):
        rgb = hex_to_rgb(color)
    else:
        rgb = tuple(color[:3])
    a = float(alpha)
    return (*rgb, int(clamp(a * 255 if a <= 1 else a, 0, 255)))


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
    ramp_rgb = [hex_to_rgb(c) for c in ramp]
    n = len(ramp)

    def palette_color(t, value=1.0):
        """Sample the canvas palette's luminance-sorted ramp at ``t`` ∈ [0, 1].

        Returns a hex string. ``t=0`` is the darkest palette slot, ``t=1`` the
        brightest, with linear RGB interpolation between adjacent slots.
        ``value`` ∈ [0, 1.5] scales brightness multiplicatively (use sparingly
        — ``palette_color(t)`` alone keeps the palette pure).

        **This is the preferred color source for every fill and stroke.**
        Hardcoded hexes or RGB tuples break palette swapping; always reach for
        ``palette_color`` (or ``canvas.palette.<slot>``) instead.
        """
        t = clamp(float(t), 0.0, 1.0)
        if n == 1:
            return ramp[0]
        x = t * (n - 1)
        i = int(x)
        if i >= n - 1:
            return ramp[-1]
        f = x - i
        r1, g1, b1 = ramp_rgb[i]
        r2, g2, b2 = ramp_rgb[i + 1]
        v = clamp(float(value), 0.0, 1.5)
        return rgb_to_hex((lerp(r1, r2, f) * v, lerp(g1, g2, f) * v, lerp(b1, b2, f) * v))

    return palette_color


def oklch_to_rgb(l, c, h):
    """Convert OKLch (lightness, chroma, hue) to an sRGB ``(r, g, b)`` tuple
    in [0, 255]. Hue ``h`` is in turns (0..1), not degrees or radians.

    OKLch is perceptually uniform — equal steps in ``l`` look like equal
    steps to the eye — so it's the right space for procedural color ramps
    that need to read evenly. For palette-bound gradients prefer
    ``palette_color(t)`` instead; reach for ``oklch_to_rgb`` when you need
    a color genuinely outside the palette (rare; ask the user first).
    """
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
    """Centers of a ``cols × rows`` grid in [0, 1]², each jittered within
    its own cell. Returns a list of ``(x, y)`` tuples in row-major order.

    ``jitter=0`` gives a perfect lattice; ``jitter=1`` lets points wander
    across cell boundaries. ``rng`` must be a seeded ``random.Random`` so
    palette replays produce identical points.

    The standard way to seed Voronoi tilings, scattered marks, or any
    "evenly distributed but not gridlike" point set.
    """
    out = []
    cw, ch = 1.0 / max(1, cols), 1.0 / max(1, rows)
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.5) * cw + (rng.random() - 0.5) * cw * jitter
            cy = (r + 0.5) * ch + (rng.random() - 0.5) * ch * jitter
            out.append((cx, cy))
    return out


def regular_polygon(cx, cy, radius, sides, rotation=0.0, y_scale=1.0):
    """Return points for a rotated regular polygon, optionally squashed on y."""
    sides = max(3, int(sides))
    return [
        (cx + math.cos(rotation + math.tau * i / sides) * radius,
         cy + math.sin(rotation + math.tau * i / sides) * radius * y_scale)
        for i in range(sides)
    ]


# ---------------------------------------------------------------------------
# Tiny 3D renderer.
# ---------------------------------------------------------------------------

def _v3(p):
    return (float(p[0]), float(p[1]), float(p[2]))


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(v):
    d = math.sqrt(_dot(v, v)) or 1.0
    return (v[0] / d, v[1] / d, v[2] / d)


def mesh(vertices, faces, color=None, colors=None):
    """Return a tiny mesh object for `render_3d`.

    `vertices` are `(x, y, z)` triples. `faces` are index lists like
    `(0, 1, 2)` or `(0, 1, 2, 3)`. Pass `color=canvas.palette.primary`
    for one palette-bound material, or `colors=[...]` for per-face colors.
    """
    return SimpleNamespace(
        vertices=[_v3(v) for v in vertices],
        faces=[tuple(int(i) for i in f) for f in faces],
        color=color,
        colors=list(colors) if colors is not None else None,
    )


def cube_mesh(size=1.0, center=(0.0, 0.0, 0.0), color=None):
    """Return a cube mesh centered at `center`, sized for `render_3d` scenes."""
    s = float(size) / 2.0
    cx, cy, cz = _v3(center)
    v = [(cx + x * s, cy + y * s, cz + z * s) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    return mesh(v, [(0, 2, 6, 4), (1, 5, 7, 3), (0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6)], color=color)


def _shade_color(color, shade):
    if isinstance(color, str):
        r, g, b = hex_to_rgb(color)
        return rgb_to_hex((r * shade, g * shade, b * shade))
    r, g, b = color[:3]
    a = color[3] if len(color) > 3 else 255
    return (int(r * shade), int(g * shade), int(b * shade), a)


def _mesh_get(m, name, default=None):
    return m.get(name, default) if isinstance(m, dict) else getattr(m, name, default)


def render_3d(image, meshes, camera=(2.8, 2.0, 3.2), target=(0, 0, 0), fov=42,
              light=(-0.4, -0.6, 1.0), fill=None, outline=None, cull=True,
              ambient=0.35, palette_color=None):
    """Project simple 3D meshes onto a PIL image with filled polygons.

    This is a dependency-free baby renderer: camera projection, face sorting,
    optional backface culling, and Lambert-ish light. It is good for cubes,
    low-poly objects, heightfields, and extruded shapes. It is not OpenGL:
    no shaders, textures, clipping, or real z-buffer.
    """
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    eye, look = _v3(camera), _v3(target)
    forward = _norm(_sub(look, eye))
    right = _cross(forward, (0.0, 1.0, 0.0))
    right = (1.0, 0.0, 0.0) if _dot(right, right) < 1e-9 else _norm(right)
    up = _cross(right, forward)
    focal = min(w, h) / (2.0 * math.tan(math.radians(float(fov)) / 2.0))
    light = _norm(light)
    items = []

    def project(p):
        rel = _sub(p, eye)
        x, y, z = _dot(rel, right), _dot(rel, up), _dot(rel, forward)
        if z <= 0.02:
            return None
        return (w * 0.5 + focal * x / z, h * 0.5 - focal * y / z, z)

    if _mesh_get(meshes, "vertices") is not None:
        meshes = [meshes]
    for m in meshes:
        verts, faces = _mesh_get(m, "vertices", []), _mesh_get(m, "faces", [])
        colors, base = _mesh_get(m, "colors"), _mesh_get(m, "color", fill)
        for i, face in enumerate(faces):
            pts = [verts[j] for j in face]
            normal = _norm(_cross(_sub(pts[1], pts[0]), _sub(pts[2], pts[0]))) if len(pts) >= 3 else (0, 0, 1)
            center = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
            if cull and _dot(normal, _sub(eye, center)) <= 0:
                continue
            projected = [project(p) for p in pts]
            if any(p is None for p in projected):
                continue
            shade = clamp(float(ambient) + (1.0 - float(ambient)) * max(0.0, _dot(normal, light)), 0.0, 1.0)
            color = colors[i] if colors and i < len(colors) else base
            color = (palette_color(0.2 + 0.7 * shade) if palette_color else "#cccccc") if color is None else _shade_color(color, shade)
            items.append((sum(p[2] for p in projected) / len(projected), [(p[0], p[1]) for p in projected], color))
    for _, poly, color in sorted(items, reverse=True):
        draw.polygon(poly, fill=color, outline=outline)
    return image


def _render_3d_fn(canvas_palette):
    palette_color = _palette_color_fn(canvas_palette)

    def draw(image, meshes, **kwargs):
        kwargs.setdefault("palette_color", palette_color)
        return render_3d(image, meshes, **kwargs)

    return draw


# ---------------------------------------------------------------------------
# Noise.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=65536)
def _hash01(rng_seed, ix, iy):
    # Deterministic value at lattice point (ix, iy). Uses a string seed for
    # Python 3.13 compatibility (tuple seeds were removed). Stable across runs.
    # Memoized: lattice corners are shared by 4 neighboring cells and across
    # fbm octaves, so the same (seed, ix, iy) is hit many times in one technique
    # execution. Cache lives in the subprocess and dies on exit.
    return random.Random(f"{rng_seed}:{int(ix)}:{int(iy)}").random()


def _hash01_np(seed, ix, iy):
    """Vectorized deterministic [0,1) hash for noise grids.

    Different sequence than the scalar ``_hash01`` — used only by ``*_grid``
    helpers. Implements a SplitMix64-style bit mixer over a packed
    ``(seed, ix, iy)`` triple, then takes the high 53 bits as a double.
    """
    import numpy as _np
    with _np.errstate(over="ignore"):
        s = _np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
        ix_u = _np.asarray(ix, dtype=_np.int64).astype(_np.uint64)
        iy_u = _np.asarray(iy, dtype=_np.int64).astype(_np.uint64)
        h = (ix_u * _np.uint64(0x9E3779B97F4A7C15)) ^ \
            (iy_u * _np.uint64(0xBF58476D1CE4E5B9)) ^ \
            (s    * _np.uint64(0x94D049BB133111EB))
        h = (h ^ (h >> _np.uint64(30))) * _np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> _np.uint64(27))) * _np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> _np.uint64(31))
        return (h >> _np.uint64(11)).astype(_np.float64) * (1.0 / (1 << 53))


def value_noise(seed, x, y):
    """Smooth 2D value noise in [0, 1] at a single point.

    Inputs are continuous floats; an integer step in ``x`` or ``y`` crosses
    one lattice cell. Cheap drop-in alternative to perlin for organic
    texture, terrain, clouds.

    **Use the scalar form for sparse or sequential sampling** — particle
    advection, 1D sweeps, one-off lookups inside a drawing loop.
    **To fill a whole lattice, use ``value_noise_grid`` instead** — it's
    typically 30-80× faster on a 160² grid. Note the two paths use
    different hash sequences, so they produce different fields at the
    same ``(seed, x, y)``.
    """
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    sx, sy = smoothstep(fx), smoothstep(fy)
    n00 = _hash01(seed, x0, y0)
    n10 = _hash01(seed, x0 + 1, y0)
    n01 = _hash01(seed, x0, y0 + 1)
    n11 = _hash01(seed, x0 + 1, y0 + 1)
    return lerp(lerp(n00, n10, sx), lerp(n01, n11, sx), sy)


def fbm(seed, x, y, octaves=4, lacunarity=2.0, gain=0.5):
    """Fractional Brownian motion over ``value_noise`` at a single point.

    Sums ``octaves`` layers of value_noise at progressively higher frequency
    (``lacunarity``) and lower amplitude (``gain``). Returns ~[0, 1].
    Output is richer-looking than raw ``value_noise`` — natural choice for
    clouds, terrain heightfields, smoke, organic texture.

    **Use the scalar form for sparse/sequential sampling** (flow-field
    particle advection, 1D angle sweeps, fbm interleaved with drawing).
    **To fill a 2D lattice, use ``fbm_grid`` instead** — ~30-80× faster on
    a 160² grid. The two paths use different hash sequences, so outputs at
    the same seed differ visually.
    """
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


def value_noise_grid(seed, xx, yy):
    """Vectorized 2D value noise over numpy coordinate arrays.

    ``xx`` and ``yy`` are float arrays of identical shape; returns a float64
    array of the same shape in [0, 1]. Typical setup:

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        field = art_kit.value_noise_grid(seed, xx * freq, yy * freq)

    **Preferred over scalar ``value_noise`` whenever you need a whole grid
    of samples.** Uses a SplitMix64-style hash (different sequence from the
    scalar path), so the same ``(seed, x, y)`` produces a different value
    here than in ``value_noise``.
    """
    import numpy as _np
    xx = _np.asarray(xx, dtype=_np.float64)
    yy = _np.asarray(yy, dtype=_np.float64)
    x0 = _np.floor(xx).astype(_np.int64)
    y0 = _np.floor(yy).astype(_np.int64)
    fx = xx - x0
    fy = yy - y0
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    n00 = _hash01_np(seed, x0,     y0)
    n10 = _hash01_np(seed, x0 + 1, y0)
    n01 = _hash01_np(seed, x0,     y0 + 1)
    n11 = _hash01_np(seed, x0 + 1, y0 + 1)
    top = n00 * (1.0 - sx) + n10 * sx
    bot = n01 * (1.0 - sx) + n11 * sx
    return top * (1.0 - sy) + bot * sy


def fbm_grid(seed, xx, yy, octaves=4, lacunarity=2.0, gain=0.5):
    """Vectorized fbm over numpy coordinate arrays. Returns a float64 array
    matching the shape of ``xx`` / ``yy``, values in ~[0, 1].

    **Preferred for any noise field big enough to want at once** — terrain
    heightmaps, cloud layers, nebula warps, ridged turbulence. Replaces the
    nested-Python pattern::

        for r in range(N):                                # SLOW
            for c in range(N):
                h[r, c] = art_kit.fbm(seed, c * freq, r * freq, octaves=k)

    with one call::

        yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)    # FAST
        h = art_kit.fbm_grid(seed, xx * freq, yy * freq, octaves=k)

    Roughly 30-80× faster on a 160² grid with 5 octaves. Different hash
    sequence than the scalar ``fbm``, so outputs at the same seed differ
    visually — pick one and stay with it inside a technique.
    """
    import numpy as _np
    xx = _np.asarray(xx, dtype=_np.float64)
    yy = _np.asarray(yy, dtype=_np.float64)
    total = _np.zeros_like(xx)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(int(octaves)):
        total += value_noise_grid(seed, xx * freq, yy * freq) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / (norm or 1.0)


# ---------------------------------------------------------------------------
# Falloffs / masks. Returned as functions (call per pixel) to avoid numpy in
# the namespace surface — techniques can still use numpy directly if they import it.
# ---------------------------------------------------------------------------

def radial_falloff(w, h, cx=None, cy=None):
    """Return a closure ``f(x, y)`` giving 1 at the center and 0 at the
    farthest canvas corner, with a linear falloff in between.

    ``cx`` / ``cy`` default to the canvas center. The closure is pure
    Python — use it for vignette masks, sun glows, lens flares.

    **For a full per-pixel mask, build it with numpy instead** (one
    ``np.hypot`` over a coordinate grid is ~100× faster than calling this
    closure in a loop). Reach for the closure form when you only need
    sparse samples, or when you're already inside a per-element loop.
    """
    cx = (w / 2.0) if cx is None else float(cx)
    cy = (h / 2.0) if cy is None else float(cy)
    max_d = math.hypot(max(cx, w - cx), max(cy, h - cy)) or 1.0

    def f(x, y):
        return 1.0 - clamp(math.hypot(x - cx, y - cy) / max_d, 0.0, 1.0)

    return f


# ---------------------------------------------------------------------------
# Numpy primitives for filter techniques.
#
# These let lens / warp / glitch transforms skip the most-repeated boilerplate:
# building a centered coordinate grid and resampling an array at fractional
# coordinates. Numpy is imported lazily so first-party techniques that don't use
# these (background techniques, PIL-only transforms) pay no import cost.
# ---------------------------------------------------------------------------

def centered_grid(size):
    """Return (xx, yy, nx, ny) for an ``size x size`` canvas.

    xx, yy: float32 pixel coordinates (0..size-1).
    nx, ny: normalized to [-1, +1] from the canvas center — what every radial
            distortion (fisheye, CA, vignette) actually wants.

    Techniques that want a custom center can compute their own normalization off
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
    chromatic aberration, and any other warp filter.
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
    """Return a closure ``f(x, y) -> (index, distance)`` of the nearest
    seed point. Useful for sparse sampling (which seed owns this random
    spot?) or for caller-driven decisions like shading edges when the
    second-nearest distance is small.

    **For a full per-pixel Voronoi map, do it inline with numpy** —
    broadcast ``(xx - sx[None, None, :])²+(yy - sy[None, None, :])²`` and
    take ``argmin`` along the seed axis. That's tens to hundreds of times
    faster than calling this closure per pixel. Reach for ``voronoi_nearest``
    only for sparse / caller-driven lookups.
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
    """Return a closure ``f(x, y) -> angle_radians`` driven by ``fbm`` noise.

    The standard scaffold for streamline, wind, hair, and ribbon techniques:
    seed particles, repeatedly step them by ``(cos(angle), sin(angle))``,
    draw each path with a palette color. ``scale`` controls swirl tightness
    — smaller is broader and smoother.

    Built on the scalar ``fbm``, which fits its access pattern (each
    particle samples a different point sequentially — you can't batch it
    into a grid). For per-pixel flow fields, sample ``fbm_grid`` once and
    take ``arctan2`` of gradients.
    """

    def f(x, y):
        return fbm(seed, x * scale, y * scale, octaves=octaves) * math.tau

    return f


# ---------------------------------------------------------------------------
# L-systems.
# ---------------------------------------------------------------------------

def lindenmayer(axiom, rules, iterations):
    """Iteratively rewrite ``axiom`` using a ``{char: replacement}`` dict.

    Classic L-system string production: each iteration replaces every
    matching character with its rule's right-hand side; non-matching chars
    pass through. Pipe the result into ``turtle_segments`` to get drawable
    line segments — together they cover trees, lightning, dragon curves,
    Koch snowflakes, Sierpinski variants, plant skeletons.

    Keep iterations modest (``≤ 6-9`` for branching grammars) — strings
    grow exponentially and turtle interpretation slows accordingly.
    """
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
    """Return a closure ``f(x, y) -> summed wave intensity``.

    ``sources`` is a list of ``(cx, cy, wavelength, phase)``. Each source
    contributes ``sin(2π * (dist/wavelength + phase))``; the closure returns
    their sum. The combined value lies in roughly [-N, +N] for N sources —
    normalize with ``/N`` or pass through ``tanh`` / ``smoothstep`` before
    mapping to color.

    Good for ripples, moiré, interference patterns, water surfaces. For a
    full per-pixel field, call the closure inside a vectorized numpy loop
    or rewrite the math inline — pure-Python evaluation at 1024² is too
    slow.
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
    """Generate ``n`` ``(x, y)`` points from a 2D strange attractor, each
    coordinate normalized to ~[-1, 1].

    Supported ``name`` values: ``"de_jong"``, ``"clifford"``. If ``params``
    is ``None``, the ``(a, b, c, d)`` constants are drawn from ``seed`` in
    a known-interesting range — different seeds produce visually distinct
    attractors. A 100-step burn-in is done before any points are emitted.

    Wispy, ribbon-like point clouds; the standard recipe is to accumulate
    the points into a density buffer (numpy histogram2d) and color-map by
    density. For dense outputs you typically want ``n`` in the hundreds of
    thousands.
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
    """Return the ``(width, height)`` in pixels that ``content`` will occupy
    when drawn with the same arguments via ``text``.

    Pass the same ``size``, ``weight``, ``italic``, ``max_width``, and
    ``line_spacing`` you'll use in the actual ``text`` call. Use the
    measurement to align labels, reserve space for chips/badges, or pick
    a font size that fits a target box (binary search ``size`` against a
    target width).
    """
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
    """Construct the `art_kit` SimpleNamespace exposed to a running technique.

    `palette_color` is pre-bound to the canvas's palette so techniques can call
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
        with_alpha=with_alpha,
        palette_color=_palette_color_fn(canvas_palette),
        oklch_to_rgb=oklch_to_rgb,
        # composition
        rule_of_thirds=rule_of_thirds,
        vogel_spiral=vogel_spiral,
        jittered_grid=jittered_grid,
        regular_polygon=regular_polygon,
        # tiny 3D renderer
        mesh=mesh,
        cube_mesh=cube_mesh,
        render_3d=_render_3d_fn(canvas_palette),
        # noise
        value_noise=value_noise,
        fbm=fbm,
        value_noise_grid=value_noise_grid,
        fbm_grid=fbm_grid,
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
