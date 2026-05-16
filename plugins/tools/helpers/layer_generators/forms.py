"""Form layer generators. ALWAYS return RGBA with transparent background."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw, ImageFilter

from plugins.tools.helpers.color_theory import harmony

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


FORM_TYPES = (
    "primary_grid",
    "soft_horizontal_bands",
    "radial_burst",
    "organic_blobs",
    "architectural_arches",
    "branching_tree",
    "voronoi_shards",
    "silhouette_horizon",
    "mandelbrot_zoom",
    "julia_set",
    "newton_rings",
    "burning_ship",
    "multibrot",
    "tricorn",
    "phoenix",
    "sierpinski_carpet",
    "menger_3d",
    "fractalize_canvas",
)

SCALE_FACTOR = {"small": 0.55, "medium": 0.85, "large": 1.0, "filling": 1.2}
DENSITY_COUNT = {"sparse": 0.45, "moderate": 0.85, "dense": 1.4}


def render(form_type: str, size: tuple[int, int], seed: int, scale: str = "medium", density: str = "moderate", session_key: str | None = None, **_) -> Image.Image:
    fn = _DISPATCH.get(form_type, _organic_blobs)
    # Generators that need canvas state (e.g. fractalize_canvas) accept session_key;
    # the rest ignore it via **_.
    img = fn(size, seed, scale, density, session_key=session_key).convert("RGBA")
    return img


def _blank_rgba(size):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _primary_grid(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = [(220, 30, 30), (30, 60, 200), (240, 200, 30), (15, 15, 15), (250, 250, 245)]
    img = _blank_rgba(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size

    # BSP partition
    def split(rect, depth):
        x0, y0, x1, y1 = rect
        if depth == 0 or (x1 - x0) < 60 or (y1 - y0) < 60:
            return [rect]
        horiz = (x1 - x0) > (y1 - y0)
        out = []
        if horiz:
            cut = rng.randint(x0 + 30, x1 - 30)
            out += split((x0, y0, cut, y1), depth - 1)
            out += split((cut, y0, x1, y1), depth - 1)
        else:
            cut = rng.randint(y0 + 30, y1 - 30)
            out += split((x0, y0, x1, cut), depth - 1)
            out += split((x0, cut, x1, y1), depth - 1)
        return out

    margin = int(min(w, h) * 0.06)
    rects = split((margin, margin, w - margin, h - margin), 4)
    border = max(3, int(min(w, h) * 0.012))
    fill_chance = 0.45 * DENSITY_COUNT.get(density, 0.85)
    for r in rects:
        if rng.random() < fill_chance:
            color = rng.choice(pal[:3])
            draw.rectangle(r, fill=color)
        draw.rectangle(r, outline=pal[3], width=border)
    return img


def _soft_horizontal_bands(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = harmony("plasma", seed)
    img = _blank_rgba(size)
    w, h = size
    n_bands = rng.choice([2, 3, 3, 4])
    if np is None:
        draw = ImageDraw.Draw(img, "RGBA")
        cuts = sorted(rng.sample(range(int(h * 0.15), int(h * 0.85)), n_bands - 1))
        edges = [0] + cuts + [h]
        for i in range(len(edges) - 1):
            color = pal[i % len(pal)]
            draw.rectangle([0, edges[i], w, edges[i + 1]], fill=(*color, 220))
        return img.filter(ImageFilter.GaussianBlur(8))

    cuts = sorted(rng.sample(range(int(h * 0.15), int(h * 0.85)), n_bands - 1))
    edges = [0] + cuts + [h]
    arr = np.zeros((h, w, 4), dtype="float32")
    for i in range(len(edges) - 1):
        color = np.array(pal[i % len(pal)], dtype="float32")
        arr[edges[i]:edges[i + 1], :, :3] = color
        arr[edges[i]:edges[i + 1], :, 3] = 235
    out = Image.fromarray(arr.astype("uint8"), "RGBA")
    return out.filter(ImageFilter.GaussianBlur(int(min(w, h) * 0.04)))


def _radial_burst(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = harmony("inferno", seed)
    img = _blank_rgba(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    radius = int(min(w, h) * 0.55 * SCALE_FACTOR.get(scale, 0.85))
    n = max(6, int(24 * DENSITY_COUNT.get(density, 0.85)))
    base_angle = rng.uniform(0, math.tau)
    for i in range(n * 2):
        if i % 2 == 0:
            continue
        a0 = base_angle + (i / (n * 2)) * math.tau
        a1 = a0 + (math.tau / (n * 2)) * 0.9
        color = pal[rng.randint(1, 3)]
        alpha = rng.randint(140, 230)
        pts = [(cx, cy),
               (cx + radius * math.cos(a0), cy + radius * math.sin(a0)),
               (cx + radius * math.cos(a1), cy + radius * math.sin(a1))]
        draw.polygon(pts, fill=(*color, alpha))
    return img.filter(ImageFilter.GaussianBlur(2))


def _organic_blobs(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = harmony("aurora", seed)
    w, h = size
    if np is None:
        img = _blank_rgba(size)
        draw = ImageDraw.Draw(img, "RGBA")
        n = int(7 * DENSITY_COUNT.get(density, 0.85))
        for _ in range(n):
            cx = rng.randint(0, w); cy = rng.randint(0, h)
            r = int(rng.randint(60, 180) * SCALE_FACTOR.get(scale, 0.85))
            color = pal[rng.randint(1, 3)]
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(*color, 200))
        return img.filter(ImageFilter.GaussianBlur(10))
    n = max(4, int(9 * DENSITY_COUNT.get(density, 0.85)))
    radius_max = min(w, h) * 0.35 * SCALE_FACTOR.get(scale, 0.85)
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    field = np.zeros((h, w), dtype="float32")
    for _ in range(n):
        cx = rng.uniform(w * 0.1, w * 0.9)
        cy = rng.uniform(h * 0.1, h * 0.9)
        r = rng.uniform(radius_max * 0.4, radius_max)
        field += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (r * 0.6) ** 2)))
    field = np.clip(field / max(field.max(), 1e-6), 0, 1)
    alpha = np.where(field > 0.35, np.clip(field * 255, 0, 255), 0)
    # gradient color across two palette stops
    lo = np.array(pal[1], dtype="float32")
    hi = np.array(pal[3], dtype="float32")
    color = lo * (1 - field[..., None]) + hi * field[..., None]
    out = np.concatenate([color, alpha[..., None]], axis=-1).astype("uint8")
    img = Image.fromarray(out, "RGBA")
    return img.filter(ImageFilter.GaussianBlur(6))


def _architectural_arches(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = harmony("gold", seed)
    img = _blank_rgba(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    n = max(3, int(5 * DENSITY_COUNT.get(density, 0.85)))
    arch_w = w / (n + 0.5)
    arch_h = h * 0.55 * SCALE_FACTOR.get(scale, 0.85)
    base_y = h * rng.uniform(0.55, 0.78)
    color = pal[0]
    for i in range(n):
        x0 = int(i * arch_w + arch_w * 0.25)
        x1 = int(x0 + arch_w * 0.7)
        top = int(base_y - arch_h)
        draw.rectangle([x0, top + (x1 - x0) // 2, x1, int(base_y)], fill=(*color, 235))
        draw.pieslice([x0, top, x1, top + (x1 - x0)], 180, 360, fill=(*color, 235))
    # ground line
    ground_color = pal[1]
    draw.rectangle([0, int(base_y), w, h], fill=(*ground_color, 220))
    return img


def _branching_tree(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = harmony("nebula", seed)
    img = _blank_rgba(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    color = pal[3]

    def branch(x, y, angle, length, depth):
        if depth == 0 or length < 4:
            return
        x2 = x + math.cos(angle) * length
        y2 = y + math.sin(angle) * length
        width = max(1, int(depth * 1.4))
        draw.line([(x, y), (x2, y2)], fill=(*color, 235), width=width)
        spread = rng.uniform(0.35, 0.65)
        n_children = rng.choice([2, 2, 3])
        for i in range(n_children):
            new_angle = angle + rng.uniform(-spread, spread) + (i - n_children / 2) * 0.25
            branch(x2, y2, new_angle, length * rng.uniform(0.6, 0.8), depth - 1)

    start_x = w / 2 + rng.uniform(-w * 0.15, w * 0.15)
    start_y = h * 0.95
    initial_len = h * 0.22 * SCALE_FACTOR.get(scale, 0.85)
    depth = 5 if DENSITY_COUNT.get(density, 0.85) < 1 else 6
    branch(start_x, start_y, -math.pi / 2, initial_len, depth)
    return img


def _voronoi_shards(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    rng = np.random.default_rng(seed)
    rng_py = random.Random(seed)
    pal = harmony("electric", seed)
    w, h = size
    n_cells = max(8, int(28 * DENSITY_COUNT.get(density, 0.85)))
    pts = rng.random((n_cells, 2)) * np.array([w, h])
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    coords = np.stack([xx, yy], axis=-1)  # (h, w, 2)
    diffs = coords[:, :, None, :] - pts[None, None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)  # (h, w, n)
    nearest = np.argmin(dist, axis=-1)
    # border detection: second-nearest distance
    sorted_d = np.sort(dist, axis=-1)
    border_mask = (sorted_d[..., 1] - sorted_d[..., 0]) < 2.5
    fill_chance = 0.55
    cell_colors = np.zeros((n_cells, 4), dtype="float32")
    for i in range(n_cells):
        if rng_py.random() < fill_chance:
            c = pal[rng_py.randint(1, 3)]
            cell_colors[i] = (*c, 220)
        else:
            cell_colors[i] = (0, 0, 0, 0)
    arr = cell_colors[nearest]
    arr[border_mask] = (15, 15, 15, 235)
    return Image.fromarray(arr.astype("uint8"), "RGBA")


def _silhouette_horizon(size, seed, scale, density, **_):
    rng = random.Random(seed)
    pal = harmony("nocturne" if "nocturne" in dir() else "ice", seed)
    img = _blank_rgba(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    horizon = h * rng.uniform(0.45, 0.7)
    # Generate horizon line via 1D smoothed noise.
    if np is not None:
        rng_np = np.random.default_rng(seed)
        n_segments = 24
        ys = rng_np.standard_normal(n_segments)
        # smooth
        smoothed = np.convolve(ys, np.ones(5) / 5, mode="same")
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-6)
        bump = h * 0.18 * SCALE_FACTOR.get(scale, 0.85)
        xs = np.linspace(0, w, n_segments)
        horizon_pts = [(float(xs[i]), float(horizon - smoothed[i] * bump)) for i in range(n_segments)]
    else:
        horizon_pts = [(x, horizon) for x in range(0, w + 1, 30)]
    polygon = [(0, h)] + horizon_pts + [(w, h)]
    color = pal[0]
    draw.polygon(polygon, fill=(*color, 245))
    return img


# --- fractal forms ---------------------------------------------------------
# All three render the interesting structure with high alpha and let the
# uninteresting background fade to transparent so they composite over the
# background layer. Seeds pick wildly different regions / parameters so two
# users with the same form get visibly different pieces.

# Famous Mandelbrot regions worth exploring: (center_x, center_y, half-span)
_MANDEL_REGIONS = [
    (-0.75, 0.0, 1.4),       # full set
    (-0.75, 0.1, 0.05),      # seahorse valley
    (-0.088, 0.654, 0.04),   # triple spiral
    (-1.25066, 0.02012, 0.02),  # mini-brot
    (0.275, 0.006, 0.02),    # elephant valley
    (-0.745, 0.113, 0.012),  # deeper seahorse
    (-0.16, 1.04, 0.03),     # northern bulb
    (-1.7693, 0.0042, 0.005),  # antenna
]

# Hand-picked Julia c values — each produces a visually distinct shape.
_JULIA_CS = [
    complex(-0.4, 0.6),
    complex(0.285, 0.01),
    complex(-0.835, -0.2321),
    complex(-0.7269, 0.1889),
    complex(0.355, 0.355),
    complex(-0.70176, -0.3842),
    complex(-0.8, 0.156),
    complex(0.4, 0.4),
    complex(-0.74543, 0.11301),
    complex(0.37, 0.1),
]


def _mandelbrot_zoom(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("plasma", seed)
    cx, cy, base_span = rng.choice(_MANDEL_REGIONS)
    span = base_span * (2.0 - SCALE_FACTOR.get(scale, 0.85))  # higher scale → tighter zoom
    w, h = size
    aspect = w / h
    xs = np.linspace(cx - span * aspect, cx + span * aspect, w, dtype="float64")
    ys = np.linspace(cy - span, cy + span, h, dtype="float64")
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y
    Z = np.zeros_like(C)
    max_iter = int(50 + 120 * DENSITY_COUNT.get(density, 0.85))
    iters = np.full(C.shape, max_iter, dtype="int32")
    active = np.ones(C.shape, dtype="bool")
    for i in range(max_iter):
        Za = Z[active]
        Ca = C[active]
        Za = Za * Za + Ca
        mag2 = Za.real * Za.real + Za.imag * Za.imag
        escaped_local = mag2 > 4.0
        Z[active] = Za
        # Map local escape mask back into the full-canvas indices.
        active_idx = np.flatnonzero(active)
        escaped_idx = active_idx[escaped_local]
        iters.flat[escaped_idx] = i
        active.flat[escaped_idx] = False
        if not active.any():
            break
    inside = active  # never escaped — points in the set
    # Smooth coloring on the escape iterations
    t = iters.astype("float32") / max_iter
    # Boost contrast on the boundary
    t = np.power(t, 0.7)
    pal_arr = np.array(pal, dtype="float32") / 255.0
    idx = (t * (len(pal_arr) - 1))
    lo = np.clip(idx.astype("int32"), 0, len(pal_arr) - 2)
    f = idx - lo
    rgb = pal_arr[lo] * (1 - f[..., None]) + pal_arr[lo + 1] * f[..., None]
    # Alpha: set itself is opaque; escape regions fade with iteration count
    alpha = np.where(inside, 1.0, np.clip(np.power(t, 1.3), 0.0, 0.95))
    rgba = np.concatenate([rgb * 255.0, (alpha * 255.0)[..., None]], axis=-1).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


def _julia_set(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("aurora", seed)
    c = rng.choice(_JULIA_CS)
    span = 1.6 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    w, h = size
    aspect = w / h
    xs = np.linspace(-span * aspect, span * aspect, w, dtype="float64")
    ys = np.linspace(-span, span, h, dtype="float64")
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    max_iter = int(80 + 160 * DENSITY_COUNT.get(density, 0.85))
    iters = np.full(Z.shape, max_iter, dtype="int32")
    active = np.ones(Z.shape, dtype="bool")
    for i in range(max_iter):
        Z[active] = Z[active] ** 2 + c
        escaped = active & (Z.real * Z.real + Z.imag * Z.imag > 4.0)
        iters[escaped] = i
        active &= ~escaped
        if not active.any():
            break
    inside = active
    t = iters.astype("float32") / max_iter
    t = np.power(t, 0.6)
    pal_arr = np.array(pal, dtype="float32") / 255.0
    idx = t * (len(pal_arr) - 1)
    lo = np.clip(idx.astype("int32"), 0, len(pal_arr) - 2)
    f = idx - lo
    rgb = pal_arr[lo] * (1 - f[..., None]) + pal_arr[lo + 1] * f[..., None]
    alpha = np.where(inside, 1.0, np.clip(np.power(t, 1.4), 0.0, 0.92))
    rgba = np.concatenate([rgb * 255.0, (alpha * 255.0)[..., None]], axis=-1).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


def _newton_rings(size, seed, scale, density, **_):
    """Newton's method on z^3 - 1 — three roots, stained-glass cells."""
    if np is None:
        return _voronoi_shards(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("electric", seed)
    span = 1.5 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    w, h = size
    aspect = w / h
    xs = np.linspace(-span * aspect, span * aspect, w, dtype="float64")
    ys = np.linspace(-span, span, h, dtype="float64")
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    roots = np.array([1.0 + 0j,
                      complex(-0.5, math.sqrt(3) / 2),
                      complex(-0.5, -math.sqrt(3) / 2)])
    max_iter = int(20 + 30 * DENSITY_COUNT.get(density, 0.85))
    converged = np.zeros(Z.shape, dtype="int32") - 1
    iters_used = np.zeros(Z.shape, dtype="int32")
    for i in range(max_iter):
        # Newton step for z^3 - 1: z - (z^3 - 1) / (3 z^2)
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = Z - (Z ** 3 - 1) / (3 * Z ** 2)
        for k, r in enumerate(roots):
            close = (np.abs(Z - r) < 1e-3) & (converged < 0)
            converged[close] = k
            iters_used[close] = i
        if (converged >= 0).all():
            break
    # Color each root region with a different palette entry; alpha by convergence speed.
    root_colors = np.array([pal[1], pal[2], pal[3]], dtype="float32") / 255.0
    safe_root = np.clip(converged, 0, 2)
    rgb = root_colors[safe_root]
    # Fast convergence → low iters_used → high alpha (clean cell interior).
    # Slow convergence (on boundary) → high iters_used → also high alpha (the rings).
    # Non-converged → transparent.
    t = iters_used.astype("float32") / max(1, max_iter)
    alpha = np.where(converged >= 0, np.clip(1.0 - t * 0.55, 0.4, 1.0), 0.0)
    rgba = np.concatenate([rgb * 255.0, (alpha * 255.0)[..., None]], axis=-1).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


# --- additional escape-time variants ---------------------------------------

def _escape_render(size, max_iter, iters, inside, pal, alpha_curve=1.3, t_curve=0.7):
    """Shared coloring tail for escape-time fractals."""
    t = iters.astype("float32") / max(1, max_iter)
    t = np.power(t, t_curve)
    pal_arr = np.array(pal, dtype="float32") / 255.0
    idx = t * (len(pal_arr) - 1)
    lo = np.clip(idx.astype("int32"), 0, len(pal_arr) - 2)
    f = idx - lo
    rgb = pal_arr[lo] * (1 - f[..., None]) + pal_arr[lo + 1] * f[..., None]
    alpha = np.where(inside, 1.0, np.clip(np.power(t, alpha_curve), 0.0, 0.92))
    rgba = np.concatenate([rgb * 255.0, (alpha * 255.0)[..., None]], axis=-1).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


def _escape_grid(size, span, center=0+0j):
    w, h = size
    aspect = w / h
    xs = np.linspace(center.real - span * aspect, center.real + span * aspect, w, dtype="float64")
    ys = np.linspace(center.imag - span, center.imag + span, h, dtype="float64")
    X, Y = np.meshgrid(xs, ys)
    return X + 1j * Y


def _burning_ship(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("inferno", seed)
    span = 1.6 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    # The burning ship's iconic region is around (-1.75, -0.05)
    center = complex(rng.uniform(-1.8, -1.6), rng.uniform(-0.1, 0.05))
    C = _escape_grid(size, span * 0.3, center)
    Z = np.zeros_like(C)
    max_iter = int(50 + 100 * DENSITY_COUNT.get(density, 0.85))
    iters = np.full(C.shape, max_iter, dtype="int32")
    active = np.ones(C.shape, dtype="bool")
    for i in range(max_iter):
        Za = Z[active]
        Za = (np.abs(Za.real) + 1j * np.abs(Za.imag)) ** 2 + C[active]
        mag2 = Za.real * Za.real + Za.imag * Za.imag
        escaped_local = mag2 > 4.0
        Z[active] = Za
        idx = np.flatnonzero(active)[escaped_local]
        iters.flat[idx] = i
        active.flat[idx] = False
        if not active.any():
            break
    return _escape_render(size, max_iter, iters, active, pal)


def _multibrot(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("nebula", seed)
    n = rng.choice([3, 4, 5, 6, 7])  # power
    span = 1.6 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    C = _escape_grid(size, span)
    Z = np.zeros_like(C)
    max_iter = int(50 + 100 * DENSITY_COUNT.get(density, 0.85))
    iters = np.full(C.shape, max_iter, dtype="int32")
    active = np.ones(C.shape, dtype="bool")
    for i in range(max_iter):
        Za = Z[active]
        Za = Za ** n + C[active]
        mag2 = Za.real * Za.real + Za.imag * Za.imag
        escaped_local = mag2 > 4.0
        Z[active] = Za
        idx = np.flatnonzero(active)[escaped_local]
        iters.flat[idx] = i
        active.flat[idx] = False
        if not active.any():
            break
    return _escape_render(size, max_iter, iters, active, pal)


def _tricorn(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    pal = harmony("ice", seed)
    span = 1.8 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    C = _escape_grid(size, span)
    Z = np.zeros_like(C)
    max_iter = int(50 + 100 * DENSITY_COUNT.get(density, 0.85))
    iters = np.full(C.shape, max_iter, dtype="int32")
    active = np.ones(C.shape, dtype="bool")
    for i in range(max_iter):
        Za = Z[active]
        # z = conj(z)^2 + c
        Za_conj = Za.real - 1j * Za.imag
        Za = Za_conj * Za_conj + C[active]
        mag2 = Za.real * Za.real + Za.imag * Za.imag
        escaped_local = mag2 > 4.0
        Z[active] = Za
        idx = np.flatnonzero(active)[escaped_local]
        iters.flat[idx] = i
        active.flat[idx] = False
        if not active.any():
            break
    return _escape_render(size, max_iter, iters, active, pal)


def _phoenix(size, seed, scale, density, **_):
    if np is None:
        return _organic_blobs(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("plasma", seed)
    span = 1.6 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    # Phoenix uses a fixed c and a complex p parameter
    c = complex(rng.uniform(0.55, 0.58), 0)
    p = complex(rng.uniform(-0.55, -0.45), 0)
    # Map screen to z0, not c — phoenix is rendered Julia-style
    Z = _escape_grid(size, span)
    Z_prev = np.zeros_like(Z)
    max_iter = int(50 + 100 * DENSITY_COUNT.get(density, 0.85))
    iters = np.full(Z.shape, max_iter, dtype="int32")
    active = np.ones(Z.shape, dtype="bool")
    for i in range(max_iter):
        Za = Z[active]
        Zp = Z_prev[active]
        Z_new = Za * Za + c + p * Zp
        mag2 = Z_new.real * Z_new.real + Z_new.imag * Z_new.imag
        escaped_local = mag2 > 4.0
        Z_prev[active] = Za
        Z[active] = Z_new
        idx = np.flatnonzero(active)[escaped_local]
        iters.flat[idx] = i
        active.flat[idx] = False
        if not active.any():
            break
    return _escape_render(size, max_iter, iters, active, pal)


# --- Menger / Sierpinski --------------------------------------------------

def _sierpinski_carpet(size, seed, scale, density, **_):
    """2D Menger: recursive subdivision, remove center cell at each level."""
    rng = random.Random(seed)
    pal = harmony("gold", seed)
    img = _blank_rgba(size)
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = size
    side = int(min(w, h) * 0.85 * SCALE_FACTOR.get(scale, 0.85))
    x0 = (w - side) // 2 + rng.randint(-w // 20, w // 20)
    y0 = (h - side) // 2 + rng.randint(-h // 20, h // 20)
    depth = {"sparse": 3, "moderate": 4, "dense": 5}.get(density, 4)
    fill = pal[2]

    def carpet(x, y, s, d):
        if d == 0:
            draw.rectangle([x, y, x + s, y + s], fill=(*fill, 240))
            return
        cell = s / 3
        for ix in range(3):
            for iy in range(3):
                if ix == 1 and iy == 1:
                    continue  # center cell is removed
                carpet(x + ix * cell, y + iy * cell, cell, d - 1)

    carpet(x0, y0, side, depth)
    return img


def _menger_3d(size, seed, scale, density, **_):
    """3D Menger sponge via SDF raymarching at half-res, then upscaled."""
    if np is None:
        return _sierpinski_carpet(size, seed, scale, density)
    rng = random.Random(seed)
    pal = harmony("aurora", seed)
    # Render at quarter res for speed, upscale at end. ~80x60 = 4800 rays.
    rw, rh = max(120, size[0] // 4), max(90, size[1] // 4)
    aspect = rw / rh
    iterations = {"sparse": 2, "moderate": 3, "dense": 4}.get(density, 3)

    # Random orbit camera
    angle = rng.uniform(0, math.tau)
    elev = rng.uniform(-0.4, 0.6)
    dist = 3.2 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    cam = np.array([math.cos(angle) * math.cos(elev) * dist,
                    math.sin(elev) * dist,
                    math.sin(angle) * math.cos(elev) * dist], dtype="float32")
    target = np.array([0.0, 0.0, 0.0], dtype="float32")
    up = np.array([0.0, 1.0, 0.0], dtype="float32")
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)

    # Build rays per pixel
    ys, xs = np.mgrid[0:rh, 0:rw].astype("float32")
    u = (xs / (rw - 1) - 0.5) * 2 * aspect
    v = -(ys / (rh - 1) - 0.5) * 2
    dirs = (fwd[None, None, :]
            + right[None, None, :] * u[..., None]
            + up2[None, None, :] * v[..., None])
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    origins = np.broadcast_to(cam, dirs.shape).copy()

    t = np.zeros((rh, rw), dtype="float32")
    hit = np.zeros((rh, rw), dtype="bool")
    iters_hit = np.zeros((rh, rw), dtype="int32")
    MAX_STEPS = 64
    MAX_T = 30.0
    HIT_EPS = 0.005

    for step in range(MAX_STEPS):
        active = (~hit) & (t < MAX_T)
        if not active.any():
            break
        p = origins[active] + dirs[active] * t[active][..., None]
        d = _menger_sdf(p, iterations)
        t_new = t[active] + d
        new_hits = d < HIT_EPS
        t[active] = t_new
        # Mark hits in the active region
        active_idx = np.flatnonzero(active)
        hit.flat[active_idx[new_hits]] = True
        iters_hit.flat[active_idx[new_hits]] = step

    # Color by depth + step count (cheap shading)
    depth_norm = np.clip(t / MAX_T, 0, 1)
    shade = 1.0 - depth_norm * 0.7
    pal_arr = np.array(pal, dtype="float32") / 255.0
    # Use step count to pick palette index (different lighting bands)
    step_t = np.clip(iters_hit.astype("float32") / MAX_STEPS, 0, 1)
    idx = step_t * (len(pal_arr) - 1)
    lo = np.clip(idx.astype("int32"), 0, len(pal_arr) - 2)
    f = idx - lo
    rgb = pal_arr[lo] * (1 - f[..., None]) + pal_arr[lo + 1] * f[..., None]
    rgb = rgb * shade[..., None]
    alpha = np.where(hit, 0.97, 0.0)
    rgba = np.concatenate([rgb * 255, (alpha * 255)[..., None]], axis=-1).astype("uint8")
    low = Image.fromarray(rgba, "RGBA")
    return low.resize(size, Image.Resampling.BICUBIC)


def _menger_sdf(p, iterations):
    """Signed distance to a Menger sponge centered at origin, half-extent 1."""
    # Distance to unit cube
    q = np.abs(p) - 1.0
    d = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)
    s = 1.0
    pp = p.copy()
    for _ in range(iterations):
        a = np.mod(pp * s, 2.0) - 1.0
        s *= 3.0
        r = np.abs(1.0 - 3.0 * np.abs(a))
        da = np.maximum(r[..., 0], r[..., 1])
        db = np.maximum(r[..., 1], r[..., 2])
        dc = np.maximum(r[..., 0], r[..., 2])
        c = (np.minimum(np.minimum(da, db), dc) - 1.0) / s
        d = np.maximum(d, c)
    return d


# --- Image-as-fractal: orbit trap with the current canvas as source --------

def _fractalize_canvas(size, seed, scale, density, session_key=None, **_):
    """Re-render the current canvas through a Mandelbrot/Julia escape lens.
    Each pixel's color comes from sampling the source image at the orbit's
    final position. Result: the canvas content scattered through fractal filaments."""
    if np is None or not session_key:
        return _organic_blobs(size, seed, scale, density)
    from plugins.tools.helpers import layered_canvas as lc
    # Prefer the most recent composite; fall back to background, then default texture.
    state = lc.get_state(session_key)
    src_path = state.get("composite_path")
    if not src_path or not Image.open(src_path):
        bg = (state.get("layers") or {}).get("background") or {}
        src_path = bg.get("path")
    if not src_path:
        # No canvas yet — degrade to organic_blobs so we don't crash
        return _organic_blobs(size, seed, scale, density)
    try:
        src = Image.open(src_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    except Exception:
        return _organic_blobs(size, seed, scale, density)
    src_arr = np.asarray(src, dtype="uint8")
    rng = random.Random(seed)

    # Two modes: Mandelbrot (varying c) or Julia (fixed c, varying z0)
    mode = rng.choice(["mandelbrot", "julia"])
    span = 1.7 * (2.0 - SCALE_FACTOR.get(scale, 0.85))
    max_iter = int(40 + 60 * DENSITY_COUNT.get(density, 0.85))
    w, h = size

    if mode == "mandelbrot":
        C = _escape_grid(size, span)
        Z = np.zeros_like(C)
    else:
        c_choice = rng.choice(_JULIA_CS)
        Z = _escape_grid(size, span)
        C = np.full(Z.shape, c_choice)

    iters = np.full(Z.shape, max_iter, dtype="int32")
    final_Z = np.zeros_like(Z)
    active = np.ones(Z.shape, dtype="bool")
    for i in range(max_iter):
        Za = Z[active]
        Ca = C[active]
        Za = Za * Za + Ca
        mag2 = Za.real * Za.real + Za.imag * Za.imag
        escaped_local = mag2 > 4.0
        Z[active] = Za
        # Capture final Z for escaped pixels
        idx = np.flatnonzero(active)[escaped_local]
        iters.flat[idx] = i
        final_Z.flat[idx] = Za[escaped_local]
        active.flat[idx] = False
        if not active.any():
            break

    # For points still in the set, sample at current z position
    in_set_idx = np.flatnonzero(active)
    final_Z.flat[in_set_idx] = Z.flat[in_set_idx]

    # Map final Z to image coordinates (with some wrap so distant escapes still sample meaningfully)
    # final_Z is unbounded for escaped points; tanh to [-1,1] then scale to image
    fx = np.tanh(final_Z.real * 0.4) * 0.5 + 0.5
    fy = np.tanh(final_Z.imag * 0.4) * 0.5 + 0.5
    sx = np.clip((fx * (w - 1)).astype("int32"), 0, w - 1)
    sy = np.clip((fy * (h - 1)).astype("int32"), 0, h - 1)
    sampled = src_arr[sy, sx]  # (h, w, 3)

    # Alpha: in-set pixels are opaque; escape regions fade out
    t = iters.astype("float32") / max(1, max_iter)
    in_set = ~np.zeros_like(active)
    in_set = active  # boolean mask
    alpha = np.where(in_set, 1.0, np.clip(np.power(t, 1.2), 0.0, 0.92))
    rgba = np.concatenate([sampled, (alpha * 255)[..., None]], axis=-1).astype("uint8")
    return Image.fromarray(rgba, "RGBA")


_DISPATCH = {
    "primary_grid": _primary_grid,
    "soft_horizontal_bands": _soft_horizontal_bands,
    "radial_burst": _radial_burst,
    "organic_blobs": _organic_blobs,
    "architectural_arches": _architectural_arches,
    "branching_tree": _branching_tree,
    "voronoi_shards": _voronoi_shards,
    "silhouette_horizon": _silhouette_horizon,
    "mandelbrot_zoom": _mandelbrot_zoom,
    "julia_set": _julia_set,
    "newton_rings": _newton_rings,
    "burning_ship": _burning_ship,
    "multibrot": _multibrot,
    "tricorn": _tricorn,
    "phoenix": _phoenix,
    "sierpinski_carpet": _sierpinski_carpet,
    "menger_3d": _menger_3d,
    "fractalize_canvas": _fractalize_canvas,
}
