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
)

SCALE_FACTOR = {"small": 0.55, "medium": 0.85, "large": 1.0, "filling": 1.2}
DENSITY_COUNT = {"sparse": 0.45, "moderate": 0.85, "dense": 1.4}


def render(form_type: str, size: tuple[int, int], seed: int, scale: str = "medium", density: str = "moderate", **_) -> Image.Image:
    fn = _DISPATCH.get(form_type, _organic_blobs)
    img = fn(size, seed, scale, density).convert("RGBA")
    return img


def _blank_rgba(size):
    return Image.new("RGBA", size, (0, 0, 0, 0))


def _primary_grid(size, seed, scale, density):
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


def _soft_horizontal_bands(size, seed, scale, density):
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


def _radial_burst(size, seed, scale, density):
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


def _organic_blobs(size, seed, scale, density):
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


def _architectural_arches(size, seed, scale, density):
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


def _branching_tree(size, seed, scale, density):
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


def _voronoi_shards(size, seed, scale, density):
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


def _silhouette_horizon(size, seed, scale, density):
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


_DISPATCH = {
    "primary_grid": _primary_grid,
    "soft_horizontal_bands": _soft_horizontal_bands,
    "radial_burst": _radial_burst,
    "organic_blobs": _organic_blobs,
    "architectural_arches": _architectural_arches,
    "branching_tree": _branching_tree,
    "voronoi_shards": _voronoi_shards,
    "silhouette_horizon": _silhouette_horizon,
}
