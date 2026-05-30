from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Bool, Palette

import math
import random
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class ImaginaryMapsTechnique(BaseTechnique):
    name = "Imaginary Maps"
    description = "Fantasy cartography. Voronoi cells become countries; an fbm height field carves out seas and lakes; thin rivers trace downhill paths from the highlands. Three styles: antique parchment, blueprint, and minimal."
    kind = "background"
    palette = Palette()
    country_count = Slider(6, 30, default=14, step=1)
    sea_level = Slider(0.25, 0.65, default=0.42, step=0.02)
    style = Enum([("antique", "Antique"), ("blueprint", "Blueprint"), ("minimal", "Minimal")], default="antique")
    show_rivers = Bool(default=True)

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed
        rng = random.Random(seed)
        style = str(self.style)

        if style == "antique":
            paper_rgb = (0.93, 0.86, 0.68)
            sea_rgb = (0.66, 0.78, 0.78)
            border_rgb = (0.20, 0.13, 0.08)
            river_rgb = (0.30, 0.45, 0.62)
        elif style == "blueprint":
            paper_rgb = (0.08, 0.18, 0.32)
            sea_rgb = (0.04, 0.10, 0.20)
            border_rgb = (0.85, 0.92, 1.0)
            river_rgb = (0.65, 0.82, 0.98)
        else:  # minimal
            paper_rgb = (0.97, 0.97, 0.96)
            sea_rgb = (0.82, 0.88, 0.92)
            border_rgb = (0.15, 0.15, 0.15)
            river_rgb = (0.35, 0.55, 0.75)

        n = int(self.country_count)
        # Country seed points, jittered grid for even coverage.
        cols = max(2, int(math.sqrt(n)))
        rows = max(2, (n + cols - 1) // cols)
        pts = art_kit.jittered_grid(rng, cols, rows, jitter=0.85)[:n]
        seeds_x = np.array([p[0] * s for p in pts], dtype=np.float32)
        seeds_y = np.array([p[1] * s for p in pts], dtype=np.float32)

        # Vectorized nearest-seed map.
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        # Distort coords with fbm so borders wiggle.
        warp_freq = 0.008
        wx = art_kit.fbm_grid(seed, xx * warp_freq, yy * warp_freq, octaves=3).astype(np.float32)
        wy = art_kit.fbm_grid(seed + 9, xx * warp_freq, yy * warp_freq, octaves=3).astype(np.float32)
        warp_amp = s * 0.04
        xw = xx + (wx - 0.5) * warp_amp
        yw = yy + (wy - 0.5) * warp_amp

        chunk = 48
        idx = np.zeros((s, s), dtype=np.int32)
        for y0 in range(0, s, chunk):
            y1 = min(s, y0 + chunk)
            dx = xw[y0:y1, ..., None] - seeds_x[None, None, :]
            dy = yw[y0:y1, ..., None] - seeds_y[None, None, :]
            idx[y0:y1] = np.argmin(dx * dx + dy * dy, axis=-1)

        # Height field decides land vs sea.
        h = art_kit.fbm_grid(seed + 17, xx * 0.005, yy * 0.005, octaves=5).astype(np.float32)
        h = (h - h.min()) / max(float(h.max() - h.min()), 1e-6)
        sea = h < float(self.sea_level)

        # Per-country tint, slight variation from the palette.
        tints = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            t = rng.random()
            r, g, b = art_kit.hex_to_rgb(art_kit.palette_color(0.5 + 0.4 * t))
            mixed = (
                0.55 * np.array(paper_rgb) + 0.45 * np.array((r / 255, g / 255, b / 255))
            )
            tints[i] = mixed

        out = tints[idx]
        out[sea] = np.array(sea_rgb, dtype=np.float32)

        # Country borders: pixels where neighbor's idx differs, only on land.
        diff_x = np.zeros((s, s), dtype=bool)
        diff_y = np.zeros((s, s), dtype=bool)
        diff_x[:, 1:] = idx[:, 1:] != idx[:, :-1]
        diff_y[1:, :] = idx[1:, :] != idx[:-1, :]
        border = (diff_x | diff_y) & (~sea)
        # Coastlines: land pixels next to sea.
        coast_x = np.zeros((s, s), dtype=bool)
        coast_y = np.zeros((s, s), dtype=bool)
        coast_x[:, 1:] = sea[:, 1:] != sea[:, :-1]
        coast_y[1:, :] = sea[1:, :] != sea[:-1, :]
        coast = coast_x | coast_y
        out[border] = np.array(border_rgb, dtype=np.float32) * 0.6 + out[border] * 0.4
        out[coast] = np.array(border_rgb, dtype=np.float32)

        if bool(self.show_rivers):
            # Pick a handful of source points in the highlands, walk downhill.
            n_rivers = max(3, n // 3)
            for _ in range(n_rivers):
                # Find a random highland pixel.
                for _try in range(30):
                    rx = rng.randrange(s)
                    ry = rng.randrange(s)
                    if not sea[ry, rx] and h[ry, rx] > 0.75:
                        break
                else:
                    continue
                self._walk_river(out, h, sea, rx, ry, river_rgb, s)

        canvas.commit_array(out)

    def _walk_river(self, out, h, sea, x, y, color, s):
        col = np.array(color, dtype=np.float32)
        for _ in range(600):
            if not (1 <= x < s - 1 and 1 <= y < s - 1):
                return
            if sea[y, x]:
                return
            out[y, x] = col
            # Pick steepest downhill neighbor (8-connected).
            best = h[y, x]
            bx, by = x, y
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nh = h[y + dy, x + dx]
                    if nh < best:
                        best = nh
                        bx, by = x + dx, y + dy
            if (bx, by) == (x, y):
                return  # local minimum
            x, y = bx, by
