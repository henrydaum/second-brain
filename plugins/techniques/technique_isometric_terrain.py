from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

def _heightmap(seed, N, mode):
    h = np.zeros((N, N), dtype=np.float32)
    if mode == "hills":
        for r in range(N):
            for c in range(N):
                h[r, c] = art_kit.fbm(seed, c * 0.10, r * 0.10, octaves=4)
    elif mode == "islands":
        for r in range(N):
            for c in range(N):
                h[r, c] = art_kit.fbm(seed, c * 0.09, r * 0.09, octaves=4)
        # Distance-from-center falloff so the field reads as an island.
        ys, xs = np.mgrid[0:N, 0:N].astype(np.float32)
        rad = np.sqrt((xs - N / 2.0) ** 2 + (ys - N / 2.0) ** 2) / (N / 2.0)
        h = np.clip(h - rad * 0.55, 0.0, 1.0)
        # Below sea level -> snap to a low flat plateau (the water).
        sea = 0.18
        h = np.where(h < sea, sea * 0.6, h)
    elif mode == "city":
        for r in range(N):
            for c in range(N):
                h[r, c] = art_kit.fbm(seed, c * 0.18, r * 0.18, octaves=3)
        h = np.round(h * 6.0) / 6.0  # quantize to 6 levels for blocky look
    elif mode == "mesa":
        for r in range(N):
            for c in range(N):
                h[r, c] = art_kit.fbm(seed, c * 0.06, r * 0.06, octaves=4)
        h = np.round(h * 4.0) / 4.0  # only 4 plateau levels
    else:  # village
        for r in range(N):
            for c in range(N):
                h[r, c] = art_kit.fbm(seed, c * 0.20, r * 0.20, octaves=3)
        h = np.round(h * 5.0) / 5.0
    h = h - float(h.min())
    h /= max(float(h.max()), 1e-6)
    return h


class IsometricTerrainTechnique(BaseTechnique):
    name = 'Isometric Terrain'
    description = 'Fake 3D from pure 2D primitives: a small fbm heightmap rasterized cell by cell in back-to-front order, each cell drawn as a footprint hex base plus a top and two shaded side panels. Three palette shades per cell -- top (brightest), left (mid), right (darkest) -- give consistent diagonal lighting without any real depth buffer. Presets: rolling hills, islands above sea level, blocky city, stepped mesa, and a village of quantized towers. Good for "terrain", "voxel", "isometric", "village", "island", "hill", or any tilted-grid worldbuilding.'
    kind = "background"
    palette = Palette()
    scene = Enum([('hills', 'Rolling Hills'), ('islands', 'Islands'), ('city', 'City Blocks'), ('mesa', 'Stepped Mesa'), ('village', 'Tower Village')], default='hills')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        self.scene = str(self.scene)

        N = 36
        h = _heightmap(seed, N, self.scene)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        # Isometric projection: screen_x = (c - r) * cos30, screen_y = (c + r) * sin30 - height.
        # Pick cell_w so the whole grid fits horizontally with margin.
        margin = s * 0.05
        span = s - 2 * margin
        cos30 = math.cos(math.radians(30.0))
        sin30 = math.sin(math.radians(30.0))
        cell_w = span / ((N - 1) * 2 * cos30 + 1e-6)
        cell_h = cell_w  # square footprint cells
        h_scale = s * 0.32

        # Origin: place the diamond center at canvas center. Heights raise the
        # top faces upward (by up to h_scale) and side panels extend down by
        # height, so this stays balanced for both flat and tall scenes.
        ox = s / 2.0
        oy = s * 0.7

        # Sort cells back-to-front: greater r+c -> drawn later.
        cells = [(r, c) for r in range(N) for c in range(N)]
        cells.sort(key=lambda rc: rc[0] + rc[1])

        for r, c in cells:
            hh = h[r, c]
            height = hh * h_scale
            bx = (c - (N - 1) / 2.0) - (r - (N - 1) / 2.0)
            by = (c - (N - 1) / 2.0) + (r - (N - 1) / 2.0)
            sx = ox + bx * cell_w * cos30
            sy = oy + by * cell_h * sin30 - height

            # Three palette shades. Higher cells get higher ramp positions.
            top_t = art_kit.clamp(0.40 + 0.55 * hh, 0.05, 0.98)
            side_l_t = art_kit.clamp(top_t - 0.18, 0.05, 0.95)
            side_r_t = art_kit.clamp(top_t - 0.30, 0.05, 0.95)
            top_color = art_kit.palette_color(top_t)
            left_color = art_kit.palette_color(side_l_t)
            right_color = art_kit.palette_color(side_r_t)

            # Top diamond (rhombus): four points around (sx, sy).
            hx = cell_w * cos30
            hy = cell_h * sin30
            top = [
                (sx,         sy - hy),
                (sx + hx,    sy),
                (sx,         sy + hy),
                (sx - hx,    sy),
            ]
            # Side panels go from the lower-left and lower-right edges down by `height`.
            left_panel = [
                (sx - hx,        sy),
                (sx,             sy + hy),
                (sx,             sy + hy + height),
                (sx - hx,        sy + height),
            ]
            right_panel = [
                (sx,             sy + hy),
                (sx + hx,        sy),
                (sx + hx,        sy + height),
                (sx,             sy + hy + height),
            ]
            if height > 0.5:
                draw.polygon(left_panel, fill=left_color)
                draw.polygon(right_panel, fill=right_color)
            draw.polygon(top, fill=top_color)

        canvas.commit(img)
