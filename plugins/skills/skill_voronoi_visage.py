from plugins.BaseSkill import BaseSkill, Enum, Palette

import math
import random
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class VoronoiVisageSkill(BaseSkill):
    name = 'Voronoi Visage'
    description = 'An abstract portrait: a head-shaped mask filled with Voronoi cells in palette tones. No eyes, no nose, no mouth -- the face is implied by the silhouette. The cell boundaries read as fragmentation or stained glass. Good for "portrait", "face", "head", "abstract figure", or "stained glass".'
    kind = "background"
    palette = Palette()
    density = Enum([('few', 'Few'), ('many', 'Many'), ('dense', 'Dense')], default='many')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        rng = random.Random(seed)
        n = {"few": 22, "many": 55, "dense": 110}.get(str(self.density), 55)

        cx, cy = s * 0.5, s * 0.5
        ax, ay = s * 0.30, s * 0.40

        points = []
        while len(points) < n:
            x = rng.uniform(cx - ax, cx + ax)
            y = rng.uniform(cy - ay, cy + ay)
            if ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 <= 1.0:
                points.append((x, y))

        px = np.array([p[0] for p in points], dtype=np.float32)
        py = np.array([p[1] for p in points], dtype=np.float32)

        y_idx, x_idx = np.mgrid[0:s, 0:s].astype(np.float32)
        nearest = np.zeros((s, s), dtype=np.int32)
        # Row-chunked to avoid an (s, s, n) tensor.
        chunk = 32
        for row in range(0, s, chunk):
            y_blk = y_idx[row:row + chunk]
            x_blk = x_idx[row:row + chunk]
            dx = x_blk[:, :, None] - px[None, None, :]
            dy = y_blk[:, :, None] - py[None, None, :]
            d2 = dx * dx + dy * dy
            nearest[row:row + chunk] = np.argmin(d2, axis=2).astype(np.int32)

        rgb = np.full((s, s, 3), art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        cell_colors = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.15 + 0.75 * rng.random()))
             for _ in range(n)],
            dtype=np.uint8,
        )

        mask = ((x_idx - cx) / ax) ** 2 + ((y_idx - cy) / ay) ** 2 <= 1.0
        rgb[mask] = cell_colors[nearest[mask]]

        # Cell edges: pixels whose nearest changes across neighbors.
        edges = np.zeros((s, s), dtype=bool)
        edges[1:, :] |= nearest[1:, :] != nearest[:-1, :]
        edges[:, 1:] |= nearest[:, 1:] != nearest[:, :-1]
        edge_color = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.05)), dtype=np.uint8)
        rgb[edges & mask] = edge_color

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
