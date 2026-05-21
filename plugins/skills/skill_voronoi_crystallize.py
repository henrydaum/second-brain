from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class VoronoiCrystallizeSkill(BaseSkill):
    name = 'Voronoi Crystallize'
    description = 'Replace the image with a Voronoi tiling where each cell takes the color of its seed pixel — like seeing the canvas through cracked glass. Determinism comes from canvas.seed. Param: cells (20-1500, default 300).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'cells', 'label': 'Cells', 'min': 20, 'max': 1500, 'step': 10, 'default': 300},
    ]

    def run(self, canvas, cells=300):
        n = int(art_kit.clamp(cells, 10, 4000))
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.uint8)
        rng = np.random.default_rng(canvas.seed)
        seeds_x = rng.integers(0, s, size=n)
        seeds_y = rng.integers(0, s, size=n)
        seed_colors = arr[seeds_y, seeds_x]  # (n, 3)

        # Chunked nearest-seed assignment to keep memory bounded.
        out = np.empty_like(arr)
        chunk = max(1, 32 if n > 600 else 64)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        for y0 in range(0, s, chunk):
            y1 = min(s, y0 + chunk)
            px = xx[y0:y1]
            py = yy[y0:y1]
            dx = px[..., None] - seeds_x[None, None, :]
            dy = py[..., None] - seeds_y[None, None, :]
            d2 = dx * dx + dy * dy
            idx = np.argmin(d2, axis=-1)
            out[y0:y1] = seed_colors[idx]
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
