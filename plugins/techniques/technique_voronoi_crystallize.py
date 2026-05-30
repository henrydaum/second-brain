from plugins.BaseTechnique import BaseTechnique, Slider

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class VoronoiCrystallizeTechnique(BaseTechnique):
    name = 'Voronoi Crystallize'
    description = 'Replace the image with a Voronoi tiling where each cell takes the color of its seed pixel — like seeing the canvas through cracked glass. Determinism comes from canvas.seed.'
    kind = "filter"

    cells = Slider(20, 1500, default=300, step=10)

    def run(self, canvas):
        n = int(self.cells)
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="uint8")
        rng = np.random.default_rng(canvas.seed)
        seeds_x = rng.integers(0, s, size=n)
        seeds_y = rng.integers(0, s, size=n)
        seed_colors = arr[seeds_y, seeds_x]
        out = np.empty_like(arr)
        chunk = max(1, 32 if n > 600 else 64)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        for y0 in range(0, s, chunk):
            y1 = min(s, y0 + chunk)
            dx = xx[y0:y1, ..., None] - seeds_x[None, None, :]
            dy = yy[y0:y1, ..., None] - seeds_y[None, None, :]
            idx = np.argmin(dx * dx + dy * dy, axis=-1)
            out[y0:y1] = seed_colors[idx]
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
