from plugins.BaseTechnique import BaseTechnique, Slider

import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class LowPolyTechnique(BaseTechnique):
    name = 'Low Poly'
    description = 'Facet the canvas into a triangular low-poly mosaic: a jittered triangular lattice is laid over the image and each triangle is flood-filled with the source color sampled at its centroid, giving that crystalline geometric-poster look. More cells = finer facets; jitter breaks the regular grid. Good for "low poly", "triangulation", "faceted", "geometric poster", "crystalline", or a polygonal repaint.'
    kind = "filter"

    cells  = Slider(10, 60, default=30, step=1)
    jitter = Slider(0.0, 0.9, default=0.45, step=0.05)

    def run(self, canvas):
        s = int(canvas.size)
        cols = int(self.cells)
        jit = float(self.jitter)
        rng = np.random.default_rng(int(canvas.seed))

        src = canvas.image_array(mode="RGB", dtype="uint8")
        step = s / cols
        rows = int(s / step) + 2

        # Build a jittered point lattice; clamp the border so we cover edges.
        pts = {}
        for j in range(rows + 1):
            for i in range(cols + 2):
                x = (i - 0.5) * step
                y = j * step
                if 0 < x < s and 0 < y < s:
                    x += (rng.random() - 0.5) * step * jit
                    y += (rng.random() - 0.5) * step * jit
                pts[(i, j)] = (x, y)

        img = Image.new("RGBA", (s, s), tuple(int(c) for c in src[0, 0]) + (255,))
        draw = ImageDraw.Draw(img, "RGBA")

        def sample(tri):
            cx = sum(p[0] for p in tri) / 3.0
            cy = sum(p[1] for p in tri) / 3.0
            ix = min(max(int(cx), 0), s - 1)
            iy = min(max(int(cy), 0), s - 1)
            return tuple(int(c) for c in src[iy, ix]) + (255,)

        for j in range(rows):
            for i in range(cols + 1):
                a = pts[(i, j)]
                b = pts[(i + 1, j)]
                c = pts[(i, j + 1)]
                d = pts[(i + 1, j + 1)]
                for tri in ((a, b, d), (a, d, c)):
                    draw.polygon(tri, fill=sample(tri))

        canvas.commit(img)
