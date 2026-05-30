from plugins.BaseTechnique import BaseTechnique, Slider, Pan

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class FisheyeTechnique(BaseTechnique):
    name = 'Fisheye'
    description = 'Lens distortion sample. Positive strength gives a fisheye bulge; negative gives pincushion (edges stretch). Pan moves the lens center.'
    kind = "filter"

    strength = Slider(-1.0, 1.0, default=0.6, step=0.1)
    zoom     = Slider(0.5, 3.0, default=2.0, step=0.1)
    cx       = Slider(0.0, 1.0, default=0.5, step=0.1)
    cy       = Slider(0.0, 1.0, default=0.5, step=0.1)
    center   = Pan(x='cx', y='cy')

    def run(self, canvas):
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        ccx = self.cx * (s - 1)
        ccy = self.cy * (s - 1)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        nx = (xx - ccx) / (s / 2.0)
        ny = (yy - ccy) / (s / 2.0)
        r = np.sqrt(nx * nx + ny * ny)
        scale = (1.0 + self.strength * (r * r)) / max(self.zoom, 1e-3)
        sx = ccx + nx * scale * (s / 2.0)
        sy = ccy + ny * scale * (s / 2.0)
        canvas.commit_array(art_kit.bilinear_sample(arr, sx, sy))
