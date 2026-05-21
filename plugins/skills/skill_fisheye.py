from plugins.BaseSkill import BaseSkill

import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class FisheyeSkill(BaseSkill):
    name = 'Fisheye'
    description = 'Lens distortion sample. Positive strength gives a fisheye bulge (center magnifies, edges compress); negative gives pincushion (edges stretch). Pan moves the lens center. Params: strength (-1.0 to 1.0, default 0.6), zoom (0.5-2.0, default 1.0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': -1.0, 'max': 1.0, 'step': 0.05, 'default': 0.6},
        {'type': 'slider', 'name': 'zoom', 'label': 'Zoom', 'min': 0.5, 'max': 2.0, 'step': 0.05, 'default': 1.0},
        {'type': 'pan', 'name': 'center', 'label': 'Center', 'x_param': 'cx', 'y_param': 'cy', 'step': 0.05, 'x_default': 0.5, 'y_default': 0.5},
    ]

    def run(self, canvas, strength=0.6, zoom=1.0, cx=0.5, cy=0.5):
        k = float(art_kit.clamp(strength, -1.5, 1.5))
        z = float(art_kit.clamp(zoom, 0.3, 3.0))
        cx = float(art_kit.clamp(cx, 0.0, 1.0))
        cy = float(art_kit.clamp(cy, 0.0, 1.0))
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        ccx = cx * (s - 1)
        ccy = cy * (s - 1)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        nx = (xx - ccx) / (s / 2.0)
        ny = (yy - ccy) / (s / 2.0)
        r = np.sqrt(nx * nx + ny * ny)
        scale = (1.0 + k * (r * r)) / max(z, 1e-3)
        sx = ccx + nx * scale * (s / 2.0)
        sy = ccy + ny * scale * (s / 2.0)
        canvas.commit_array(art_kit.bilinear_sample(arr, sx, sy))
