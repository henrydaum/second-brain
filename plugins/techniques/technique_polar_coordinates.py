from plugins.BaseTechnique import BaseTechnique, Slider, Enum

import math
import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class PolarCoordinatesTechnique(BaseTechnique):
    name = 'Polar Coordinates'
    description = 'Remap rectangular ↔ polar coordinates. "to_polar" wraps the image as a circle; "from_polar" unrolls it into a strip.'
    kind = "filter"

    mode     = Enum([('to_polar', 'To Polar'), ('from_polar', 'From Polar')], default='to_polar')
    rotation = Slider(0, 360, default=0, step=5)

    def run(self, canvas):
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        rot = math.radians(float(self.rotation))
        xx, yy, _, _ = art_kit.centered_grid(s)
        cx = (s - 1) / 2.0
        if self.mode == 'to_polar':
            theta = (xx / max(s - 1, 1)) * 2.0 * math.pi + rot
            radius = (yy / max(s - 1, 1)) * (s / 2.0)
            sx = cx + np.cos(theta) * radius
            sy = cx + np.sin(theta) * radius
        else:
            dx = xx - cx
            dy = yy - cx
            r = np.sqrt(dx * dx + dy * dy)
            theta = (np.arctan2(dy, dx) - rot) % (2.0 * math.pi)
            sx = (theta / (2.0 * math.pi)) * (s - 1)
            sy = (r / (s / 2.0)) * (s - 1)
        canvas.commit_array(art_kit.bilinear_sample(arr, sx, sy))
