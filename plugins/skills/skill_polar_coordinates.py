from plugins.BaseSkill import BaseSkill

import math
import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class PolarCoordinatesSkill(BaseSkill):
    name = 'Polar Coordinates'
    description = 'Remap rectangular ↔ polar coordinates. "to_polar" wraps the image as a circle (tunnel/lollipop feel). "from_polar" unrolls it into a strip (panoramic effect on radial images). Param: mode enum, rotation (0-360, default 0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'enum', 'name': 'mode', 'label': 'Mode', 'options': [
            {'value': 'to_polar', 'label': 'To Polar'},
            {'value': 'from_polar', 'label': 'From Polar'},
        ], 'default': 'to_polar'},
        {'type': 'slider', 'name': 'rotation', 'label': 'Rotation', 'min': 0, 'max': 360, 'step': 5, 'default': 0},
    ]

    def run(self, canvas, mode='to_polar', rotation=0):
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        rot = math.radians(float(art_kit.clamp(rotation, 0, 360)))
        xx, yy, nx, ny = art_kit.centered_grid(s)
        cx = (s - 1) / 2.0
        if str(mode) == 'to_polar':
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
