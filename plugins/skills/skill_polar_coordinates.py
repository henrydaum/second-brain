from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


def _sample(arr, fx, fy):
    h, w, _ = arr.shape
    fx = np.clip(fx, 0, w - 1)
    fy = np.clip(fy, 0, h - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (fx - x0)[..., None]
    wy = (fy - y0)[..., None]
    a = arr[y0, x0]
    b = arr[y0, x1]
    c = arr[y1, x0]
    d = arr[y1, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


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
        import math
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.float32)
        rot = math.radians(float(art_kit.clamp(rotation, 0, 360)))
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cx = cy = (s - 1) / 2.0
        if str(mode) == 'to_polar':
            # Each output pixel (xx, yy) is interpreted as (theta, radius).
            theta = (xx / max(s - 1, 1)) * 2.0 * math.pi + rot
            radius = (yy / max(s - 1, 1)) * (s / 2.0)
            sx = cx + np.cos(theta) * radius
            sy = cy + np.sin(theta) * radius
        else:  # from_polar
            dx = xx - cx
            dy = yy - cy
            r = np.sqrt(dx * dx + dy * dy)
            theta = (np.arctan2(dy, dx) - rot) % (2.0 * math.pi)
            sx = (theta / (2.0 * math.pi)) * (s - 1)
            sy = (r / (s / 2.0)) * (s - 1)
        out = _sample(arr, sx, sy)
        out = np.clip(out, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
