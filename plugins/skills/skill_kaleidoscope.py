from plugins.BaseSkill import BaseSkill

import math
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


class KaleidoscopeSkill(BaseSkill):
    name = 'Kaleidoscope'
    description = 'Fold N angular wedges around the center; the result is N-fold rotational symmetry from a single source slice. Classic toy. Params: segments (3-24, default 8), rotation (0-360, default 0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'segments', 'label': 'Segments', 'min': 3, 'max': 24, 'step': 1, 'default': 8},
        {'type': 'slider', 'name': 'rotation', 'label': 'Rotation', 'min': 0, 'max': 360, 'step': 5, 'default': 0},
    ]

    def run(self, canvas, segments=8, rotation=0):
        n = int(art_kit.clamp(segments, 2, 36))
        rot = math.radians(float(art_kit.clamp(rotation, 0, 360)))
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.float32)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cx = cy = (s - 1) / 2.0
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx) - rot
        wedge = 2.0 * math.pi / n
        # Fold theta into [0, wedge], mirror every other wedge for that mirror-symmetry feel.
        t = np.mod(theta, 2.0 * wedge)
        t = np.where(t > wedge, 2.0 * wedge - t, t)
        sx = cx + np.cos(t + rot) * r
        sy = cy + np.sin(t + rot) * r
        out = _sample(arr, sx, sy)
        out = np.clip(out, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
