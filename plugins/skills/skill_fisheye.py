from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


def _bilinear_sample(arr, fx, fy):
    h, w, c = arr.shape
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
    c0 = arr[y1, x0]
    d = arr[y1, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c0 * (1 - wx) + d * wx) * wy


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
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.float32)
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        ccx = cx * (s - 1)
        ccy = cy * (s - 1)
        nx = (xx - ccx) / (s / 2.0)
        ny = (yy - ccy) / (s / 2.0)
        r = np.sqrt(nx * nx + ny * ny)
        # Barrel/pincushion via radial polynomial.
        scale = 1.0 + k * (r * r)
        # Apply zoom (smaller scale -> magnify).
        scale = scale / max(z, 1e-3)
        sx = ccx + nx * scale * (s / 2.0)
        sy = ccy + ny * scale * (s / 2.0)
        out = _bilinear_sample(arr, sx, sy)
        out = np.clip(out, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
