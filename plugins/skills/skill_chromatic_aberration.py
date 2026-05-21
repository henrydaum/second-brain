from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class ChromaticAberrationSkill(BaseSkill):
    name = 'Chromatic Aberration'
    description = 'Lens-fringe color separation — split R/G/B and offset each. Radial mode pushes channels outward from the center (camera-lens look). Uniform mode offsets all channels along a fixed direction (CRT/print misregistration look). Params: strength (0-30 px, default 6), radial bool, angle (0-360, only used when radial=False).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': 0, 'max': 30, 'step': 1, 'default': 6},
        {'type': 'bool', 'name': 'radial', 'label': 'Radial', 'default': True},
        {'type': 'slider', 'name': 'angle', 'label': 'Angle', 'min': 0, 'max': 360, 'step': 5, 'default': 0},
    ]

    def run(self, canvas, strength=6, radial=True, angle=0):
        import math
        amt = float(art_kit.clamp(strength, 0, 60))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        s = canvas.size
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        if bool(radial):
            yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
            cx = cy = (s - 1) / 2.0
            nx = (xx - cx) / max(cx, 1.0)
            ny = (yy - cy) / max(cy, 1.0)
            length = np.sqrt(nx * nx + ny * ny) + 1e-6
            ux = nx / length
            uy = ny / length
            # R channel pushed outward, B pulled inward; G stays put.
            sxr = xx + ux * amt
            syr = yy + uy * amt
            sxb = xx - ux * amt
            syb = yy - uy * amt
            r_new = _sample(r, sxr, syr)
            b_new = _sample(b, sxb, syb)
            out = np.stack([r_new, g, b_new], axis=-1)
        else:
            a = math.radians(float(art_kit.clamp(angle, 0, 360)))
            dx = int(round(math.cos(a) * amt))
            dy = int(round(math.sin(a) * amt))
            r_new = np.roll(r, shift=(dy, dx), axis=(0, 1))
            b_new = np.roll(b, shift=(-dy, -dx), axis=(0, 1))
            out = np.stack([r_new, g, b_new], axis=-1)
        out = np.clip(out, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))


def _sample(plane, fx, fy):
    h, w = plane.shape
    fx = np.clip(fx, 0, w - 1)
    fy = np.clip(fy, 0, h - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = fx - x0
    wy = fy - y0
    a = plane[y0, x0]
    b = plane[y0, x1]
    c = plane[y1, x0]
    d = plane[y1, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
