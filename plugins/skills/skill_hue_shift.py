from plugins.BaseSkill import BaseSkill

import colorsys
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class HueShiftSkill(BaseSkill):
    name = 'Hue Shift'
    description = 'Rotate every pixel\'s hue in HSV space. Quick way to remap an image off the canvas palette into a new color family without re-running the creation skill. Params: degrees (0-360, default 60).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'degrees', 'label': 'Degrees', 'min': 0, 'max': 360, 'step': 5, 'default': 60},
    ]

    def run(self, canvas, degrees=60):
        shift = float(art_kit.clamp(degrees, 0, 360)) / 360.0
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        v = mx
        delta = mx - mn
        s = np.where(mx > 0, delta / np.maximum(mx, 1e-9), 0.0)
        # Hue
        h = np.zeros_like(v)
        mask = delta > 1e-9
        rc = np.where(mask, (mx - r) / np.maximum(delta, 1e-9), 0.0)
        gc = np.where(mask, (mx - g) / np.maximum(delta, 1e-9), 0.0)
        bc = np.where(mask, (mx - b) / np.maximum(delta, 1e-9), 0.0)
        h = np.where(r == mx, bc - gc, h)
        h = np.where(g == mx, 2.0 + rc - bc, h)
        h = np.where(b == mx, 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        h = (h + shift) % 1.0
        # Back to RGB
        i = np.floor(h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i_mod = i % 6
        rr = np.choose(i_mod, [v, q, p, p, t, v])
        gg = np.choose(i_mod, [t, v, v, q, p, p])
        bb = np.choose(i_mod, [p, p, t, v, v, q])
        out = np.stack([rr, gg, bb], axis=-1)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
