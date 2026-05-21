from plugins.BaseSkill import BaseSkill

import math
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class MotionBlurSkill(BaseSkill):
    name = 'Motion Blur'
    description = 'Directional blur along an angle — fakes camera shake or fast motion. Params: length (3-60, default 18), angle in degrees (0-360, default 0).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'length', 'label': 'Length', 'min': 3, 'max': 60, 'step': 1, 'default': 18},
        {'type': 'slider', 'name': 'angle', 'label': 'Angle', 'min': 0, 'max': 360, 'step': 5, 'default': 0},
    ]

    def run(self, canvas, length=18, angle=0):
        n = int(art_kit.clamp(length, 1, 80))
        a = math.radians(float(art_kit.clamp(angle, 0, 360)))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)
        h, w, _ = arr.shape
        dx = math.cos(a)
        dy = math.sin(a)
        acc = np.zeros_like(arr)
        weight = 0.0
        for i in range(n):
            t = i - (n - 1) / 2.0
            sx = int(round(t * dx))
            sy = int(round(t * dy))
            shifted = np.roll(arr, shift=(sy, sx), axis=(0, 1))
            acc += shifted
            weight += 1.0
        out = np.clip(acc / max(weight, 1.0), 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
