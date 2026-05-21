from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class ThresholdSkill(BaseSkill):
    name = 'Threshold'
    description = 'Palette-aware two-tone threshold. Pixels above the luminance cutoff are painted palette.primary, below get palette.background. Softness adds a smooth ramp between the two so it doesn\'t look jagged. Params: level (0-255, default 128), softness (0.0-0.4, default 0.05).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'level', 'label': 'Level', 'min': 0, 'max': 255, 'step': 1, 'default': 128},
        {'type': 'slider', 'name': 'softness', 'label': 'Softness', 'min': 0.0, 'max': 0.4, 'step': 0.01, 'default': 0.05},
    ]

    def run(self, canvas, level=128, softness=0.05):
        cutoff = float(art_kit.clamp(level, 0, 255)) / 255.0
        soft = float(art_kit.clamp(softness, 0.0, 0.5))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        if soft < 1e-3:
            mask = (lum >= cutoff).astype(np.float32)
        else:
            t = np.clip((lum - (cutoff - soft)) / (2.0 * soft + 1e-6), 0.0, 1.0)
            mask = t * t * (3.0 - 2.0 * t)
        lo = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        hi = np.array(art_kit.hex_to_rgb(canvas.palette.primary), dtype=np.float32) / 255.0
        m = mask[..., None]
        out = lo * (1.0 - m) + hi * m
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
