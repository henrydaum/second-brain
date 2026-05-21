from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class FilmGrainSkill(BaseSkill):
    name = 'Film Grain'
    description = 'Deterministic per-pixel noise overlay seeded from canvas.seed. Adds tactile texture; great over flat palette grades. Params: intensity (0.0-0.3, default 0.07), monochrome (bool, default True).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'intensity', 'label': 'Intensity', 'min': 0.0, 'max': 0.4, 'step': 0.005, 'default': 0.07},
        {'type': 'bool', 'name': 'monochrome', 'label': 'Monochrome', 'default': True},
    ]

    def run(self, canvas, intensity=0.07, monochrome=True):
        img = canvas.image.convert("RGB")
        s = canvas.size
        intensity = float(art_kit.clamp(intensity, 0.0, 0.4))
        rng = np.random.default_rng(canvas.seed)
        arr = np.asarray(img).astype(np.float32) / 255.0
        if bool(monochrome):
            noise = rng.standard_normal((s, s, 1)).astype(np.float32) * intensity
        else:
            noise = rng.standard_normal((s, s, 3)).astype(np.float32) * intensity
        out = np.clip(arr + noise, 0.0, 1.0)
        out = (out * 255.0).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
