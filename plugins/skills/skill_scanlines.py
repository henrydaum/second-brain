from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class ScanlinesSkill(BaseSkill):
    name = 'Scanlines'
    description = 'CRT-style horizontal scanlines: darken every Nth row toward palette.background. Tactile retro overlay; pairs with chromatic aberration for full CRT vibe. Params: intensity (0.0-1.0, default 0.45), thickness (1-6 px, default 1).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'intensity', 'label': 'Intensity', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.45},
        {'type': 'slider', 'name': 'thickness', 'label': 'Thickness', 'min': 1, 'max': 6, 'step': 1, 'default': 1},
    ]

    def run(self, canvas, intensity=0.45, thickness=1):
        amt = float(art_kit.clamp(intensity, 0.0, 1.0))
        thk = int(art_kit.clamp(thickness, 1, 12))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        s = arr.shape[0]
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        period = thk * 2
        rows = np.arange(s)
        mask = ((rows // thk) % 2 == 0).astype(np.float32) * amt
        m = mask[:, None, None]
        out = arr * (1.0 - m) + bg[None, None, :] * m
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
