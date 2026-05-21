from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class DotScreenSkill(BaseSkill):
    name = 'Dot Screen'
    description = 'Overlay a regular dot pattern at canvas resolution, with dot alpha controlled by local luminance — keeps original colors but adds a printed-screen texture. Lighter than full halftone. Params: cell_size (4-30 px, default 10), strength (0.0-1.0, default 0.5).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'cell_size', 'label': 'Cell Size', 'min': 4, 'max': 30, 'step': 1, 'default': 10},
        {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.5},
    ]

    def run(self, canvas, cell_size=10, strength=0.5):
        c = int(art_kit.clamp(cell_size, 2, 60))
        amt = float(art_kit.clamp(strength, 0.0, 1.0))
        img = canvas.image.convert("RGB")
        s = canvas.size
        arr = np.asarray(img, dtype=np.float32) / 255.0
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722

        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        # Distance to nearest grid intersection.
        gx = xx % c - (c / 2.0)
        gy = yy % c - (c / 2.0)
        d = np.sqrt(gx * gx + gy * gy)
        # Dot radius from luminance (dark -> bigger).
        radius = (1.0 - lum) * (c * 0.45)
        mask = np.clip((radius - d) / 1.5, 0.0, 1.0) * amt
        m = mask[..., None]
        dot_color = np.array(art_kit.hex_to_rgb(canvas.palette.accent), dtype=np.float32) / 255.0
        out = arr * (1.0 - m) + dot_color[None, None, :] * m
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
