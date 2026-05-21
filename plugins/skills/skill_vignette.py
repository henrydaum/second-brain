from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class VignetteSkill(BaseSkill):
    name = 'Vignette'
    description = 'Radial darken tinted with palette.background. Pulls the eye toward the center; pairs well with palette_grade. Params: strength (0.0-1.0, default 0.6), softness (0.05-0.95, default 0.55).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.6},
        {'type': 'slider', 'name': 'softness', 'label': 'Softness', 'min': 0.05, 'max': 0.95, 'step': 0.05, 'default': 0.55},
    ]

    def run(self, canvas, strength=0.6, softness=0.55):
        img = canvas.image.convert("RGB")
        s = canvas.size
        strength = float(art_kit.clamp(strength, 0.0, 1.0))
        softness = float(art_kit.clamp(softness, 0.05, 0.95))

        arr = np.asarray(img).astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cx = cy = s / 2.0
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (s * 0.5 * np.sqrt(2.0))
        edge0 = 1.0 - softness
        t = np.clip((d - edge0) / max(1e-6, 1.0 - edge0), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        falloff = (1.0 - smooth * strength)[..., None]
        tint = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        out = arr * falloff + tint * (1.0 - falloff)
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
