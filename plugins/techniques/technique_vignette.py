from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class VignetteTechnique(BaseTechnique):
    name = 'Vignette'
    description = 'Radial darken tinted with palette.background. Pulls the eye toward the center; pairs well with palette_grade.'
    kind = "filter"

    palette  = Palette()
    strength = Slider(0.0, 1.0, default=0.6, step=0.05)
    softness = Slider(0.05, 0.95, default=0.55, step=0.05)

    def run(self, canvas):
        arr = canvas.image_array(mode="RGB", dtype="float")
        s = canvas.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cx = cy = s / 2.0
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (s * 0.5 * np.sqrt(2.0))
        edge0 = 1.0 - float(self.softness)
        t = np.clip((d - edge0) / max(1e-6, 1.0 - edge0), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        falloff = (1.0 - smooth * float(self.strength))[..., None]
        tint = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        canvas.commit_array(arr * falloff + tint * (1.0 - falloff))
