from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class ThresholdTechnique(BaseTechnique):
    name = 'Threshold'
    description = 'Palette-aware two-tone threshold. Pixels above the luminance cutoff are painted palette.primary, below get palette.background. Softness adds a smooth ramp between the two.'
    kind = "filter"

    palette  = Palette()
    level    = Slider(0, 255, default=128, step=1)
    softness = Slider(0.0, 0.4, default=0.05, step=0.01)

    def run(self, canvas):
        cutoff = float(self.level) / 255.0
        soft = float(self.softness)
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        if soft < 1e-3:
            mask = (lum >= cutoff).astype(np.float32)
        else:
            t = np.clip((lum - (cutoff - soft)) / (2.0 * soft + 1e-6), 0.0, 1.0)
            mask = t * t * (3.0 - 2.0 * t)
        lo = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        hi = np.array(art_kit.hex_to_rgb(canvas.palette.primary), dtype=np.float32) / 255.0
        m = mask[..., None]
        canvas.commit_array(lo * (1.0 - m) + hi * m)
