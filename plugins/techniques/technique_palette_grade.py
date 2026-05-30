from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class PaletteGradeTechnique(BaseTechnique):
    name = 'Palette Grade'
    description = 'Map luminance through the canvas palette ramp for a cohesive tonal feel. The single most-useful post-process — call after any background technique.'
    kind = "filter"

    palette = Palette()
    mix     = Slider(0.0, 1.0, default=0.66, step=0.05)

    def run(self, canvas):
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        lo = float(np.percentile(lum, 2))
        hi = float(np.percentile(lum, 99))
        lum = np.clip((lum - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        stops = np.array([
            art_kit.hex_to_rgb(art_kit.palette_color(i / 4.0)) for i in range(5)
        ], dtype=np.float32) / 255.0
        x = lum * (len(stops) - 1)
        i = np.clip(x.astype(np.int32), 0, len(stops) - 2)
        f = (x - i)[..., None]
        mapped = stops[i] * (1.0 - f) + stops[i + 1] * f
        canvas.commit_array(arr * (1.0 - float(self.mix)) + mapped * float(self.mix))
