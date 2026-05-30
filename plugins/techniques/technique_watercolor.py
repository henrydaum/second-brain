from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np
from PIL import ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class WatercolorTechnique(BaseTechnique):
    name = 'Watercolor'
    description = 'Watercolor stylization: median-blur to soften details into pooled regions, posterize for flat washes, then re-darken at edges with palette.accent.'
    kind = "filter"

    palette = Palette()
    pool    = Slider(1, 8, default=4, step=1)
    edges   = Slider(0.0, 1.0, default=0.55, step=0.05)

    def run(self, canvas):
        p = int(self.pool)
        e = float(self.edges)
        img = canvas.image.convert("RGB")
        median_size = max(3, 2 * p + 1)
        pooled = img.filter(ImageFilter.MedianFilter(size=median_size))
        pooled = pooled.filter(ImageFilter.GaussianBlur(p * 0.5))
        arr = np.asarray(pooled, dtype=np.float32) / 255.0
        levels = 6
        arr = np.round(arr * levels) / levels
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
        gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        mag = np.clip(np.sqrt(gx * gx + gy * gy) * 4.0, 0.0, 1.0) * e
        ink = np.array(art_kit.hex_to_rgb(canvas.palette.accent), dtype=np.float32) / 255.0
        m = mag[..., None]
        canvas.commit_array(arr * (1.0 - m) + ink[None, None, :] * m)
