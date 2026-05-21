from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class WatercolorSkill(BaseSkill):
    name = 'Watercolor'
    description = 'Watercolor stylization: median-blur to soften details into pooled regions, posterize for flat washes, then re-darken at edges with palette.accent so it looks brushed. Params: pool (1-8, default 4) — wash size; edges (0.0-1.0, default 0.55) — edge ink amount.'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'pool', 'label': 'Pool', 'min': 1, 'max': 8, 'step': 1, 'default': 4},
        {'type': 'slider', 'name': 'edges', 'label': 'Edges', 'min': 0.0, 'max': 1.0, 'step': 0.05, 'default': 0.55},
    ]

    def run(self, canvas, pool=4, edges=0.55):
        p = int(art_kit.clamp(pool, 1, 12))
        e = float(art_kit.clamp(edges, 0.0, 1.0))
        img = canvas.image.convert("RGB")
        # Pool color regions.
        median_size = max(3, 2 * p + 1)
        pooled = img.filter(ImageFilter.MedianFilter(size=median_size))
        # Soften further.
        pooled = pooled.filter(ImageFilter.GaussianBlur(p * 0.5))
        # Posterize for flat washes.
        arr = np.asarray(pooled, dtype=np.float32) / 255.0
        levels = 6
        arr = np.round(arr * levels) / levels

        # Edge ink.
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
        gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        mag = np.clip(mag * 4.0, 0.0, 1.0) * e
        ink = np.array(art_kit.hex_to_rgb(canvas.palette.accent), dtype=np.float32) / 255.0
        m = mag[..., None]
        out = arr * (1.0 - m) + ink[None, None, :] * m
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
