from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class DepthFogTechnique(BaseTechnique):
    name = 'Depth Fog'
    description = 'Blend the current canvas toward the palette background by a depth mask: linear top-fade, bottom-fade, or radial edge-fade.'
    kind = "filter"

    palette   = Palette()
    direction = Enum([
        ('top',    'Top Fade'),
        ('bottom', 'Bottom Fade'),
        ('radial', 'Radial Edge'),
    ], default='radial')
    strength  = Slider(0.0, 1.0, default=0.55, step=0.05)

    def run(self, canvas):
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float") * 255.0
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)
        ys, xs = np.mgrid[0:s, 0:s].astype(np.float32)
        if self.direction == "top":
            mask = 1.0 - (ys / max(s - 1, 1))
        elif self.direction == "bottom":
            mask = ys / max(s - 1, 1)
        else:
            cx = cy = s / 2.0
            d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            d_max = np.sqrt(2.0) * cx
            mask = np.clip(d / d_max, 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask) * float(self.strength)
        mask = mask[..., None]
        canvas.commit_array((arr * (1.0 - mask) + bg[None, None, :] * mask) / 255.0)
