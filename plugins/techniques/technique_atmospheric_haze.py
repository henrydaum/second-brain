from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


def _desaturate(rgb, amount):
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    gray = np.stack([lum, lum, lum], axis=-1)
    return rgb * (1.0 - amount[..., None]) + gray * amount[..., None]


class AtmosphericHazeTechnique(BaseTechnique):
    name = 'Atmospheric Haze'
    description = 'Shift hue and reduce saturation toward the palette background as a function of vertical position, mimicking the way distant landscapes lose contrast and cool toward sky.'
    kind = "filter"

    palette   = Palette()
    direction = Enum([
        ('top',    'Top (Atmospheric)'),
        ('bottom', 'Bottom (Low Fog)'),
        ('both',   'Both Ends'),
    ], default='top')
    strength  = Slider(0.0, 1.0, default=0.45, step=0.05)

    def run(self, canvas):
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float") * 255.0
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)
        ys = np.linspace(0.0, 1.0, s, dtype=np.float32)
        if self.direction == "top":
            mask = 1.0 - ys
        elif self.direction == "bottom":
            mask = ys
        else:
            mask = 1.0 - 2.0 * np.abs(ys - 0.5)
            mask = np.clip(mask, 0.0, 1.0)
            mask = 1.0 - mask
        mask = mask * mask * (3.0 - 2.0 * mask) * float(self.strength)
        mask2d = np.broadcast_to(mask[:, None], (s, s)).astype(np.float32)
        arr = _desaturate(arr, mask2d * 0.6)
        mask3 = mask2d[..., None]
        out = arr * (1.0 - mask3) + bg[None, None, :] * mask3
        canvas.commit_array(out / 255.0)
