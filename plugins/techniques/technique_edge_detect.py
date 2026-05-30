from plugins.BaseTechnique import BaseTechnique, Slider, Bool, Palette

import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class EdgeDetectTechnique(BaseTechnique):
    name = 'Edge Detect'
    description = 'Sobel edge map rendered in palette colors: edges painted with palette.primary on a palette.background field.'
    kind = "filter"

    palette  = Palette()
    strength = Slider(0.5, 6.0, default=2.0, step=0.1)
    invert   = Bool(default=False)

    def run(self, canvas):
        k = float(self.strength)
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        gx = np.zeros_like(lum)
        gy = np.zeros_like(lum)
        gx[:, 1:-1] = (
            lum[:, 2:] - lum[:, :-2]
            + 0.5 * (np.roll(lum[:, 2:], -1, axis=0) - np.roll(lum[:, :-2], -1, axis=0))
            + 0.5 * (np.roll(lum[:, 2:], 1, axis=0) - np.roll(lum[:, :-2], 1, axis=0))
        )
        gy[1:-1, :] = (
            lum[2:, :] - lum[:-2, :]
            + 0.5 * (np.roll(lum[2:, :], -1, axis=1) - np.roll(lum[:-2, :], -1, axis=1))
            + 0.5 * (np.roll(lum[2:, :], 1, axis=1) - np.roll(lum[:-2, :], 1, axis=1))
        )
        mag = np.clip(np.sqrt(gx * gx + gy * gy) * k, 0.0, 1.0)
        if self.invert:
            mag = 1.0 - mag
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        fg = np.array(art_kit.hex_to_rgb(canvas.palette.primary), dtype=np.float32) / 255.0
        m = mag[..., None]
        canvas.commit_array(bg[None, None, :] * (1.0 - m) + fg[None, None, :] * m)
