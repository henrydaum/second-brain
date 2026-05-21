from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class EdgeDetectSkill(BaseSkill):
    name = 'Edge Detect'
    description = 'Sobel edge map rendered in palette colors: edges painted with palette.primary on a palette.background field. Strength dials the gradient magnitude, invert flips which side gets edges. Params: strength (0.5-6.0, default 2.0), invert bool.'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'palette', 'name': 'palette', 'label': 'Palette'},
        {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': 0.5, 'max': 6.0, 'step': 0.1, 'default': 2.0},
        {'type': 'bool', 'name': 'invert', 'label': 'Invert', 'default': False},
    ]

    def run(self, canvas, strength=2.0, invert=False):
        k = float(art_kit.clamp(strength, 0.1, 10.0))
        img = canvas.image.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722
        # Sobel via slicing.
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
        mag = np.sqrt(gx * gx + gy * gy)
        mag = np.clip(mag * k, 0.0, 1.0)
        if bool(invert):
            mag = 1.0 - mag
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32) / 255.0
        fg = np.array(art_kit.hex_to_rgb(canvas.palette.primary), dtype=np.float32) / 255.0
        m = mag[..., None]
        out = bg[None, None, :] * (1.0 - m) + fg[None, None, :] * m
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        canvas.commit(Image.fromarray(out, "RGB").convert("RGBA"))
