from plugins.BaseTechnique import BaseTechnique, Slider

import numpy as np
from PIL import Image, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class TiltShiftTechnique(BaseTechnique):
    name = 'Tilt Shift'
    description = "Blur the top and bottom of the image while keeping a horizontal focus band sharp — the 'miniature' look."
    kind = "filter"

    focus_y    = Slider(0.05, 0.95, default=0.55, step=0.05)
    focus_band = Slider(0.05, 0.6, default=0.22, step=0.05)
    max_blur   = Slider(1, 40, default=12, step=1)

    def run(self, canvas):
        img = canvas.image.convert("RGB")
        s = canvas.size
        blurred = img.filter(ImageFilter.GaussianBlur(float(self.max_blur)))
        yy = np.arange(s).astype(np.float32) / max(1, s - 1)
        d = np.abs(yy - float(self.focus_y))
        edge0 = float(self.focus_band) / 2.0
        edge1 = edge0 + 0.15
        t = np.clip((d - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
        smooth = (t * t * (3.0 - 2.0 * t))
        mask_strip = (smooth * 255.0).astype(np.uint8).reshape(s, 1)
        mask = Image.fromarray(mask_strip, "L").resize((s, s), Image.NEAREST)
        out = Image.composite(blurred, img, mask)
        canvas.commit(out.convert("RGBA"))
