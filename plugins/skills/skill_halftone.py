from plugins.BaseSkill import BaseSkill, Slider, Palette

import math
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class HalftoneSkill(BaseSkill):
    name = 'Halftone'
    description = 'Newspaper-style halftone dot screen. The image is replaced by a regular grid of palette-tinted dots whose radius scales with local luminance.'
    kind = "effect"

    palette   = Palette()
    cell_size = Slider(6, 40, default=12, step=1)
    angle     = Slider(0, 90, default=0, step=5)

    def run(self, canvas):
        c = int(self.cell_size)
        a = math.radians(float(self.angle))
        s = canvas.size
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722

        out = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(out, "RGBA")
        cos_a, sin_a = math.cos(a), math.sin(a)
        diag = int(s * 1.5)
        for j in range(-diag // c, diag // c):
            for i in range(-diag // c, diag // c):
                gx = i * c
                gy = j * c
                x = s / 2.0 + (gx * cos_a - gy * sin_a)
                y = s / 2.0 + (gx * sin_a + gy * cos_a)
                if not (0 <= x < s and 0 <= y < s):
                    continue
                x0 = max(0, int(x - c / 2))
                y0 = max(0, int(y - c / 2))
                x1 = min(s, x0 + c)
                y1 = min(s, y0 + c)
                if x1 <= x0 or y1 <= y0:
                    continue
                l_avg = float(lum[y0:y1, x0:x1].mean())
                r = (1.0 - l_avg) * (c * 0.55)
                if r < 0.5:
                    continue
                draw.ellipse((x - r, y - r, x + r, y + r), fill=art_kit.palette_color(l_avg))
        canvas.commit(out)
