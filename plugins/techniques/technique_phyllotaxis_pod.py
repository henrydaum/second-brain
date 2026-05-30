from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Pan, Slider

import math
from PIL import ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class PhyllotaxisPodTechnique(BaseTechnique):
    name = "Phyllotaxis Pod"
    description = "Object overlay: a golden-angle seed pod that can read as a bloom, pinecone, or small mandala."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.5, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=0.9, step=0.05)
    variant = Enum([("bloom", "Bloom"), ("cone", "Cone"), ("mandala", "Mandala")], default="bloom")

    def run(self, canvas):
        s = canvas.size
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, R = self.cx * s, self.cy * s, s * 0.25 * self.scale
        n = {"bloom": 260, "cone": 180, "mandala": 320}[str(self.variant)]
        for i, (x, y) in enumerate(art_kit.vogel_spiral(n, scale=R)):
            t = i / max(1, n - 1)
            squash = 0.62 if self.variant == "cone" else 1.0
            px, py = cx + x, cy + y * squash
            r = s * self.scale * (0.014 if self.variant == "mandala" else 0.018) * (1.08 - t * 0.55)
            sides = 6 if self.variant == "cone" else 5 if self.variant == "mandala" else 24
            fill = art_kit.with_alpha(art_kit.palette_color(0.12 + 0.85 * (1 - t)), 150 + int(80 * (1 - t)))
            if sides == 24:
                draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)
            else:
                draw.polygon(art_kit.regular_polygon(px, py, r, sides, i * 0.2), fill=fill)
        canvas.commit(img.filter(ImageFilter.GaussianBlur(radius=0.2)))
