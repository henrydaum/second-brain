from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
import random
from PIL import ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class AsemicRibbonSkill(BaseSkill):
    name = "Asemic Ribbon"
    description = "Object overlay: seeded flow-field calligraphy, like invented handwriting wrapped around a focal point."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.52, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=0.95, step=0.05)
    variant = Enum([("script", "Script"), ("wire", "Wire"), ("orbit", "Orbit")], default="script")

    def run(self, canvas):
        s, rng = canvas.size, random.Random(canvas.seed)
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, r = self.cx * s, self.cy * s, s * 0.27 * self.scale
        field = art_kit.flow_field(canvas.seed, scale=0.004 if self.variant != "wire" else 0.009, octaves=4)
        for k in range(14 if self.variant == "wire" else 9):
            a = rng.random() * art_kit.tau
            x, y = cx + math.cos(a) * r * rng.random(), cy + math.sin(a) * r * 0.45 * rng.random()
            pts = []
            for i in range(70):
                ang = field(x, y) if self.variant != "orbit" else a + i * 0.045 + math.sin(i * 0.17 + k)
                x += math.cos(ang) * s * 0.006 * self.scale
                y += math.sin(ang) * s * 0.006 * self.scale
                pts.append((x, y))
            if len(pts) > 1:
                draw.line(pts, fill=art_kit.with_alpha(art_kit.palette_color(0.25 + 0.65 * k / 13), 160), width=max(1, int(s * 0.004 * self.scale)))
        canvas.commit(img.filter(ImageFilter.GaussianBlur(radius=0.35)))
