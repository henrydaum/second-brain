from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
import random
from PIL import ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class ParticleConstellationSkill(BaseSkill):
    name = "Particle Constellation"
    description = "Object overlay: connected particle nodes arranged as a cluster, arc, or loose star map."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.48, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=1.0, step=0.05)
    variant = Enum([("cluster", "Cluster"), ("arc", "Arc"), ("map", "Map")], default="cluster")

    def run(self, canvas):
        s, rng = canvas.size, random.Random(canvas.seed)
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, R, n = self.cx * s, self.cy * s, s * 0.29 * self.scale, {"cluster": 54, "arc": 44, "map": 62}[str(self.variant)]
        pts = []
        for i in range(n):
            a = rng.random() * art_kit.tau if self.variant != "arc" else art_kit.remap(i, 0, n - 1, -2.5, 0.4)
            d = R * (rng.random() ** 0.55 if self.variant != "map" else rng.random())
            pts.append((cx + math.cos(a) * d, cy + math.sin(a) * d * 0.72))
        limit = R * (0.24 if self.variant != "map" else 0.18)
        for i, (x, y) in enumerate(pts):
            for x2, y2 in pts[i + 1:]:
                d = math.hypot(x - x2, y - y2)
                if d < limit:
                    draw.line((x, y, x2, y2), fill=art_kit.with_alpha(art_kit.palette_color(d / limit), 75), width=1)
        for i, (x, y) in enumerate(pts):
            r = s * rng.uniform(0.004, 0.011) * self.scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill=art_kit.with_alpha(art_kit.palette_color(i / n), 210))
        canvas.commit(img.filter(ImageFilter.GaussianBlur(radius=0.15)))
