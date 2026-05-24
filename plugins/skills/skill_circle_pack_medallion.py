from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
import random
from PIL import ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class CirclePackMedallionSkill(BaseSkill):
    name = "Circle Pack Medallion"
    description = "Object overlay: a compact non-overlapping circle pack badge with organic, gridded, or spiral seeding."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.5, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=0.9, step=0.05)
    variant = Enum([("organic", "Organic"), ("grid", "Grid"), ("swirl", "Swirl")], default="organic")

    def run(self, canvas):
        s, rng = canvas.size, random.Random(canvas.seed)
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, R, circles = self.cx * s, self.cy * s, s * 0.26 * self.scale, []
        for i in range(560):
            if self.variant == "swirl":
                a, d = i * 2.39996, R * (i / 560) ** 0.5
                x, y = cx + math.cos(a) * d, cy + math.sin(a) * d
            else:
                x, y = cx + rng.uniform(-R, R), cy + rng.uniform(-R, R)
                if self.variant == "grid":
                    step = max(5, int(s * 0.035 * self.scale))
                    x, y = round(x / step) * step, round(y / step) * step
            if math.hypot(x - cx, y - cy) > R:
                continue
            r = rng.uniform(s * 0.008, s * 0.033) * self.scale
            if all(math.hypot(x - px, y - py) > r + pr + 1 for px, py, pr in circles):
                circles.append((x, y, r))
            if len(circles) >= 92:
                break
        for i, (x, y, r) in enumerate(sorted(circles, key=lambda c: c[2], reverse=True)):
            fill = art_kit.with_alpha(art_kit.palette_color(0.15 + 0.8 * i / max(1, len(circles))), 185)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=art_kit.with_alpha(canvas.palette.background, 110), width=max(1, int(r * 0.12)))
        canvas.commit(img)
