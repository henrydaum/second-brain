from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
import random
from PIL import ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class DifferentialVineSkill(BaseSkill):
    name = "Differential Vine"
    description = "Object overlay: fast faux differential-growth tendrils that read as roots, coral, or forked lightning."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.68, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=1.0, step=0.05)
    variant = Enum([("root", "Root"), ("coral", "Coral"), ("lightning", "Lightning")], default="root")

    def run(self, canvas):
        s, rng = canvas.size, random.Random(canvas.seed)
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        base = (-math.pi / 2 if self.variant != "root" else math.pi / 2)
        branches = [(self.cx * s, self.cy * s, base, s * 0.12 * self.scale, 0)]
        max_depth = {"root": 5, "coral": 6, "lightning": 4}[str(self.variant)]
        while branches:
            x, y, a, length, depth = branches.pop()
            pts = [(x, y)]
            for _ in range(7):
                a += rng.uniform(-0.35, 0.35) * (1.5 if self.variant == "lightning" else 1.0)
                x += math.cos(a) * length / 7
                y += math.sin(a) * length / 7
                pts.append((x, y))
            color = art_kit.with_alpha(art_kit.palette_color(0.25 + 0.65 * depth / max(1, max_depth)), 190)
            draw.line(pts, fill=color, width=max(1, int(s * 0.01 * self.scale * (1 - depth / (max_depth + 1)))))
            if depth < max_depth:
                for turn in (-1, 1):
                    if rng.random() > (0.25 if self.variant == "lightning" else 0.08):
                        branches.append((x, y, a + turn * rng.uniform(0.35, 0.9), length * rng.uniform(0.56, 0.76), depth + 1))
        canvas.commit(img.filter(ImageFilter.GaussianBlur(radius=0.25)))
