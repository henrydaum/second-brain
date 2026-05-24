from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
from PIL import ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class SpirographOrbitSkill(BaseSkill):
    name = "Spirograph Orbit"
    description = "Object overlay: layered hypotrochoid and rose curves, useful as a precise orbital emblem over noisy backgrounds."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.5, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=0.85, step=0.05)
    variant = Enum([("rose", "Rose"), ("gear", "Gear"), ("nested", "Nested")], default="gear")

    def run(self, canvas):
        s = canvas.size
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, R = self.cx * s, self.cy * s, s * 0.23 * self.scale
        layers = 5 if self.variant == "nested" else 3
        for j in range(layers):
            pts = []
            k = (4 + j) if self.variant == "rose" else (2.7 + j * 0.55)
            for i in range(720):
                t = art_kit.tau * i / 719
                if self.variant == "rose":
                    r = R * (0.58 + 0.1 * j) * math.cos(k * t)
                    x, y = cx + math.cos(t) * r, cy + math.sin(t) * r
                else:
                    a, b, d = R * 0.58, R * (0.18 + j * 0.025), R * (0.35 + j * 0.08)
                    x = cx + (a - b) * math.cos(t) + d * math.cos((a - b) / b * t + j)
                    y = cy + (a - b) * math.sin(t) - d * math.sin((a - b) / b * t + j)
                pts.append((x, y))
            draw.line(pts, fill=art_kit.with_alpha(art_kit.palette_color(0.2 + 0.65 * j / max(1, layers - 1)), 160), width=max(1, int(s * 0.0035 * self.scale)), joint="curve")
        canvas.commit(img)
