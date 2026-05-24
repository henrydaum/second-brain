from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
from PIL import ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class WaveRibbonSkill(BaseSkill):
    name = "Wave Ribbon"
    description = "Object overlay: braided sine and moire ribbons, quick to render and strong over fractal or terrain layers."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.5, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=1.0, step=0.05)
    variant = Enum([("braid", "Braid"), ("moire", "Moire"), ("banner", "Banner")], default="braid")

    def run(self, canvas):
        s = canvas.size
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, L, amp = self.cx * s, self.cy * s, s * 0.34 * self.scale, s * 0.055 * self.scale
        strands = 7 if self.variant == "moire" else 5
        for j in range(strands):
            pts = []
            phase = j * art_kit.tau / strands
            for i in range(180):
                t = i / 179
                x = cx - L + 2 * L * t
                y = cy + math.sin(t * art_kit.tau * (2.0 if self.variant != "banner" else 1.0) + phase) * amp
                y += math.sin(t * art_kit.tau * 7 + phase * 0.7) * amp * (0.33 if self.variant == "moire" else 0.12)
                pts.append((x, y + (j - strands / 2) * s * 0.012 * self.scale))
            draw.line(pts, fill=art_kit.with_alpha(art_kit.palette_color(0.15 + 0.75 * j / max(1, strands - 1)), 150), width=max(1, int(s * 0.005 * self.scale)), joint="curve")
        canvas.commit(img.filter(ImageFilter.GaussianBlur(radius=0.18)))
