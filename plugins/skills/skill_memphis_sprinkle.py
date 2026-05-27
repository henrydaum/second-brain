from plugins.BaseSkill import BaseSkill, Palette, Slider
from PIL import ImageDraw

import math
import random

try:
    art_kit
except NameError:
    art_kit = None


class MemphisSprinkleSkill(BaseSkill):
    name = "Memphis Sprinkle"
    description = "Object overlay: scattered Memphis-style ornaments — squiggles, terrazzo blobs, jagged triangles, confetti dots — in palette colors."
    kind = "object"
    palette = Palette()
    density = Slider(0.2, 1.0, default=0.55, step=0.02)
    squiggle_share = Slider(0.0, 1.0, default=0.35, step=0.02)
    rotation_jitter = Slider(0.0, 1.0, default=0.6, step=0.02)

    def _squiggle(self, draw, cx, cy, r, rotation, color, lw):
        pts = []
        n = 6
        for i in range(n):
            t = i / (n - 1) - 0.5
            x = t * 2 * r
            y = math.sin(t * art_kit.tau) * r * 0.45
            cosr, sinr = math.cos(rotation), math.sin(rotation)
            pts.append((cx + x * cosr - y * sinr, cy + x * sinr + y * cosr))
        draw.line(pts, fill=color, width=lw, joint="curve")

    def _blob(self, draw, cx, cy, r, rng, color, outline, lw):
        n = 7
        pts = []
        for i in range(n):
            theta = art_kit.tau * i / n
            rr = r * (0.7 + rng.random() * 0.5)
            pts.append((cx + math.cos(theta) * rr, cy + math.sin(theta) * rr))
        if lw <= 1:
            draw.polygon(pts, fill=color, outline=outline)
        else:
            draw.polygon(pts, fill=color)
            for i in range(len(pts)):
                draw.line([pts[i], pts[(i + 1) % len(pts)]], fill=outline, width=lw)

    def _jagged_triangle(self, draw, cx, cy, r, rotation, color, outline, lw):
        pts = art_kit.regular_polygon(cx, cy, r, 3, rotation=rotation)
        if lw <= 1:
            draw.polygon(pts, fill=color, outline=outline)
        else:
            draw.polygon(pts, fill=color)
            for i in range(3):
                draw.line([pts[i], pts[(i + 1) % 3]], fill=outline, width=lw)

    def _dot(self, draw, cx, cy, r, color):
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    def run(self, canvas):
        img = canvas.new_layer()
        rng = random.Random(canvas.seed)
        size = canvas.size
        draw = ImageDraw.Draw(img, "RGBA")
        density = float(self.density)
        squig_p = float(self.squiggle_share)
        rot_jit = float(self.rotation_jitter)
        outline = canvas.palette.background
        accent_colors = [canvas.palette.primary, canvas.palette.accent, canvas.palette.secondary]

        n_cells = int(8 + density * 16)
        cells = art_kit.jittered_grid(rng, n_cells, n_cells, jitter=0.65)
        margin = size * 0.04
        usable = size - 2 * margin
        cell_size = usable / n_cells
        ornament_r = cell_size * 0.32

        keep = int(len(cells) * (0.3 + 0.5 * density))
        rng.shuffle(cells)
        cells = cells[:keep]
        lw = max(1, int(round(size / 220)))

        for cell in cells:
            cx = margin + cell[0] * usable
            cy = margin + cell[1] * usable
            r = ornament_r * rng.uniform(0.7, 1.4)
            rot = (rng.random() - 0.5) * art_kit.tau * rot_jit
            color = rng.choice(accent_colors)
            roll = rng.random()
            if roll < squig_p:
                self._squiggle(draw, cx, cy, r, rot, color, lw + 1)
            elif roll < squig_p + (1 - squig_p) / 3:
                self._blob(draw, cx, cy, r, rng, color, outline, lw)
            elif roll < squig_p + 2 * (1 - squig_p) / 3:
                self._jagged_triangle(draw, cx, cy, r, rot, color, outline, lw)
            else:
                self._dot(draw, cx, cy, r * 0.55, color)

        canvas.commit(img)
