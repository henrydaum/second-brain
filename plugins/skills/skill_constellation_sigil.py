from plugins.BaseSkill import BaseSkill, Palette, Slider
from PIL import ImageDraw

import math
import random

try:
    art_kit
except NameError:
    art_kit = None


class ConstellationSigilSkill(BaseSkill):
    name = "Constellation Sigil"
    description = "Object overlay: a sparse star-map of palette dots and thin connecting lines, with a few highlighted bright stars."
    kind = "object"
    palette = Palette()
    density = Slider(0.2, 1.0, default=0.55, step=0.02)
    link_chance = Slider(0.0, 1.0, default=0.35, step=0.02)
    big_star_count = Slider(1, 7, default=3, step=1)

    def run(self, canvas):
        img = canvas.new_layer()
        rng = random.Random(canvas.seed)
        size = canvas.size
        draw = ImageDraw.Draw(img, "RGBA")

        n = int(round(24 + float(self.density) * 90))
        cols = max(3, int(math.sqrt(n) * 1.1))
        rows = max(3, int(math.sqrt(n) * 0.9))
        pts_norm = art_kit.jittered_grid(rng, cols, rows, jitter=0.85)
        rng.shuffle(pts_norm)
        pts_norm = pts_norm[:n]

        margin = size * 0.08
        pts = [
            (margin + p[0] * (size - 2 * margin), margin + p[1] * (size - 2 * margin))
            for p in pts_norm
        ]

        link_p = float(self.link_chance)
        max_link_dist = size * 0.22
        max_link2 = max_link_dist * max_link_dist
        line_w = max(1, int(round(size / 600)))
        line_color = art_kit.with_alpha(canvas.palette.accent, 150)

        def dist2(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return dx * dx + dy * dy

        for i, p in enumerate(pts):
            ranked = sorted(
                ((dist2(p, q), j) for j, q in enumerate(pts) if j != i),
                key=lambda t: t[0],
            )[:2]
            for d2, j in ranked:
                if j <= i or d2 > max_link2:
                    continue
                if rng.random() > link_p:
                    continue
                draw.line([p, pts[j]], fill=line_color, width=line_w)

        dot_radius = max(2, int(round(size / 420)))
        dot_color = art_kit.with_alpha(canvas.palette.primary, 230)
        for p in pts:
            r = dot_radius
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=dot_color)

        big_n = int(round(float(self.big_star_count)))
        big_indices = rng.sample(range(len(pts)), min(big_n, len(pts)))
        glow_arm_color = art_kit.with_alpha(canvas.palette.secondary, 130)
        for bi in big_indices:
            x, y = pts[bi]
            R = max(6, int(round(size / 110)))
            for kr, alpha in ((int(R * 1.8), 55), (int(R * 1.4), 110), (R, 240)):
                fill = art_kit.with_alpha(canvas.palette.accent, alpha)
                draw.ellipse([x - kr, y - kr, x + kr, y + kr], fill=fill)
            arm = int(R * 2.6)
            draw.line([(x - arm, y), (x + arm, y)], fill=glow_arm_color, width=line_w)
            draw.line([(x, y - arm), (x, y + arm)], fill=glow_arm_color, width=line_w)

        canvas.commit(img)
