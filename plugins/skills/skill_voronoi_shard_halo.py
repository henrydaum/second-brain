from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
import random
from PIL import ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class VoronoiShardHaloSkill(BaseSkill):
    name = "Voronoi Shard Halo"
    description = "Object overlay: crystalline radial shards that echo Voronoi cells without paying for a full pixel map."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.5, step=0.04)
    cy = Slider(0, 1, default=0.5, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.45, 1.45, default=1.0, step=0.05)
    variant = Enum([("halo", "Halo"), ("burst", "Burst"), ("crown", "Crown")], default="halo")

    def run(self, canvas):
        s, rng = canvas.size, random.Random(canvas.seed)
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, R, n = self.cx * s, self.cy * s, s * 0.28 * self.scale, {"halo": 22, "burst": 32, "crown": 16}[str(self.variant)]
        rings = (0.25, 0.58, 1.0) if self.variant != "crown" else (0.42, 0.75, 1.02)
        pts = [[(cx + math.cos(art_kit.tau * i / n + rng.uniform(-0.05, 0.05)) * R * r, cy + math.sin(art_kit.tau * i / n + rng.uniform(-0.05, 0.05)) * R * r) for i in range(n)] for r in rings]
        for i in range(n):
            for a in range(len(pts) - 1):
                poly = [pts[a][i], pts[a][(i + 1) % n], pts[a + 1][(i + 1) % n], pts[a + 1][i]]
                fill = art_kit.with_alpha(art_kit.palette_color((i / n + a * 0.18) % 1), 80 + 45 * a)
                draw.polygon(poly, fill=fill, outline=art_kit.with_alpha(canvas.palette.background, 80))
        canvas.commit(img)
