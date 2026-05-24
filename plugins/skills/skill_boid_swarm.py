from plugins.BaseSkill import BaseSkill, Enum, Palette, Pan, Slider

import math
import random
from PIL import ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class BoidSwarmSkill(BaseSkill):
    name = "Boid Swarm"
    description = "Object overlay: a frozen flock of small palette triangles aligned by migration, vortex, or burst motion."
    kind = "object"
    palette = Palette()
    cx = Slider(0, 1, default=0.52, step=0.04)
    cy = Slider(0, 1, default=0.46, step=0.04)
    center = Pan(x="cx", y="cy")
    scale = Slider(0.5, 1.5, default=1.0, step=0.05)
    variant = Enum([("migration", "Migration"), ("vortex", "Vortex"), ("burst", "Burst")], default="migration")

    def run(self, canvas):
        s, rng = canvas.size, random.Random(canvas.seed)
        img = canvas.new_layer()
        draw = ImageDraw.Draw(img, "RGBA")
        cx, cy, spread, n = self.cx * s, self.cy * s, s * 0.34 * self.scale, int(46 * self.scale)
        for i in range(max(18, n)):
            a = rng.random() * art_kit.tau
            d = spread * (rng.random() ** 0.55)
            x, y = cx + math.cos(a) * d, cy + math.sin(a) * d * 0.7
            heading = (-0.35 if self.variant == "migration" else a + math.pi / 2 if self.variant == "vortex" else a)
            size = s * rng.uniform(0.011, 0.024) * self.scale
            tri = art_kit.regular_polygon(x, y, size, 3, heading, y_scale=0.72)
            fill = art_kit.with_alpha(art_kit.palette_color(0.2 + 0.75 * rng.random()), 135 + rng.randrange(90))
            draw.polygon(tri, fill=fill)
            draw.line((x - math.cos(heading) * size * 1.4, y - math.sin(heading) * size * 1.4, x, y), fill=art_kit.with_alpha(canvas.palette.background, 60), width=1)
        canvas.commit(img)
