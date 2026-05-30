from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import math
import random
import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class GeometricAvoidanceTechnique(BaseTechnique):
    name = "Geometric Avoidance"
    description = "Neon circuit routing: particles draw angular traces, reroute before crossing any trail, and quietly terminate when boxed in. Some routes can be nudged by a noise field."
    kind = "background"
    palette = Palette()
    particles = Slider(20, 220, default=150, step=5)
    noise_fraction = Slider(0.0, 1.0, default=0.25, step=0.05)
    step_size = Slider(1, 6, default=2, step=1)
    palette_mode = Enum([("neon", "Neon"), ("per_particle", "Per Particle"), ("gradient", "Gradient")], default="neon")

    def run(self, canvas):
        s, aa, seed = canvas.size, 2, canvas.seed
        rng = random.Random(seed)
        img = Image.new("RGBA", (s * aa, s * aa), "#05060a")
        draw = ImageDraw.Draw(img, "RGBA")
        occ = np.zeros((s * aa, s * aa), dtype=bool)
        grid = max(6, int(float(self.step_size) * 4))
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        field = art_kit.flow_field(seed, scale=0.002, octaves=3)

        def snap(v):
            return round(v / grid) * grid

        def line_pts(a, b):
            x0, y0 = a
            x1, y1 = b
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) * aa * 1.5))
            return [(int((x0 + (x1 - x0) * i / n) * aa), int((y0 + (y1 - y0) * i / n) * aa)) for i in range(1, n + 1)]

        def blocked(pts):
            S = s * aa
            for x, y in pts[4:]:
                if not (2 <= x < S - 2 and 2 <= y < S - 2) or occ[y - 1:y + 2, x - 1:x + 2].any():
                    return True
            return False

        def mark(pts):
            S = s * aa
            for x, y in pts:
                if 2 <= x < S - 2 and 2 <= y < S - 2:
                    occ[y - 1:y + 2, x - 1:x + 2] = True

        def color(i, t):
            if str(self.palette_mode) == "neon":
                return ["#ff4f9a", "#68a8ff", "#ffb26b", "#b95cff", "#8bf6ff"][i % 5]
            if str(self.palette_mode) == "gradient":
                return art_kit.palette_color(i / max(1, int(self.particles) - 1))
            return art_kit.palette_color(t)

        starts = []
        for i in range(int(self.particles)):
            if rng.random() < 0.82:
                x = snap(rng.uniform(s * 0.25, s * 0.75))
                y = snap(rng.uniform(s * 0.30, s * 0.70))
            else:
                side = rng.randrange(4)
                x = snap(rng.choice([rng.uniform(0.04 * s, 0.20 * s), rng.uniform(0.80 * s, 0.96 * s)]) if side < 2 else rng.uniform(0.04 * s, 0.96 * s))
                y = snap(rng.uniform(0.04 * s, 0.96 * s) if side < 2 else rng.choice([rng.uniform(0.04 * s, 0.20 * s), rng.uniform(0.80 * s, 0.96 * s)]))
            starts.append((x, y, dirs[rng.randrange(len(dirs))], i < int(self.particles) * float(self.noise_fraction), color(i, rng.random())))

        for i, (x, y, direction, noisy, ink) in enumerate(starts):
            for _ in range(rng.randint(28, 78)):
                choices = dirs[:]
                if noisy:
                    a = field(x, y)
                    choices.sort(key=lambda d: -math.cos(a - math.atan2(d[1], d[0])))
                else:
                    rng.shuffle(choices)
                    choices.insert(0, direction)
                moved = False
                for dx, dy in choices[:6]:
                    length = grid * rng.randint(1, 5)
                    nx, ny = x + dx * length, y + dy * length
                    pts = line_pts((x, y), (nx, ny))
                    if blocked(pts):
                        continue
                    c = art_kit.with_alpha(ink, 0.72 + 0.25 * rng.random())
                    draw.line((x * aa, y * aa, nx * aa, ny * aa), fill=c, width=2)
                    if rng.random() < 0.22:
                        ox, oy = -dy * grid * 0.32, dx * grid * 0.32
                        draw.line(((x + ox) * aa, (y + oy) * aa, (nx + ox) * aa, (ny + oy) * aa), fill=art_kit.with_alpha(ink, 0.55), width=1)
                    mark(pts)
                    x, y, direction, moved = nx, ny, (dx, dy), True
                    if rng.random() < 0.18:
                        notch = grid * rng.choice([-1, 1])
                        draw.line((x * aa, y * aa, (x - dy * notch) * aa, (y + dx * notch) * aa), fill=art_kit.with_alpha(ink, 0.6), width=1)
                    break
                if not moved:
                    break

        canvas.commit(img.resize((s, s), Image.Resampling.LANCZOS))
