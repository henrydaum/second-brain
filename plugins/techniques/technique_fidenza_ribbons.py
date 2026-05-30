from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import math
import random
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class FidenzaRibbonsTechnique(BaseTechnique):
    name = "Fidenza Ribbons"
    description = "Tyler Hobbs-style flow-field ribbons. Particles march through an fbm-driven angle field; each particle stamps a thick palette-colored band of small offset blocks, with cheap occupancy-based avoidance so ribbons don't pile on top of each other."
    kind = "background"
    palette = Palette()
    band_count = Slider(20, 140, default=70, step=5)
    curl = Slider(0.2, 2.0, default=1.0, step=0.1)
    band_width = Slider(6, 28, default=14, step=1)
    bg = Enum([("ivory", "Ivory"), ("palette_bg", "Palette BG"), ("off_white", "Off White")], default="ivory")

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed
        rng = random.Random(seed)

        bg = {
            "ivory": "#efe9d8",
            "palette_bg": canvas.palette.background,
            "off_white": "#f2f2ee",
        }[str(self.bg)]
        img = Image.new("RGBA", (s, s), bg)
        draw = ImageDraw.Draw(img, "RGBA")

        # Occupancy grid for cheap collision avoidance between ribbons.
        cell = max(4, int(float(self.band_width) * 0.8))
        gw = s // cell + 2
        occupied = [[-1] * gw for _ in range(gw)]

        scale = 0.0015 * float(self.curl)
        field = art_kit.flow_field(seed, scale=scale, octaves=4)

        n = int(self.band_count)
        base_width = float(self.band_width)
        step_len = max(2.0, base_width * 0.45)
        max_steps = int(s / step_len * 2.6)

        for i in range(n):
            x = rng.uniform(-s * 0.05, s * 1.05)
            y = rng.uniform(-s * 0.05, s * 1.05)
            t_ramp = rng.random()
            base_color = art_kit.palette_color(0.15 + 0.8 * t_ramp)
            width = base_width * rng.uniform(0.55, 1.85)
            block_len = rng.uniform(width * 3.0, width * 8.0)
            traveled = 0.0
            for step in range(max_steps):
                if not (-s * 0.1 <= x <= s * 1.1 and -s * 0.1 <= y <= s * 1.1):
                    break
                gx = int(x / cell) + 1
                gy = int(y / cell) + 1
                if 0 <= gx < gw and 0 <= gy < gw and occupied[gy][gx] not in (-1, i):
                    break
                ang = field(x, y)
                nx = x + math.cos(ang) * step_len
                ny = y + math.sin(ang) * step_len
                # Stamp a rotated rectangle perpendicular to motion.
                px = -math.sin(ang) * width * 0.5
                py = math.cos(ang) * width * 0.5
                quad = [
                    (x + px, y + py),
                    (x - px, y - py),
                    (nx - px, ny - py),
                    (nx + px, ny + py),
                ]
                draw.polygon(quad, fill=base_color)
                # Periodic block-break: leave a small visual gap to mimic Fidenza's segmented look.
                traveled += step_len
                if traveled >= block_len:
                    traveled = 0.0
                    block_len = rng.uniform(width * 3.0, width * 8.0)
                    ang = field(nx, ny)
                    nx = nx + math.cos(ang) * step_len * 0.35
                    ny = ny + math.sin(ang) * step_len * 0.35
                    # Occasional color swap.
                    if rng.random() < 0.15:
                        base_color = art_kit.palette_color(rng.random())
                if 0 <= gx < gw and 0 <= gy < gw:
                    occupied[gy][gx] = i
                x, y = nx, ny

        canvas.commit(img)
