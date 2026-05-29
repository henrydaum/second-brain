from plugins.BaseSkill import BaseSkill, Slider, Palette

import math
import random
import numpy as np
from PIL import Image, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class BrownianTreeSkill(BaseSkill):
    name = 'Brownian Tree'
    description = 'Diffusion-limited aggregation (DLA): particles random-walk inward from a ring and freeze where they touch the growing cluster, building a frost-like dendritic Brownian tree from a single central seed. Attachment time drives the palette ramp so the oldest core is dark and fresh outer tips glow. Good for "DLA", "Brownian tree", "dendrite", "frost", "lightning", "coral", or organic fractal growth.'
    kind = "background"

    palette = Palette()
    density = Slider(0.4, 2.5, default=1.0, step=0.1)

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        rng = random.Random(seed)

        G = 240                       # coarse simulation grid, upscaled at the end
        occ = np.zeros((G, G), dtype=bool)
        age = np.zeros((G, G), dtype=np.float32)
        c = G // 2
        occ[c, c] = True
        max_r = 2.0                   # current cluster radius (grid units)

        target = int(2600 * float(self.density))
        spawn_pad = 5
        step = 0
        attached = 0
        guard = 0
        max_guard = target * 600

        while attached < target and max_r < (G / 2 - 4) and guard < max_guard:
            sr = min(G / 2 - 2, max_r + spawn_pad)
            ang = rng.random() * math.tau
            x = int(c + sr * math.cos(ang))
            y = int(c + sr * math.sin(ang))
            kill = sr + 12
            while True:
                guard += 1
                if guard >= max_guard:
                    break
                x += rng.randint(-1, 1)
                y += rng.randint(-1, 1)
                if x < 1 or y < 1 or x >= G - 1 or y >= G - 1:
                    break
                if (x - c) ** 2 + (y - c) ** 2 > kill * kill:
                    break
                if (occ[y - 1, x] or occ[y + 1, x] or occ[y, x - 1] or occ[y, x + 1]
                        or occ[y - 1, x - 1] or occ[y - 1, x + 1]
                        or occ[y + 1, x - 1] or occ[y + 1, x + 1]):
                    occ[y, x] = True
                    step += 1
                    age[y, x] = step
                    attached += 1
                    rr = math.hypot(x - c, y - c)
                    if rr > max_r:
                        max_r = rr
                    break

        amax = float(age.max()) or 1.0
        t = np.zeros((G, G), dtype=np.float32)
        m = occ
        # Newest (outer) = bright; oldest core = dark.
        t[m] = 0.2 + 0.75 * (age[m] / amax)

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT - 1))) for k in range(LUT)],
            dtype=np.uint8,
        )
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        rgb = np.empty((G, G, 3), dtype=np.uint8)
        rgb[:] = bg
        idx = np.clip((t * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb[m] = lut[idx[m]]

        small = Image.fromarray(rgb, "RGB")
        out = small.resize((s, s), Image.NEAREST).convert("RGBA")
        glow = small.resize((s, s), Image.BILINEAR).convert("RGBA").filter(
            ImageFilter.GaussianBlur(radius=s * 0.004)
        )
        canvas.commit(Image.alpha_composite(glow, out))
