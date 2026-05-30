from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import random
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class VogelBloomTechnique(BaseTechnique):
    name = 'Vogel Bloom'
    description = 'A flower as a Vogel sunflower spiral: cells laid out by the golden angle, sized by their seed index, colored by palette ramp position. No stacked petals, no center circle -- the bloom emerges from the spiral itself. Good for "flower", "sunflower", "bloom", "seed pod", or "mandala".'
    kind = "background"
    palette = Palette()
    density = Enum([('sparse', 'Sparse'), ('full', 'Full'), ('packed', 'Packed')], default='full')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        n = {"sparse": 220, "full": 600, "packed": 1100}.get(str(self.density), 600)
        rng = random.Random(seed)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")

        cx, cy = s / 2.0, s / 2.0
        scale = s * 0.46
        points = art_kit.vogel_spiral(n, scale=scale)

        for i, (x, y) in enumerate(points):
            t = i / max(1, n - 1)
            # Radius grows toward outer cells; inner cells small and dense.
            r = 2.0 + (1.0 - t) * (s * 0.012) + t * (s * 0.018)
            r *= 0.85 + rng.random() * 0.3
            # Palette ramp: inner cells brighter (closer to accent), outer toward secondary.
            ramp_t = 0.15 + 0.8 * (1.0 - t)
            color = art_kit.palette_color(ramp_t, value=0.9 + rng.random() * 0.2)
            px, py = cx + x, cy + y
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color)

        # Soft bloom on a copy, composited under for glow.
        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.012))
        out = Image.alpha_composite(glow, img)
        canvas.commit(out)
