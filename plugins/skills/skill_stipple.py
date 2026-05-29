from plugins.BaseSkill import BaseSkill, Slider, Palette

import numpy as np
from PIL import Image, ImageDraw

try:
    art_kit
except NameError:
    art_kit = None


class StippleSkill(BaseSkill):
    name = 'Stipple'
    description = 'Stipple the canvas into a field of dots whose density follows the image: dark regions gather many dots, bright regions stay sparse, recreating tone purely from point density like pen-and-ink pointillism. Dots are palette-colored on the palette background. Good for "stipple", "pointillism", "dotwork", "ink dots", "halftone points", or a stippled engraving.'
    kind = "filter"

    palette  = Palette()
    density  = Slider(0.3, 3.0, default=1.0, step=0.1)
    dot_size = Slider(0.5, 4.0, default=1.6, step=0.1)

    def run(self, canvas):
        s = int(canvas.size)
        rng = np.random.default_rng(int(canvas.seed))
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        darkness = np.clip(1.0 - lum, 0.0, 1.0)

        n_candidates = int(120000 * float(self.density))
        xs = rng.integers(0, s, n_candidates)
        ys = rng.integers(0, s, n_candidates)
        prob = darkness[ys, xs] ** 2.2          # contrasty: dots cluster in shadow
        keep = rng.random(n_candidates) < prob
        xs, ys = xs[keep], ys[keep]

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        dr = float(self.dot_size) * (s / 768.0)
        ink = art_kit.palette_color(0.88)        # single ink tone reads as dotwork
        for x, y in zip(xs.tolist(), ys.tolist()):
            draw.ellipse((x - dr, y - dr, x + dr, y + dr), fill=ink)

        canvas.commit(img)
