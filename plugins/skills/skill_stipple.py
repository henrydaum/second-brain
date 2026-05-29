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
    density  = Slider(0.4, 2.5, default=1.0, step=0.1)
    dot_size = Slider(0.5, 1.6, default=1.0, step=0.05)

    def run(self, canvas):
        s = int(canvas.size)
        rng = np.random.default_rng(int(canvas.seed))
        arr = canvas.image_array(mode="RGB", dtype="float")
        lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        darkness = np.clip(1.0 - lum, 0.0, 1.0)

        # Evenly-spaced jittered grid (blue-noise-ish) with clear gaps between
        # cells; dot SIZE encodes tone, so dark areas grow fat dots and light
        # areas thin to nothing — intentional dotwork, not random scatter.
        cells = int(45 * float(self.density)) + 25
        cell = s / cells
        rmax = cell * 0.5 * float(self.dot_size)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        ink = art_kit.palette_color(0.9)

        jx = (rng.random((cells, cells)) - 0.5) * cell * 0.65
        jy = (rng.random((cells, cells)) - 0.5) * cell * 0.65
        for j in range(cells):
            for i in range(cells):
                x = (i + 0.5) * cell + jx[j, i]
                y = (j + 0.5) * cell + jy[j, i]
                ix = min(max(int(x), 0), s - 1)
                iy = min(max(int(y), 0), s - 1)
                d = float(darkness[iy, ix])
                r = rmax * (d ** 1.1)            # steep: light tones vanish
                if r < 0.5:
                    continue
                draw.ellipse((x - r, y - r, x + r, y + r), fill=ink)

        canvas.commit(img)
