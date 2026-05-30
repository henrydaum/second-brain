from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np
from PIL import Image, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class BarnsleyFernTechnique(BaseTechnique):
    name = 'Barnsley Fern'
    description = 'The Barnsley fern: an iterated function system (the chaos game) run as thousands of parallel walkers, accumulated into a palette-graded density buffer. The four affine maps grow the stem, the shrinking copy, and the left/right fronds into a self-similar leaf. Good for "fern", "fractal plant", "IFS", "chaos game", or a delicate botanical fractal.'
    kind = "background"

    palette = Palette()
    density = Slider(0.4, 3.0, default=1.2, step=0.1)

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        rng = np.random.default_rng(seed)

        # Affine maps: rows are (a, b, c, d, e, f) for x' = ax+by+e, y' = cx+dy+f.
        A = np.array([
            [0.00,  0.00,  0.00,  0.16, 0.00, 0.00],   # stem
            [0.85,  0.04, -0.04,  0.85, 0.00, 1.60],   # successively smaller leaflets
            [0.20, -0.26,  0.23,  0.22, 0.00, 1.60],   # left frond
            [-0.15, 0.28,  0.26,  0.24, 0.00, 0.44],   # right frond
        ], dtype=np.float32)
        cum = np.array([0.01, 0.86, 0.93, 1.00], dtype=np.float32)

        n_walkers = 4000
        n_sample = max(40, int(110 * float(self.density)))
        x = np.zeros(n_walkers, dtype=np.float32)
        y = np.zeros(n_walkers, dtype=np.float32)
        for _ in range(30):  # burn-in onto the attractor
            sel = np.searchsorted(cum, rng.random(n_walkers).astype(np.float32))
            a, b, c, d, e, f = (A[sel, k] for k in range(6))
            x, y = a * x + b * y + e, c * x + d * y + f

        xs = np.empty(n_walkers * n_sample, dtype=np.float32)
        ys = np.empty(n_walkers * n_sample, dtype=np.float32)
        for i in range(n_sample):
            sel = np.searchsorted(cum, rng.random(n_walkers).astype(np.float32))
            a, b, c, d, e, f = (A[sel, k] for k in range(6))
            x, y = a * x + b * y + e, c * x + d * y + f
            xs[i * n_walkers:(i + 1) * n_walkers] = x
            ys[i * n_walkers:(i + 1) * n_walkers] = y

        # Fern spans roughly x in [-2.18, 2.66], y in [0, 9.98]; keep aspect.
        margin = s * 0.05
        avail = s - 2 * margin
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        span = max(x1 - x0, y1 - y0) or 1.0
        scale = avail / span
        cx = (xs - x0) * scale + (s - (x1 - x0) * scale) * 0.5
        # Flip y so the fern grows upward.
        cy = s - (margin + (ys - y0) * scale)
        ix = np.clip(cx.astype(np.int32), 0, s - 1)
        iy = np.clip(cy.astype(np.int32), 0, s - 1)

        density = np.zeros((s, s), dtype=np.float32)
        np.add.at(density, (iy, ix), 1.0)
        density = np.log1p(density)
        dmax = float(density.max()) or 1.0
        density = (density / dmax) ** 0.6

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.25 + 0.7 * (k / (LUT - 1))))
             for k in range(LUT)],
            dtype=np.uint8,
        )
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        idx = np.clip((density * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        rgb[density < 0.02] = bg

        out = Image.fromarray(rgb, "RGB").convert("RGBA")
        glow = out.filter(ImageFilter.GaussianBlur(radius=s * 0.003))
        canvas.commit(Image.alpha_composite(glow, out))
