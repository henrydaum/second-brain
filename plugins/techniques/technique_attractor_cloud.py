from plugins.BaseTechnique import BaseTechnique, Slider, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class AttractorCloudTechnique(BaseTechnique):
    name = 'Attractor Cloud'
    description = 'Strange-attractor point cloud (de Jong or Clifford) accumulated into a palette-graded density image.'
    kind = "background"
    owner = 'web:0e0c7c0c-92af-46ef-bb48-69154d2c9f44'
    created_at = 1779071212.705876

    palette       = Palette()
    attractor     = Enum([('de_jong', 'de Jong'), ('clifford', 'Clifford')], default='de_jong', label='Attractor Type')
    density_boost = Slider(0.5, 3.0, default=1.0, step=0.1, label='Density')

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed
        kind = self.attractor

        # Curated canonical params. The previous de Jong list had small-d
        # cases (e.g. d=0.4) that collapsed the y-range and produced a thin
        # band; replaced with Paul Bourke's classic set.
        if kind == "de_jong":
            presets = [
                (-2.0,  -2.0,  -1.2,   2.0),
                ( 1.4,  -2.3,   2.4,  -2.1),
                ( 2.01, -2.53,  1.61, -0.33),
                (-2.7,  -0.09, -0.86, -2.2),
                (-0.827, -1.637, 1.659, -0.943),
                (-2.24,  0.43, -0.65, -2.43),
            ]
        else:
            presets = [
                (-1.4,  1.6,   1.0,   0.7),
                (-1.7,  1.3,  -0.1,  -1.21),
                ( 1.5, -1.8,   1.6,   0.9),
                (-1.7, -1.7,  -1.1,  -1.5),
                ( 1.6, -0.6,  -1.2,   1.6),
            ]
        a, b, c, d = presets[seed % len(presets)]

        # Vectorize via parallel walkers. The point sequence is sequential
        # per walker, but running 1024 walkers in lockstep turns 220k Python
        # iterations of math.sin into ~220 numpy vector ops -- the attractor
        # is ergodic so the limit density is identical.
        n_walkers = 1024
        burn = 300
        n_sample = max(50, int(220 * float(self.density_boost)))
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, n_walkers).astype(np.float32)
        y = rng.uniform(-1.0, 1.0, n_walkers).astype(np.float32)

        def _step(x, y):
            if kind == "de_jong":
                xn = np.sin(a * y) - np.cos(b * x)
                yn = np.sin(c * x) - np.cos(d * y)
            else:
                xn = np.sin(a * y) + c * np.cos(a * x)
                yn = np.sin(b * x) + d * np.cos(b * y)
            return xn, yn

        for _ in range(burn):
            x, y = _step(x, y)

        xs = np.empty(n_walkers * n_sample, dtype=np.float32)
        ys = np.empty(n_walkers * n_sample, dtype=np.float32)
        for i in range(n_sample):
            x, y = _step(x, y)
            xs[i * n_walkers:(i + 1) * n_walkers] = x
            ys[i * n_walkers:(i + 1) * n_walkers] = y

        margin = s * 0.06
        span = s - 2 * margin
        px_lo, px_hi = float(np.percentile(xs, 2)), float(np.percentile(xs, 98))
        py_lo, py_hi = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
        px_spread = px_hi - px_lo or 1.0
        py_spread = py_hi - py_lo or 1.0
        cx = (xs - px_lo) / px_spread * span + margin
        cy = (ys - py_lo) / py_spread * span + margin
        ix = np.clip(cx.astype(np.int32), 0, s - 1)
        iy = np.clip(cy.astype(np.int32), 0, s - 1)

        density = np.zeros((s, s), dtype=np.float32)
        np.add.at(density, (iy, ix), 1.0)
        density = np.log1p(density)
        dmax = float(density.max()) or 1.0
        density = (density / dmax) ** 0.7

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.1 + 0.85 * (k / (LUT - 1))))
             for k in range(LUT)],
            dtype=np.uint8,
        )
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        idx = np.clip((density * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        rgb[density < 0.02] = bg

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
