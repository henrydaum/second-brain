from plugins.BaseTechnique import BaseTechnique, Slider, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class ChladniPlateTechnique(BaseTechnique):
    name = 'Chladni Plate'
    description = 'Chladni figures / cymatics: the standing-wave nodal pattern of a vibrating square plate, where sand settles along the lines that do not move. Two eigenmodes superpose into the classic symmetric lattice of curves; the bright "sand" gathers on the nodes over a dark plate. Good for "Chladni plate", "cymatics", "standing waves", "nodal lines", "resonance", or a symmetric wave-interference pattern.'
    kind = "background"

    palette = Palette()
    mode_n  = Slider(1, 12, default=4, step=1)
    mode_m  = Slider(1, 12, default=7, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        n = float(int(self.mode_n))
        m = float(int(self.mode_m))

        lin = np.linspace(0.0, 1.0, s, dtype=np.float32)
        x, y = np.meshgrid(lin, lin)
        pi = np.pi

        def eigenmode(p, q):
            return (np.cos(p * pi * x) * np.cos(q * pi * y)
                    - np.cos(q * pi * x) * np.cos(p * pi * y))

        # Superpose several harmonics of the chosen mode so the nodal web is
        # far more intricate than a single eigenmode's lattice.
        field = (1.00 * eigenmode(n, m)
                 + 0.55 * eigenmode(n + 2.0, m + 3.0)
                 + 0.35 * eigenmode(n + 5.0, m + 1.0)
                 + 0.22 * eigenmode(n + 1.0, m + 6.0))
        field = (field / 2.12).astype(np.float32)

        # Two passes of "sand": tight bright cores plus a wider soft gather, so
        # the nodal lines read as ridges of settled grains rather than flat ink.
        sand = (0.65 * np.exp(-(field / 0.018) ** 2)
                + 0.45 * np.exp(-(field / 0.05) ** 2)).astype(np.float32)
        sand = np.clip(sand, 0.0, 1.0)

        # Fine grain texture so the sand looks physical up close.
        rng = np.random.default_rng(seed)
        grain = rng.random((s, s)).astype(np.float32)
        sand = sand * (0.72 + 0.28 * grain)

        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)
        sand_col = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.92)), dtype=np.float32)
        mid_col = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.4)), dtype=np.float32)
        # Faint mid-tone in the antinode bellies, bright sand on the nodes.
        belly = (0.16 * (1.0 - np.clip(sand, 0.0, 1.0)))[..., None]
        rgb = bg[None, None, :] * (1.0 - belly) + mid_col[None, None, :] * belly
        sa = np.clip(sand, 0.0, 1.0)[..., None]
        rgb = rgb * (1.0 - sa) + sand_col[None, None, :] * sa

        canvas.commit_array(rgb / 255.0)
