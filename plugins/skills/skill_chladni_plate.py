from plugins.BaseSkill import BaseSkill, Slider, Palette

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None


class ChladniPlateSkill(BaseSkill):
    name = 'Chladni Plate'
    description = 'Chladni figures / cymatics: the standing-wave nodal pattern of a vibrating square plate, where sand settles along the lines that do not move. Two eigenmodes superpose into the classic symmetric lattice of curves; the bright "sand" gathers on the nodes over a dark plate. Good for "Chladni plate", "cymatics", "standing waves", "nodal lines", "resonance", or a symmetric wave-interference pattern.'
    kind = "background"

    palette = Palette()
    mode_n  = Slider(1, 12, default=4, step=1)
    mode_m  = Slider(1, 12, default=7, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        n = float(int(self.mode_n))
        m = float(int(self.mode_m))

        lin = np.linspace(0.0, 1.0, s, dtype=np.float32)
        x, y = np.meshgrid(lin, lin)
        pi = np.pi
        # Superposed square-plate eigenmodes (symmetric combination).
        field = (np.cos(n * pi * x) * np.cos(m * pi * y)
                 - np.cos(m * pi * x) * np.cos(n * pi * y))

        # Sand collects where the plate is still (field ~ 0): bright nodal lines.
        eps = 0.06
        sand = np.exp(-(field / eps) ** 2).astype(np.float32)

        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.float32)
        sand_col = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.9)), dtype=np.float32)
        mid_col = np.array(art_kit.hex_to_rgb(art_kit.palette_color(0.45)), dtype=np.float32)
        # Faint mid-tone for the antinode bellies, bright sand on the nodes.
        belly = (0.18 * (1.0 - sand))[..., None]
        rgb = bg[None, None, :] * (1.0 - belly) + mid_col[None, None, :] * belly
        rgb = rgb * (1.0 - sand[..., None]) + sand_col[None, None, :] * sand[..., None]

        canvas.commit_array(rgb / 255.0)
