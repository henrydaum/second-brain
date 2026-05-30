from plugins.BaseTechnique import BaseTechnique, Slider, Bool

import math
import numpy as np

try:
    art_kit
except NameError:
    art_kit = None


class ChromaticAberrationTechnique(BaseTechnique):
    name = 'Chromatic Aberration'
    description = 'Lens-fringe color separation — split R/G/B and offset each. Radial mode pushes channels outward from the center; uniform mode offsets along a fixed direction.'
    kind = "filter"

    strength = Slider(0, 30, default=6, step=1)
    radial   = Bool(default=True)
    angle    = Slider(0, 360, default=0, step=5)

    def run(self, canvas):
        amt = float(self.strength)
        arr = canvas.image_array(mode="RGB", dtype="float")
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        if self.radial:
            xx, yy, nx, ny = art_kit.centered_grid(canvas.size)
            length = np.sqrt(nx * nx + ny * ny) + 1e-6
            ux = nx / length
            uy = ny / length
            r_new = art_kit.bilinear_sample(r, xx + ux * amt, yy + uy * amt)
            b_new = art_kit.bilinear_sample(b, xx - ux * amt, yy - uy * amt)
            out = np.stack([r_new, g, b_new], axis=-1)
        else:
            a = math.radians(float(self.angle))
            dx = int(round(math.cos(a) * amt))
            dy = int(round(math.sin(a) * amt))
            r_new = np.roll(r, shift=(dy, dx), axis=(0, 1))
            b_new = np.roll(b, shift=(-dy, -dx), axis=(0, 1))
            out = np.stack([r_new, g, b_new], axis=-1)
        canvas.commit_array(out)
