from plugins.BaseSkill import BaseSkill

import math
import numpy as np

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class ChromaticAberrationSkill(BaseSkill):
    name = 'Chromatic Aberration'
    description = 'Lens-fringe color separation — split R/G/B and offset each. Radial mode pushes channels outward from the center (camera-lens look). Uniform mode offsets all channels along a fixed direction (CRT/print misregistration look). Params: strength (0-30 px, default 6), radial bool, angle (0-360, only used when radial=False).'
    kind = 'transform'
    owner = 'library'
    created_at = 1730000000.0
    hidden = False
    controls = [
        {'type': 'slider', 'name': 'strength', 'label': 'Strength', 'min': 0, 'max': 30, 'step': 1, 'default': 6},
        {'type': 'bool', 'name': 'radial', 'label': 'Radial', 'default': True},
        {'type': 'slider', 'name': 'angle', 'label': 'Angle', 'min': 0, 'max': 360, 'step': 5, 'default': 0},
    ]

    def run(self, canvas, strength=6, radial=True, angle=0):
        amt = float(art_kit.clamp(strength, 0, 60))
        arr = canvas.image_array(mode="RGB", dtype="float")
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        if bool(radial):
            xx, yy, nx, ny = art_kit.centered_grid(canvas.size)
            length = np.sqrt(nx * nx + ny * ny) + 1e-6
            ux = nx / length
            uy = ny / length
            r_new = art_kit.bilinear_sample(r, xx + ux * amt, yy + uy * amt)
            b_new = art_kit.bilinear_sample(b, xx - ux * amt, yy - uy * amt)
            out = np.stack([r_new, g, b_new], axis=-1)
        else:
            a = math.radians(float(art_kit.clamp(angle, 0, 360)))
            dx = int(round(math.cos(a) * amt))
            dy = int(round(math.sin(a) * amt))
            r_new = np.roll(r, shift=(dy, dx), axis=(0, 1))
            b_new = np.roll(b, shift=(-dy, -dx), axis=(0, 1))
            out = np.stack([r_new, g, b_new], axis=-1)
        canvas.commit_array(out)
