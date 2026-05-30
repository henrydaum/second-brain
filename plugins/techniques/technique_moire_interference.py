from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class MoireInterferenceTechnique(BaseTechnique):
    name = 'Moire Interference'
    description = 'Sum of plane waves: field(x,y) = sum_i A_i * sin(k_i * (x*cos(theta_i) + y*sin(theta_i)) + phi_i). Two waves with nearly the same theta produce slow moire beats; orthogonal waves give crosshatch; mixing in a radial wave bends the linear interference into rosettes. Four named presets dial in distinct interference geometries; the summed field is normalized and pushed through a palette ramp. Good for "interference", "moire", "waves", "beats", "crosshatch", "rosette", or any optical-pattern algorithmic motif.'
    kind = "background"
    palette = Palette()
    pattern = Enum([('beats', 'Parallel Beats'), ('crosshatch', 'Crosshatch'), ('rosette', 'Radial Rosette'), ('turbulent', 'Turbulent')], default='beats')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        self.pattern = str(self.pattern)
        rng = np.random.default_rng(seed)

        ys, xs = np.mgrid[0:s, 0:s].astype(np.float32)
        cx = s / 2.0
        cy = s / 2.0
        # Normalize coordinates to roughly [-1, 1] so wavelength constants stay scale-free.
        nx = (xs - cx) / (s * 0.5)
        ny = (ys - cy) / (s * 0.5)

        field = np.zeros((s, s), dtype=np.float32)
        n_waves = 0

        if self.pattern == "beats":
            # Three waves: two nearly parallel (very close theta) -> visible moire,
            # plus one orthogonal at a different wavelength.
            base_theta = rng.uniform(0, math.pi)
            for dt, wl in ((0.00, 0.10), (0.06, 0.10), (math.pi / 2.0, 0.18)):
                theta = base_theta + dt
                k = 2 * math.pi / wl
                phi = rng.uniform(0, 1)
                field += np.sin(k * (nx * math.cos(theta) + ny * math.sin(theta)) + phi * 2 * math.pi)
                n_waves += 1
        elif self.pattern == "crosshatch":
            for dt in (0.0, math.pi / 2.0, math.pi / 4.0, 3 * math.pi / 4.0):
                wl = float(rng.uniform(0.10, 0.18))
                k = 2 * math.pi / wl
                phi = float(rng.uniform(0, 1))
                field += np.sin(k * (nx * math.cos(dt) + ny * math.sin(dt)) + phi * 2 * math.pi)
                n_waves += 1
        elif self.pattern == "rosette":
            # One radial wave + three plane waves.
            r = np.sqrt(nx * nx + ny * ny)
            wl_r = 0.10
            field += np.sin(2 * math.pi * (r / wl_r))
            n_waves += 1
            for _ in range(3):
                theta = float(rng.uniform(0, math.pi))
                wl = float(rng.uniform(0.12, 0.22))
                k = 2 * math.pi / wl
                phi = float(rng.uniform(0, 1))
                field += np.sin(k * (nx * math.cos(theta) + ny * math.sin(theta)) + phi * 2 * math.pi)
                n_waves += 1
        else:  # turbulent
            for _ in range(8):
                theta = float(rng.uniform(0, math.pi))
                wl = float(rng.uniform(0.06, 0.30))
                k = 2 * math.pi / wl
                phi = float(rng.uniform(0, 1))
                field += np.sin(k * (nx * math.cos(theta) + ny * math.sin(theta)) + phi * 2 * math.pi)
                n_waves += 1

        field /= max(1, n_waves)
        field = (field + 1.0) * 0.5  # to [0,1]
        field = field * field * (3.0 - 2.0 * field)  # smoothstep contrast

        LUT = 512
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.05 + 0.9 * (k / (LUT - 1))))
             for k in range(LUT)],
            dtype=np.uint8,
        )
        idx = np.clip((field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
