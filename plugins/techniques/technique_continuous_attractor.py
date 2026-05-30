from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import math
import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None

def _rossler_trajectory(n, seed):
    rng = np.random.default_rng(seed)
    a, b, c = 0.2, 0.2, 5.7  # canonical Rossler producing a clean ribbon
    dt = 0.02

    x = float(0.1 + rng.uniform(-0.05, 0.05))
    y = float(0.0)
    z = float(0.0)
    for _ in range(2000):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
    xs = np.empty(n, dtype=np.float32)
    ys = np.empty(n, dtype=np.float32)
    zs = np.empty(n, dtype=np.float32)
    for i in range(n):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        xs[i] = x
        ys[i] = y
        zs[i] = z
    return xs, ys, zs

def _pickover_trajectory(n, seed):
    # 3D discrete Pickover map. Several known-stable presets, chosen by seed.
    presets = [
        (-0.759494, 2.449138,  1.253602, 1.875347),
        (2.07,     -2.36,      1.20,    -0.66),
        (-1.4,     -1.1,       0.65,    -1.0),
        (-1.7,     -1.3,       0.50,     1.5),
    ]
    a, b, c, d = presets[int(seed) % len(presets)]
    x, y, z = 0.1, 0.1, 0.1
    for _ in range(200):
        x_n = math.sin(a * y) - z * math.cos(b * x)
        y_n = z * math.sin(c * x) - math.cos(d * y)
        z_n = math.sin(x)
        x, y, z = x_n, y_n, z_n
    xs = np.empty(n, dtype=np.float32)
    ys = np.empty(n, dtype=np.float32)
    zs = np.empty(n, dtype=np.float32)
    for i in range(n):
        x_n = math.sin(a * y) - z * math.cos(b * x)
        y_n = z * math.sin(c * x) - math.cos(d * y)
        z_n = math.sin(x)
        x, y, z = x_n, y_n, z_n
        xs[i] = x
        ys[i] = y
        zs[i] = z
    return xs, ys, zs


class ContinuousAttractorTechnique(BaseTechnique):
    name = 'Continuous Attractor'
    description = 'Strange attractors beyond Lorenz: the Rossler system (dx=-y-z, dy=x+a*y, dz=b+z*(x-c)) which folds into a single ribbon, and the 3D Pickover map projected to a woven cloud. 200,000 integration steps, log-compressed density, palette-graded by depth (z) so the structure reads three-dimensional. Distinct from the Lorenz butterfly and the discrete de Jong / Clifford clouds. Good for "rossler", "pickover", "strange attractor", "chaos", "ribbon", or "woven".'
    kind = "background"

    palette = Palette()
    system  = Enum([('rossler', 'Rossler'), ('pickover', 'Pickover')], default='rossler')

    def run(self, canvas):
        s = canvas.size
        seed = canvas.seed
        system = self.system

        n = 200_000
        if system == "pickover":
            xs, ys, zs = _pickover_trajectory(n, seed)
        else:
            xs, ys, zs = _rossler_trajectory(n, seed)

        margin = s * 0.06
        span = s - 2 * margin
        x_lo, x_hi = float(np.percentile(xs, 1)), float(np.percentile(xs, 99))
        y_lo, y_hi = float(np.percentile(ys, 1)), float(np.percentile(ys, 99))
        z_lo, z_hi = float(np.percentile(zs, 1)), float(np.percentile(zs, 99))
        cx = (xs - x_lo) / ((x_hi - x_lo) or 1.0) * span + margin
        cy = (ys - y_lo) / ((y_hi - y_lo) or 1.0) * span + margin
        cy = s - cy
        z_t = np.clip((zs - z_lo) / ((z_hi - z_lo) or 1.0), 0.0, 1.0)

        # 3x3 splat so the trajectory has weight at full resolution.
        density = np.zeros((s, s), dtype=np.float32)
        z_acc = np.zeros((s, s), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = 1.0 if (dy == 0 and dx == 0) else 0.5
                ix2 = np.clip(cx.astype(np.int32) + dx, 0, s - 1)
                iy2 = np.clip(cy.astype(np.int32) + dy, 0, s - 1)
                np.add.at(density, (iy2, ix2), w)
                np.add.at(z_acc, (iy2, ix2), z_t * w)

        safe = density > 0
        z_mean = np.zeros_like(z_acc)
        z_mean[safe] = z_acc[safe] / density[safe]

        density = np.log1p(density)
        dmax = float(density.max()) or 1.0
        density = (density / dmax) ** 0.55

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.18 + 0.78 * (k / (LUT - 1))))
             for k in range(LUT)],
            dtype=np.uint8,
        )
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        t_field = np.clip(density * 0.7 + z_mean * 0.3, 0.0, 1.0)
        idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        rgb[density < 0.04] = bg

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
