from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class LorenzAttractorTechnique(BaseTechnique):
    name = 'Lorenz Attractor'
    description = 'The Lorenz system integrated with explicit Euler -- dx=sigma*(y-x), dy=x*(rho-z)-y, dz=x*y-beta*z with sigma=10, rho=28, beta=8/3 -- traced for 200,000 steps. Project the 3D trajectory onto xy, xz (the canonical butterfly), or yz and accumulate into a 2D density buffer, log-compress, palette-map. The time-ordered palette gradient lets you read the flow direction around the two lobes. Good for "lorenz", "butterfly", "strange attractor", "chaos", or any continuous-dynamical-systems motif.'
    kind = "background"
    palette = Palette()
    projection = Enum([('xz', 'XZ (butterfly)'), ('xy', 'XY (front)'), ('yz', 'YZ (side)')], default='xz')
    zoom = Slider(0.4, 2.0, default=0.85, step=0.05, label='Zoom')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        self.projection = str(self.projection)
        rng = np.random.default_rng(seed)

        n_steps = 200_000
        dt = 0.005
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

        # Start from a known-good point near the attractor with tiny jitter for
        # per-seed variety. Lorenz is globally bounded for these constants, so
        # NaN/Inf during burn-in means numerical drift; fall back to the canonical
        # (1, 1, 1) seed which is well-tested.
        def _burn(x, y, z, steps):
            for _ in range(steps):
                dx = sigma * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
                x += dx * dt; y += dy * dt; z += dz * dt
            return x, y, z
        x = 1.0 + float(rng.uniform(-0.04, 0.04))
        y = 1.0 + float(rng.uniform(-0.04, 0.04))
        z = 1.0 + float(rng.uniform(-0.04, 0.04))
        x, y, z = _burn(x, y, z, 2000)
        if not all(np.isfinite([x, y, z])):
            x, y, z = _burn(1.0, 1.0, 1.0, 2000)

        xs = np.empty(n_steps, dtype=np.float32)
        ys = np.empty(n_steps, dtype=np.float32)
        zs = np.empty(n_steps, dtype=np.float32)
        # Burn-in 2000 steps so we drop the transient approach onto the attractor.
        for _ in range(2000):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
        for i in range(n_steps):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            xs[i] = x
            ys[i] = y
            zs[i] = z

        if self.projection == "xy":
            u, v = xs, ys
        elif self.projection == "yz":
            u, v = ys, zs
        else:
            u, v = xs, zs

        # Robust normalization using percentile bounds (same as attractor_cloud).
        margin = s * 0.06
        span = s - 2 * margin
        u_lo, u_hi = float(np.percentile(u, 1)), float(np.percentile(u, 99))
        v_lo, v_hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
        u_spread = (u_hi - u_lo) or 1.0
        v_spread = (v_hi - v_lo) or 1.0
        # Zoom scales the projected attractor about the canvas center; default
        # 0.85 because the percentile-fit tends to push the lobes against the
        # canvas edges, which crops the curl of the trajectory near the wings.
        zoom = float(self.zoom)
        effective_span = span * zoom
        center = s * 0.5
        cx = (u - u_lo) / u_spread * effective_span + (center - effective_span * 0.5)
        cy = (v - v_lo) / v_spread * effective_span + (center - effective_span * 0.5)
        # Flip y so that increasing z reads upward visually for the xz projection.
        cy = s - cy

        # Splat with a 3x3 cross stamp so the trajectory has weight at 1024^2 even
        # though we only integrate ~200k points.
        density = np.zeros((s, s), dtype=np.float32)
        time_acc = np.zeros((s, s), dtype=np.float32)
        t_axis = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                w = 1.0 if (dy == 0 and dx == 0) else 0.5
                ix2 = np.clip(cx.astype(np.int32) + dx, 0, s - 1)
                iy2 = np.clip(cy.astype(np.int32) + dy, 0, s - 1)
                np.add.at(density, (iy2, ix2), w)
                np.add.at(time_acc, (iy2, ix2), t_axis * w)

        safe = density > 0
        time_mean = np.zeros_like(time_acc)
        time_mean[safe] = time_acc[safe] / density[safe]

        density = np.log1p(density)
        dmax = float(density.max()) or 1.0
        density = (density / dmax) ** 0.55

        LUT = 256
        # Density picks the LUT index; time-of-visit slightly shifts the ramp
        # position so the two lobes read as different "ages" of the flow.
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.18 + 0.78 * (k / (LUT - 1))))
             for k in range(LUT)],
            dtype=np.uint8,
        )
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        t_field = np.clip(density * 0.7 + time_mean * 0.3, 0.0, 1.0)
        idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        rgb[density < 0.04] = bg

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
