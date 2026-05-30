from plugins.BaseTechnique import BaseTechnique, Slider, Pan, Palette

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class BuddhabrotTechnique(BaseTechnique):
    name = 'Buddhabrot'
    description = 'The Mandelbrot rendered backwards: sample random c values, iterate z = z^2 + c, and if the orbit escapes, plot every z it visited along the way into a 2D density buffer. The accumulated trajectories form a luminous, Buddha-like figure -- ghostly internal structure that the standard escape-time render hides. Monte Carlo, so more samples = smoother. Pan, zoom, and iteration controls let you frame the figure and dial up trajectory length; density controls how many samples to fire.'
    kind = "background"

    palette = Palette()
    iterations = Slider(100, 2000, default=1000, step=10, label='Iterations')
    density_boost = Slider(0.5, 5.0, default=1.0, step=0.1, label='Density')
    # Pan range ±3 (= 1.5 view-widths to either side) for consistency with the
    # other fractal explorers. Default view is the whole figure.
    pan_x = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan_y = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan = Pan(x='pan_x', y='pan_y', label='Pan')
    zoom_extra = Slider(0.5, 32.0, default=1.4, step=0.05, label='Zoom')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        n_iter = int(self.iterations)
        density = float(self.density_boost)
        zoom = float(self.zoom_extra)

        # View bounds in c-space (= same coords as the z trajectories we plot).
        # Center on (-0.5, 0) -- the Mandelbrot's natural center -- and frame
        # a 3-unit-wide window at zoom=1 so the whole figure fits.
        view_half = 1.5 / zoom
        view_cx = -0.5 + float(self.pan_x) * view_half
        view_cy = 0.0  + float(self.pan_y) * view_half

        rng = np.random.default_rng(seed)
        hist = np.zeros((s, s), dtype=np.float32)

        # Total samples drawn from the bounding box of the Mandelbrot set.
        # 2.5M gives an acceptably smooth default; Monte Carlo noise scales
        # as 1/sqrt(samples), so the slider runs to 12.5M for poster quality.
        total_samples = int(2_500_000 * density)
        chunk = 100_000
        n_chunks = max(1, (total_samples + chunk - 1) // chunk)
        compact_every = 8
        inv_view = 1.0 / (2.0 * view_half)

        for _ in range(n_chunks):
            # Sample c uniformly across the Mandelbrot bounding region. Sampling
            # the full c-plane instead of just the view box is important: orbits
            # of c values *outside* the view can still pass *through* the view,
            # contributing to the figure. That's most of the structure.
            c_re = rng.uniform(-2.1, 0.6, chunk).astype(np.float32)
            c_im = rng.uniform(-1.2, 1.2, chunk).astype(np.float32)

            # Cull cardioid + period-2 bulb -- these never escape, so they
            # contribute nothing and would burn the iteration budget.
            q = (c_re - 0.25) ** 2 + c_im * c_im
            in_cardioid = q * (q + (c_re - 0.25)) <= 0.25 * c_im * c_im
            in_bulb = (c_re + 1.0) ** 2 + c_im * c_im <= 0.0625
            keep = ~(in_cardioid | in_bulb)
            c_re = c_re[keep]
            c_im = c_im[keep]
            if c_re.size == 0:
                continue

            # PASS 1: detect which orbits escape inside n_iter. Compaction
            # drops escapers as they leave so the survivors (~99%, mostly
            # interior + unsampled non-escapees) don't drag the others.
            zr = np.zeros_like(c_re)
            zi = np.zeros_like(c_im)
            escape_iter = np.full(c_re.shape, n_iter + 1, dtype=np.int32)
            live_idx = np.arange(c_re.shape[0])
            live_cr = c_re.copy()
            live_ci = c_im.copy()

            for i in range(n_iter):
                zr_new = zr * zr - zi * zi + live_cr
                zi_new = 2.0 * zr * zi + live_ci
                zr = zr_new
                zi = zi_new
                if (i + 1) % compact_every == 0 or i == n_iter - 1:
                    absZ2 = zr * zr + zi * zi
                    esc = absZ2 > 4.0
                    if esc.any():
                        escape_iter[live_idx[esc]] = i + 1
                        keep2 = ~esc
                        zr = zr[keep2]
                        zi = zi[keep2]
                        live_cr = live_cr[keep2]
                        live_ci = live_ci[keep2]
                        live_idx = live_idx[keep2]
                        if live_idx.size == 0:
                            break

            escaping = escape_iter <= n_iter
            if not escaping.any():
                continue
            cr_esc = c_re[escaping]
            ci_esc = c_im[escaping]
            max_replay = int(escape_iter[escaping].max())

            # PASS 2: replay escaping orbits from z=0 and stamp every step
            # into the histogram up to (and including) the escape iteration.
            zr = np.zeros_like(cr_esc)
            zi = np.zeros_like(ci_esc)
            live_cr_p = cr_esc.copy()
            live_ci_p = ci_esc.copy()

            for i in range(max_replay):
                zr_new = zr * zr - zi * zi + live_cr_p
                zi_new = 2.0 * zr * zi + live_ci_p
                zr = zr_new
                zi = zi_new

                # Plot every live trajectory point. Buddhabrot's whole appeal
                # is that orbits of points OUTSIDE the view sweep through the
                # view, painting the figure -- so we don't filter by c-pos.
                px = ((zr - view_cx + view_half) * inv_view * s).astype(np.int32)
                py = ((zi - view_cy + view_half) * inv_view * s).astype(np.int32)
                ok = (px >= 0) & (px < s) & (py >= 0) & (py < s)
                if ok.any():
                    np.add.at(hist, (py[ok], px[ok]), 1.0)

                if (i + 1) % compact_every == 0 or i == max_replay - 1:
                    absZ2 = zr * zr + zi * zi
                    esc = absZ2 > 4.0
                    if esc.any():
                        keep3 = ~esc
                        zr = zr[keep3]
                        zi = zi[keep3]
                        live_cr_p = live_cr_p[keep3]
                        live_ci_p = live_ci_p[keep3]
                        if zr.size == 0:
                            break

        # Tone map: log compression to handle the long tail (a few very-hot
        # central pixels would otherwise swamp the rest), then a mild gamma.
        hmax = float(hist.max())
        if hmax > 0:
            density_norm = np.log1p(hist) / np.log1p(hmax)
            density_norm = density_norm ** 0.7
        else:
            density_norm = hist

        LUT = 512
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(0.05 + 0.92 * (k / (LUT - 1)))) for k in range(LUT)],
            dtype=np.uint8,
        )
        idx = np.clip((density_norm * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]

        # Empty regions go to background; the figure should sit on the palette
        # background rather than the LUT's first slot.
        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        rgb[hist < 0.5] = bg

        # Rotate 90° CW so the Buddha sits upright (head at top, body below,
        # arms reaching left/right) -- the conventional orientation. Using
        # k=1 (CCW) lands it head-down.
        rgb = np.rot90(rgb, k=-1)

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
