from plugins.BaseTechnique import BaseTechnique, Enum, Palette, Slider, Pan

import numpy as np
from PIL import Image

try:
    art_kit
except NameError:
    art_kit = None

_SPOTS = {
    "full":      (-0.5,      -0.5,      0.0,  220),
    "main_ship": (-1.762,    -0.028,    5.0,  300),
    "antenna":   (-1.625,    -0.0085,   7.5,  400),
    "mini_ship": (-1.7755,   -0.0335,  10.0,  500),
    "mast":      (-1.7395,   -0.000045,11.0,  500),
    "deep_keel": (-1.76225,  -0.03415, 13.0,  600),
}


class BurningShipExplorerTechnique(BaseTechnique):
    name = 'Burning Ship Explorer'
    description = 'A guided tour of the Burning Ship fractal -- the inferno cousin of the Mandelbrot set. Jagged, ship-like silhouettes with antennas, masts, and embedded mini-ships. Pan, zoom, and iteration controls let you wander off the preset and chase your own detail. Optimized: paired-real working set (the |Re|/|Im| step doesn\'t fit numpy complex), live-buffer compaction every 3 iterations.'
    kind = "background"

    palette = Palette()
    spot    = Enum([
        ('full',      'Full Set'),
        ('main_ship', 'Main Ship'),
        ('antenna',   'Antenna'),
        ('mini_ship', 'Embedded Mini-Ship'),
        ('mast',      'Mast Spire'),
        ('deep_keel', 'Deep Keel'),
    ], default='main_ship')
    # Pan range ±3 (= 1.5 view-widths to either side) so you can navigate
    # between adjacent features at deep zoom; capping at ±1 only let you
    # move by half a screen, which felt like nothing once zoomed in.
    pan_x = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan_y = Slider(-3.0, 3.0, default=0.0, step=0.05)
    pan = Pan(x='pan_x', y='pan_y', label='Pan')
    zoom_extra = Slider(0.5, 32.0, default=1.0, step=0.05, label='Zoom')
    iterations = Slider(0, 1500, default=1500, step=10, label='Iterations')

    def run(self, canvas):
        cx, cy, zoom_exp, detail = _SPOTS.get(str(self.spot), _SPOTS["main_ship"])
        s = canvas.size
        zoom = float(2.0 ** zoom_exp) * float(self.zoom_extra)
        iter_override = int(self.iterations)
        n_iter = iter_override if iter_override >= 50 else int(detail)

        # Step up to f64 once total zoom passes ~2^10, regardless of whether
        # the extra zoom came from the preset or the user slider.
        use_f64 = zoom > 1024.0
        real = np.float64 if use_f64 else np.float32

        view = 3.0 / zoom
        half = view * 0.5
        # Pan offsets the view center; ±1 on the slider moves by a full half-view.
        cx_eff = cx + float(self.pan_x) * half
        cy_eff = cy + float(self.pan_y) * half
        re = np.linspace(cx_eff - half, cx_eff + half, s, dtype=real)
        # im linspace is reversed so the ship silhouette reads right-side up.
        im = np.linspace(cy_eff + half, cy_eff - half, s, dtype=real)
        R, I = np.meshgrid(re, im)

        N = s * s
        out_flat = np.zeros(N, dtype=np.float64)
        inside_flat = np.zeros(N, dtype=bool)

        zr = np.zeros(N, dtype=real)
        zi = np.zeros(N, dtype=real)
        cr = R.ravel().copy()
        ci = I.ravel().copy()

        live_idx = np.arange(N)
        bailout2 = real(16 * 16) if not use_f64 else real(1 << 16)
        inv_log2 = 1.0 / np.log(2.0)
        compact_every = 3

        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(n_iter):
                azr = np.abs(zr)
                azi = np.abs(zi)
                zr_new = azr * azr - azi * azi + cr
                zi_new = 2.0 * azr * azi + ci
                zr = zr_new
                zi = zi_new
                if (i + 1) % compact_every == 0 or i == n_iter - 1:
                    absZ2 = zr * zr + zi * zi
                    esc = absZ2 > bailout2
                    if esc.any():
                        log_mod = 0.5 * np.log(absZ2[esc].astype(np.float64))
                        nu = np.log(log_mod * inv_log2) * inv_log2
                        out_flat[live_idx[esc]] = (i + 1) - nu
                        keep = ~esc
                        live_idx = live_idx[keep]
                        zr = zr[keep]
                        zi = zi[keep]
                        cr = cr[keep]
                        ci = ci[keep]
                        if live_idx.size == 0:
                            break

        inside_flat[live_idx] = True

        valid = ~inside_flat
        t = np.zeros_like(out_flat)
        if valid.any():
            v = np.log(out_flat[valid] + 1.0)
            vmin = float(v.min())
            vmax = float(v.max())
            if vmax - vmin > 1e-9:
                v = (v - vmin) / (vmax - vmin)
            else:
                v = np.zeros_like(v)
            t[valid] = v

        LUT_SIZE = 512
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k / (LUT_SIZE - 1))) for k in range(LUT_SIZE)],
            dtype=np.uint8,
        )
        idx_lut = np.clip((t * (LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
        rgb = lut[idx_lut].reshape(s, s, 3)

        bg = np.array(art_kit.hex_to_rgb(canvas.palette.background), dtype=np.uint8)
        rgb[inside_flat.reshape(s, s)] = bg

        canvas.commit(Image.fromarray(rgb, "RGB").convert("RGBA"))
