from plugins.BaseSkill import BaseSkill, Enum, Palette

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


class BurningShipExplorerSkill(BaseSkill):
    name = 'Burning Ship Explorer'
    description = 'A guided tour of the Burning Ship fractal -- the inferno cousin of the Mandelbrot set. Jagged, ship-like silhouettes with antennas, masts, and embedded mini-ships.'
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

    def run(self, canvas):
        cx, cy, zoom_exp, detail = _SPOTS[self.spot]
        s = canvas.size
        zoom = float(2.0 ** zoom_exp)
        n_iter = int(detail)

        use_f64 = zoom_exp > 10.0
        real = np.float64 if use_f64 else np.float32

        view = 3.0 / zoom
        half = view * 0.5
        re = np.linspace(cx - half, cx + half, s, dtype=real)
        im = np.linspace(cy + half, cy - half, s, dtype=real)
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
