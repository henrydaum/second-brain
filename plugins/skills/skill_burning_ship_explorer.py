from plugins.BaseSkill import BaseSkill

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None


class BurningShipExplorerSkill(BaseSkill):
    name = 'Burning Ship Explorer'
    description = 'A guided tour of the Burning Ship fractal -- the inferno cousin of the Mandelbrot set. The iteration takes absolute values before squaring (z = (|Re| + i|Im|)^2 + c), which breaks holomorphic symmetry and yields jagged, ship-like silhouettes with antennas, masts, and embedded mini-ships. Pick a landmark and pair with any palette. Good for "fractal", "ship", "burning", "inferno", "jagged", or any algorithmic flame-and-iron motif.'
    kind = 'creation'
    owner = 'library'
    created_at = 1779667200.0
    hidden = False
    controls = [{'type': 'enum', 'name': 'spot', 'label': 'Spot', 'options': [{'value': 'full', 'label': 'Full Set'}, {'value': 'main_ship', 'label': 'Main Ship'}, {'value': 'antenna', 'label': 'Antenna'}, {'value': 'mini_ship', 'label': 'Embedded Mini-Ship'}, {'value': 'mast', 'label': 'Mast Spire'}, {'value': 'deep_keel', 'label': 'Deep Keel'}], 'default': 'main_ship'}, {'type': 'palette', 'name': 'palette', 'label': 'Palette'}]

    _SPOTS = {
        "full":      (-0.5,      -0.5,      0.0,  220),  # whole armada
        "main_ship": (-1.762,    -0.028,    5.0,  300),  # the big iconic ship
        "antenna":   (-1.625,    -0.0085,   7.5,  400),  # vertical mast above main ship
        "mini_ship": (-1.7755,   -0.0335,  10.0,  500),  # tiny embedded ship in the main hull
        "mast":      (-1.7395,   -0.000045,11.0,  500),  # thin top of the antenna
        "deep_keel": (-1.76225,  -0.03415, 13.0,  600),  # deep-zoom keel detail
    }

    def run(self, canvas, spot="main_ship", **_):
        cx, cy, zoom_exp, detail = _SPOTS.get(str(spot), _SPOTS["main_ship"])
        s = int(canvas.size)
        zoom = float(2.0 ** zoom_exp)
        n_iter = int(detail)

        use_f64 = zoom_exp > 10.0
        real = np.float64 if use_f64 else np.float32

        view = 3.0 / zoom
        half = view * 0.5
        re = np.linspace(cx - half, cx + half, s, dtype=real)
        # Flip y so the ship reads right-side up on screen.
        im = np.linspace(cy + half, cy - half, s, dtype=real)
        R, I = np.meshgrid(re, im)

        N = s * s
        out_flat = np.zeros(N, dtype=np.float64)
        inside_flat = np.zeros(N, dtype=bool)

        # Track real / imag parts separately so we can take abs each step.
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
